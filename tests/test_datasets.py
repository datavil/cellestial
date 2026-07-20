import urllib.request

import anndata as ad
import numpy as np
import pytest

from cellestial.datasets import datasets


def _small_adata() -> ad.AnnData:
    data = ad.AnnData(X=np.ones((4, 4)))
    data.obs_names = [f"cell_{i}" for i in range(4)]
    data.var_names = ["MT-ND1", "RPS3", "HBA1", "CD3D"]
    return data


def test_resolve_cache_file_bring_and_remove(tmp_path, monkeypatch):
    global_cache = tmp_path / "global"
    local_cache = tmp_path / "local"
    global_cache.mkdir()
    (global_cache / "example.h5ad").write_text("cached")
    monkeypatch.setattr(datasets, "_GLOBAL_CACHE", global_cache)

    cache_file = datasets._resolve_cache_file(
        local_cache,
        "example.h5ad",
        use_cache=True,
        bring=True,
    )
    assert cache_file.read_text() == "cached"

    removed = datasets._resolve_cache_file(
        local_cache,
        "example.h5ad",
        use_cache=False,
        bring=True,
    )
    assert removed == cache_file
    assert not cache_file.exists()


@pytest.mark.parametrize(
    ("loader", "expected_filename"),
    [
        (datasets.pbmc3k, "pbmc3k_pped.h5ad"),
        (datasets.pancreas, "endocrinogenesis_day15_pped.h5ad"),
        (datasets.human_lymph_node, "V1_Human_Lymph_Node_pped.h5ad"),
    ],
)
def test_bundled_loader_delegates_to_from_url(loader, expected_filename, tmp_path, monkeypatch):
    # The bundled datasets are now downloaded pre-preprocessed, so each loader is
    # a thin wrapper that forwards its caching args to `from_url` with a fixed URL.
    calls = {}

    def fake_from_url(url, *, cache_directory, use_cache, bring):
        calls.update(url=url, cache_directory=cache_directory, use_cache=use_cache, bring=bring)
        return _small_adata()

    monkeypatch.setattr(datasets, "from_url", fake_from_url)

    data = loader(cache_directory=tmp_path, use_cache=False, bring=False)

    assert isinstance(data, ad.AnnData)
    assert calls["url"].endswith(expected_filename)
    assert calls["cache_directory"] == tmp_path
    assert calls["use_cache"] is False
    assert calls["bring"] is False


def test_from_url_downloads_and_reads(tmp_path, monkeypatch):
    read_data = _small_adata()
    monkeypatch.setattr(datasets, "read_h5ad", lambda path: read_data)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, block_size):
            chunks = [b"abc", b"def", b""]
            return chunks.pop(0)

    chunks = [b"abc", b"def", b""]

    class StatefulResponse(FakeResponse):
        def __init__(self):
            self.headers = {"Content-Length": "6"}

        def read(self, block_size):
            return chunks.pop(0)

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _url, **_kwargs: StatefulResponse(),
    )

    data = datasets.from_url(
        "https://example.test/example.h5ad",
        cache_directory=tmp_path,
        use_cache=False,
        bring=False,
    )

    assert data is read_data
    assert (tmp_path / "example.h5ad").read_bytes() == b"abcdef"


def test_from_url_requires_name_for_pathless_url(tmp_path):
    with pytest.raises(ValueError, match="derive a filename"):
        datasets.from_url("https://example.test", cache_directory=tmp_path)


def test_from_url_removes_partial_file_on_incomplete_download(tmp_path, monkeypatch):
    class ShortResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, block_size):
            chunks = [b"abc", b""]
            return chunks.pop(0)

    chunks = [b"abc", b""]

    class StatefulShortResponse(ShortResponse):
        def __init__(self):
            self.headers = {"Content-Length": "6"}

        def read(self, block_size):
            return chunks.pop(0)

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _url, **_kwargs: StatefulShortResponse(),
    )

    with pytest.raises(OSError, match="Download incomplete"):
        datasets.from_url(
            "https://example.test/example.h5ad",
            cache_directory=tmp_path,
            use_cache=False,
            bring=False,
        )

    assert not (tmp_path / "example.h5ad.part").exists()


def test_from_cellxgene_builds_download_url(tmp_path, monkeypatch):
    calls = {}

    def fake_from_url(url, *, name, cache_directory, use_cache, bring):
        calls.update(
            url=url,
            name=name,
            cache_directory=cache_directory,
            use_cache=use_cache,
            bring=bring,
        )
        return _small_adata()

    monkeypatch.setattr(datasets, "from_url", fake_from_url)
    uuid = "7cdea341-ca7a-40fd-8192-b8ecb2d7b91e"

    data = datasets.from_cellxgene(uuid, cache_directory=tmp_path, use_cache=False, bring=False)

    assert isinstance(data, ad.AnnData)
    assert calls["url"] == f"https://datasets.cellxgene.cziscience.com/{uuid}.h5ad"
    assert calls["name"] == uuid


def test_from_cellxgene_accepts_url_and_rejects_invalid(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        datasets,
        "from_url",
        lambda url, **kwargs: calls.setdefault("url", url) or _small_adata(),
    )
    url = "https://datasets.cellxgene.cziscience.com/7cdea341-ca7a-40fd-8192-b8ecb2d7b91e.h5ad"

    datasets.from_cellxgene(url, name="custom", cache_directory=tmp_path)

    assert calls["url"] == url
    with pytest.raises(ValueError, match="CELLxGENE"):
        datasets.from_cellxgene("not-a-uuid", cache_directory=tmp_path)


def test_breast_cancer_atlas_uses_named_cellxgene_dataset(tmp_path, monkeypatch):
    calls = {}

    def fake_from_cellxgene(source, *, name, cache_directory, use_cache, bring):
        calls.update(source=source, name=name, cache_directory=cache_directory)
        return _small_adata()

    monkeypatch.setattr(datasets, "from_cellxgene", fake_from_cellxgene)

    data = datasets.breast_cancer_atlas(cache_directory=tmp_path)

    assert isinstance(data, ad.AnnData)
    assert calls["name"] == "breast_cancer_atlas"
    assert "7cdea341" in calls["source"]
