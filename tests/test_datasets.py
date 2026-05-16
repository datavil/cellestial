import sys
import types
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


def _run_steps_immediately(steps, desc):
    assert desc.startswith("Preprocessing")
    for _, _, action in steps:
        action()


def test_require_extras_reports_missing_module(monkeypatch):
    monkeypatch.setattr(datasets.importlib.util, "find_spec", lambda module: None)

    with pytest.raises(ImportError, match="requires optional dependencies"):
        datasets._require_extras("demo", "missing")


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


def test_run_steps_executes_actions():
    calls = []

    datasets._run_steps(
        [
            ("first", "one()", lambda: calls.append("one")),
            ("second", "two()", lambda: calls.append("two")),
        ],
        desc="Preprocessing {demo}",
    )

    assert calls == ["one", "two"]


def test_pbmc3k_preprocess_pipeline_with_fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "_require_extras", lambda *args: None)
    monkeypatch.setattr(datasets, "_run_steps", _run_steps_immediately)

    class FakeRegistry:
        def load_registry_from_doi(self):
            self.loaded = True

        def fetch(self, filename):
            return filename

    fake_pooch = types.SimpleNamespace(
        create=lambda path, base_url: FakeRegistry(),
        os_cache=lambda name: str(tmp_path / name),
    )

    def read_10x_h5(path):
        data = _small_adata()
        data.obs["source"] = path
        return data

    def leiden(data, key_added="leiden", resolution=None, **kwargs):
        data.obs[key_added] = ["0", "1", "2", "3", "0", "1", "2", "3"][: data.n_obs]

    fake_scanpy = types.SimpleNamespace(
        read_10x_h5=read_10x_h5,
        pp=types.SimpleNamespace(
            calculate_qc_metrics=lambda *args, **kwargs: None,
            filter_cells=lambda *args, **kwargs: None,
            filter_genes=lambda *args, **kwargs: None,
            scrublet=lambda *args, **kwargs: None,
            normalize_total=lambda *args, **kwargs: None,
            log1p=lambda *args, **kwargs: None,
            highly_variable_genes=lambda *args, **kwargs: None,
            neighbors=lambda *args, **kwargs: None,
        ),
        tl=types.SimpleNamespace(
            pca=lambda *args, **kwargs: None,
            umap=lambda *args, **kwargs: None,
            tsne=lambda *args, **kwargs: None,
            leiden=leiden,
        ),
    )

    monkeypatch.setitem(sys.modules, "pooch", fake_pooch)
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)

    data = datasets.pbmc3k(cache_directory=tmp_path, use_cache=False, bring=False)

    assert isinstance(data, ad.AnnData)
    assert "counts" in data.layers
    assert "cell_type_lvl1" in data.obs
    assert (tmp_path / "pbmc3k_pped.h5ad").exists()


def test_pancreas_preprocess_pipeline_with_fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "_require_extras", lambda *args: None)
    monkeypatch.setattr(datasets, "_run_steps", _run_steps_immediately)

    fake_scanpy = types.SimpleNamespace(
        pp=types.SimpleNamespace(neighbors=lambda *args, **kwargs: None),
    )
    fake_scvelo = types.SimpleNamespace(
        datasets=types.SimpleNamespace(pancreas=lambda raw_file: _small_adata()),
        pp=types.SimpleNamespace(
            filter_and_normalize=lambda *args, **kwargs: None,
            moments=lambda *args, **kwargs: None,
        ),
        tl=types.SimpleNamespace(
            velocity=lambda *args, **kwargs: None,
            velocity_graph=lambda *args, **kwargs: None,
            velocity_embedding=lambda *args, **kwargs: None,
        ),
    )

    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)
    monkeypatch.setitem(sys.modules, "scvelo", fake_scvelo)

    data = datasets.pancreas(cache_directory=tmp_path, use_cache=False, bring=False)

    assert isinstance(data, ad.AnnData)
    assert (tmp_path / "endocrinogenesis_day15_pped.h5ad").exists()


def test_human_lymph_node_preprocess_pipeline_with_fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "_require_extras", lambda *args: None)
    monkeypatch.setattr(datasets, "_run_steps", _run_steps_immediately)

    def visium_sge(sample_id):
        data = _small_adata()
        data.obs["pct_counts_mt"] = [5.0, 10.0, 25.0, 1.0]
        return data

    def calculate_qc_metrics(data, qc_vars, inplace):
        data.obs["pct_counts_mt"] = [5.0, 10.0, 25.0, 1.0]

    def leiden(data, key_added, **kwargs):
        data.obs[key_added] = [str(i) for i in range(data.n_obs)]

    fake_scanpy = types.SimpleNamespace(
        datasets=types.SimpleNamespace(visium_sge=visium_sge),
        pp=types.SimpleNamespace(
            calculate_qc_metrics=calculate_qc_metrics,
            filter_cells=lambda *args, **kwargs: None,
            filter_genes=lambda *args, **kwargs: None,
            normalize_total=lambda *args, **kwargs: None,
            log1p=lambda *args, **kwargs: None,
            highly_variable_genes=lambda *args, **kwargs: None,
            pca=lambda *args, **kwargs: None,
            neighbors=lambda *args, **kwargs: None,
        ),
        tl=types.SimpleNamespace(
            umap=lambda *args, **kwargs: None,
            leiden=leiden,
        ),
    )

    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)

    data = datasets.human_lymph_node(cache_directory=tmp_path, use_cache=False, bring=False)

    assert isinstance(data, ad.AnnData)
    assert data.n_obs == 3
    assert "clusters" in data.obs
    assert (tmp_path / "V1_Human_Lymph_Node_pped.h5ad").exists()


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

    monkeypatch.setattr(urllib.request, "urlopen", lambda url: StatefulResponse())

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

    monkeypatch.setattr(urllib.request, "urlopen", lambda url: StatefulShortResponse())

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
