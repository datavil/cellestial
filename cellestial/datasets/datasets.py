import importlib.util
import re
import shutil
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

from anndata import AnnData, read_h5ad

_GLOBAL_CACHE = Path.home() / ".cache" / "cellestial" / "datasets"

_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def _require_extras(function_name: str, *modules: str) -> None:
    """Raise ImportError with an install hint if any module is missing."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        message = (
            f"datasets.{function_name} requires optional dependencies: {', '.join(missing)}. "
            "Install them with: pip install 'cellestial[extras]'"
        )
        raise ImportError(message)


def _resolve_cache_file(
    cache_directory: str | Path | PathLike,
    filename: str,
    *,
    use_cache: bool,
    bring: bool,
) -> Path:
    """
    Prepare the cache directory and return the target cache file path.

    Creates `cache_directory` if missing. When `bring` is True and the user
    points at a non-default directory that doesn't yet contain `filename`,
    copies the file from `_GLOBAL_CACHE` if it exists there. When `use_cache`
    is False, removes the target cache file so it will be regenerated.
    """
    if not isinstance(cache_directory, Path):
        cache_directory = Path(cache_directory)

    if not cache_directory.exists():
        cache_directory.mkdir(parents=True)

    cache_file = cache_directory / filename

    if use_cache and bring and cache_directory != _GLOBAL_CACHE and not cache_file.exists():
        global_cache_file = _GLOBAL_CACHE / filename
        if global_cache_file.exists():
            print(f"Bringing cached file from {global_cache_file} to {cache_file}")
            shutil.copy(global_cache_file, cache_file)

    if not use_cache and cache_file.exists():
        print(f"Removing existing cache: {cache_file}")
        cache_file.unlink()

    return cache_file


def _run_steps(
    steps: Sequence[tuple[str, str, Callable[[], object]]],
    desc: str,
) -> None:
    """
    Drive a sequence of preprocessing steps with a tqdm progress bar.

    Parameters
    ----------
    steps : Sequence[tuple[str, str, Callable[[], object]]]
        Steps to execute in order. `label` is human-readable, `code` is a
        short representation of the call, `action` performs the work.
    desc : str
        Prefix shown on every step's description, e.g. `"Preprocessing pbmc3k"`.

    Notes
    -----
    Each step is a `(label, code, action)` triple. The bar description is set
    to `"{desc} | {label}: {code}"` before `action` runs, so the user sees
    both a human label and the underlying call. `FutureWarning` and
    `UserWarning` are silenced for the duration so they don't clutter the bar;
    the suppression is scoped to this call only.
    """
    import contextlib
    import os
    import sys
    import warnings

    import tqdm as tqdm_module
    from tqdm import tqdm

    devnull_path = Path(os.devnull)

    # The static prefix is baked into bar_format (escaping any literal braces)
    # so that {desc} is free to carry the per-step info after the bar.
    prefix = desc.replace("{", "{{").replace("}", "}}")
    bar_format = f"{prefix} | {{percentage:3.0f}}%|{{bar}}| {{n_fmt}}/{{total_fmt}} {{desc}}"

    # Keep a handle on the real stdout/stderr from before the redirects so our
    # own bar can stay visible. Route the bar to *stdout* (same stream as the
    # surrounding print()s) — otherwise Jupyter's separate stdout/stderr
    # buffering reorders the bar relative to neighboring prints.
    real_stdout = sys.stdout
    # Flush pending stdout so anything printed before _run_steps emerges
    # before the bar appears.
    real_stdout.flush()

    # Some libraries (e.g. scvelo's velocity_graph) draw their own tqdm bars
    # that bypass our stdout/stderr redirect (notebook backend writes via
    # IPython display). Patch tqdm's __init__ to force disable=True for any
    # bar created after our own — this silences nested bars without touching
    # this outer bar (already constructed before patching takes effect).
    tqdm_classes: list[Any] = [tqdm_module.std.tqdm]
    with contextlib.suppress(ImportError):
        import tqdm.notebook as tqdm_notebook

        tqdm_classes.append(tqdm_notebook.tqdm)

    @contextlib.contextmanager
    def silence_nested_tqdm():
        originals = [(cls, cls.__init__) for cls in tqdm_classes]
        for cls, original in originals:

            def patched(self, *args, _original=original, **kwargs):  # type: ignore[no-untyped-def]
                kwargs["disable"] = True
                _original(self, *args, **kwargs)

            cls.__init__ = patched
        try:
            yield
        finally:
            for cls, original in originals:
                cls.__init__ = original

    with (
        warnings.catch_warnings(),
        devnull_path.open("w") as devnull,
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
        tqdm(
            total=len(steps),
            bar_format=bar_format,
            file=real_stdout,
            leave=True,
        ) as bar,
        silence_nested_tqdm(),
    ):
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        for label, code, action in steps:
            bar.set_description_str(f"{label}: {code}")
            action()
            bar.update(1)


def pbmc3k(
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    *,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download, preprocess, and load the pbmc3k dataset.

    Parameters
    ----------
    cache_directory : str | Path | PathLike
        Directory where the preprocessed `.h5ad` file is cached.
    use_cache : bool, default=True
        If True, load from the cached file when present. If False, remove any
        existing cached file and regenerate from scratch.
    bring : bool, default=True
        If True and `cache_directory` is not the default global cache, copy the
        file from the global cache when it already exists there. Set to False
        to disable this behavior.

    Returns
    -------
    AnnData
        The preprocessed pbmc3k dataset.

    Notes
    -----
    Adopted from
    https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html.
    """
    _require_extras("pbmc3k", "scanpy", "pooch")

    import anndata as ad
    import pooch
    import scanpy as sc

    cache_file = _resolve_cache_file(
        cache_directory, "pbmc3k_pped.h5ad", use_cache=use_cache, bring=bring
    )

    if cache_file.exists():
        return read_h5ad(cache_file)

    print(f"No cache at {cache_file} !")
    print("Downloading and preprocessing pbmc3k...")

    sample_files = {
        "s1d1": "s1d1_filtered_feature_bc_matrix.h5",
        "s1d3": "s1d3_filtered_feature_bc_matrix.h5",
    }
    sample_adatas: dict = {}
    example_data: Any = None
    adata: Any = None

    def step_setup_registry() -> None:
        nonlocal example_data
        example_data = pooch.create(
            path=pooch.os_cache("scverse_tutorials"),
            base_url="doi:10.6084/m9.figshare.22716739.v1/",
        )
        example_data.load_registry_from_doi()

    def step_fetch_samples() -> None:
        for sample_id, filename in sample_files.items():
            path = example_data.fetch(filename)
            sample_adata = sc.read_10x_h5(path)
            sample_adata.var_names_make_unique()
            sample_adatas[sample_id] = sample_adata

    def step_concat() -> None:
        nonlocal adata
        adata = ad.concat(sample_adatas, label="sample")
        adata.raw = adata.copy()
        adata.obs_names_make_unique()

    def step_gene_families() -> None:
        # mitochondrial ("MT-" human, "Mt-" mouse), ribosomal, hemoglobin
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
        adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    def step_save_counts() -> None:
        adata.layers["counts"] = adata.X.copy()

    def step_leiden_multi() -> None:
        for resolution in [0.02, 0.5, 2.0]:
            sc.tl.leiden(
                adata,
                key_added=f"leiden_res_{resolution:4.2f}",
                resolution=resolution,
                flavor="igraph",
            )

    def step_cell_types() -> None:
        adata.obs["cell_type_lvl1"] = adata.obs["leiden_res_0.02"].map(
            {
                "0": "Lymphocytes",
                "1": "Monocytes",
                "2": "Erythroid",
                "3": "B Cells",
            }
        )

    steps: list[tuple[str, str, Callable[[], object]]] = [
        ("setup registry", "pooch.load_registry_from_doi(...)", step_setup_registry),
        ("fetching samples", "sc.read_10x_h5(pooch.fetch(...))", step_fetch_samples),
        ("concatenating samples", "ad.concat(sample_adatas, label='sample')", step_concat),
        ("annotating gene families", "adata.var['mt'|'ribo'|'hb'] = ...", step_gene_families),
        (
            "computing QC metrics",
            "sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'])",
            lambda: sc.pp.calculate_qc_metrics(
                adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
            ),
        ),
        (
            "filtering cells",
            "sc.pp.filter_cells(adata, min_genes=100)",
            lambda: sc.pp.filter_cells(adata, min_genes=100),
        ),
        (
            "filtering genes",
            "sc.pp.filter_genes(adata, min_cells=3)",
            lambda: sc.pp.filter_genes(adata, min_cells=3),
        ),
        (
            "doublet detection",
            "sc.pp.scrublet(adata, batch_key='sample')",
            lambda: sc.pp.scrublet(adata, batch_key="sample"),
        ),
        ("saving raw counts", "adata.layers['counts'] = adata.X.copy()", step_save_counts),
        (
            "normalizing totals",
            "sc.pp.normalize_total(adata)",
            lambda: sc.pp.normalize_total(adata),
        ),
        ("log1p transform", "sc.pp.log1p(adata)", lambda: sc.pp.log1p(adata)),
        (
            "highly variable genes",
            "sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='sample')",
            lambda: sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample"),
        ),
        ("PCA", "sc.tl.pca(adata)", lambda: sc.tl.pca(adata)),
        ("neighbors graph", "sc.pp.neighbors(adata)", lambda: sc.pp.neighbors(adata)),
        ("UMAP", "sc.tl.umap(adata)", lambda: sc.tl.umap(adata)),
        ("t-SNE", "sc.tl.tsne(adata)", lambda: sc.tl.tsne(adata)),
        (
            "leiden clustering",
            "sc.tl.leiden(adata, flavor='igraph', n_iterations=2)",
            lambda: sc.tl.leiden(adata, flavor="igraph", n_iterations=2),
        ),
        (
            "leiden multi-resolution",
            "sc.tl.leiden(adata, resolution=[0.02, 0.5, 2.0])",
            step_leiden_multi,
        ),
        (
            "mapping cell types",
            "adata.obs['cell_type_lvl1'] = adata.obs['leiden_res_0.02'].map(...)",
            step_cell_types,
        ),
    ]

    _run_steps(steps, desc="Preprocessing pbmc3k")

    adata.write(cache_file)

    print(f"Saved to {cache_file}")

    return adata


def pancreas(
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    *,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download, preprocess, and load the endocrinogenesis day-15 pancreas dataset.

    Parameters
    ----------
    cache_directory : str | Path | PathLike
        Directory where the preprocessed `.h5ad` file is cached.
    use_cache : bool, default=True
        If True, load from the cached file when present. If False, remove any
        existing cached file and regenerate from scratch.
    bring : bool, default=True
        If True and `cache_directory` is not the default global cache, copy the
        file from the global cache when it already exists there. Set to False
        to disable this behavior.

    Returns
    -------
    AnnData
        The preprocessed pancreas dataset with velocity computed.
    """
    _require_extras("pancreas", "scanpy", "scvelo")

    import scanpy as sc
    import scvelo as scv

    cache_file = _resolve_cache_file(
        cache_directory,
        "endocrinogenesis_day15_pped.h5ad",
        use_cache=use_cache,
        bring=bring,
    )
    raw_file = cache_file.parent / "endocrinogenesis_day15.h5ad"

    if cache_file.exists():
        return read_h5ad(cache_file)

    print(f"No cache at {cache_file} !")
    print("Downloading and preprocessing pancreas...")

    adata: Any = None

    def step_load() -> None:
        nonlocal adata
        adata = scv.datasets.pancreas(raw_file)

    steps: list[tuple[str, str, Callable[[], object]]] = [
        ("loading dataset", "scv.datasets.pancreas(raw_file)", step_load),
        (
            "filter + normalize",
            "scv.pp.filter_and_normalize(adata, min_shared_counts=20)",
            lambda: scv.pp.filter_and_normalize(adata, min_shared_counts=20),
        ),
        (
            "neighbors graph",
            "sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)",
            lambda: sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30),
        ),
        ("moments", "scv.pp.moments(adata)", lambda: scv.pp.moments(adata)),
        (
            "velocity",
            "scv.tl.velocity(adata, mode='deterministic')",
            lambda: scv.tl.velocity(adata, mode="deterministic"),
        ),
        (
            "velocity graph",
            "scv.tl.velocity_graph(adata)",
            lambda: scv.tl.velocity_graph(adata),
        ),
        (
            "velocity embedding",
            "scv.tl.velocity_embedding(adata, basis='umap')",
            lambda: scv.tl.velocity_embedding(adata, basis="umap"),
        ),
    ]

    _run_steps(steps, desc="Preprocessing pancreas")

    adata.write(cache_file)

    print(f"Saved to {cache_file}")

    return adata


def from_url(
    url: str,
    *,
    name: str | None = None,
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download and load an `.h5ad` dataset from a direct HTTP(S) URL.

    Parameters
    ----------
    url : str
        Direct URL to a downloadable `.h5ad` file.
    name : str, optional
        Name for the cached file (without extension). If omitted, the cache
        filename is derived from the URL's last path segment.
    cache_directory : str | Path | PathLike
        Directory where the `.h5ad` file is cached.
    use_cache : bool, default=True
        If True, load from the cached file when present. If False, remove any
        existing cached file and regenerate from scratch.
    bring : bool, default=True
        If True and `cache_directory` is not the default global cache, copy the
        file from the global cache when it already exists there. Set to False
        to disable this behavior.

    Returns
    -------
    AnnData
        The downloaded dataset.

    Raises
    ------
    ValueError
        If `name` is not given and no filename can be derived from `url`.
    OSError
        If the downloaded file size does not match the expected size.

    Notes
    -----
    Generic loader for any host that serves an AnnData `.h5ad` file at a
    stable URL (CELLxGENE, Zenodo, Figshare, S3 buckets, etc.).
    """
    from urllib.parse import urlparse
    from urllib.request import urlopen

    from tqdm import tqdm

    if name is not None:
        filename = f"{name}.h5ad"
    else:
        filename = Path(urlparse(url).path).name
        if not filename:
            message = "Could not derive a filename from the URL; pass `name=...` explicitly."
            raise ValueError(message)

    cache_file = _resolve_cache_file(cache_directory, filename, use_cache=use_cache, bring=bring)

    if cache_file.exists():
        return read_h5ad(cache_file)

    print(f"No cache at {cache_file} !")
    print(f"Downloading from {url} ...")

    block_size = 1024 * 1024
    # Stream to a .part file and atomically rename on success, so an interrupted
    # download cannot poison the cache.
    partial_file = cache_file.with_name(cache_file.name + ".part")

    with urlopen(url) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        progress_bar = tqdm(
            total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {filename}"
        )

        try:
            with partial_file.open("wb") as file:
                while chunk := response.read(block_size):
                    progress_bar.update(len(chunk))
                    file.write(chunk)

            if total_size != 0 and progress_bar.n != total_size:
                message = (
                    f"Download incomplete: expected {total_size} bytes, got {progress_bar.n}."
                )
                raise OSError(message)  # noqa: TRY301

            partial_file.replace(cache_file)
        except BaseException:
            if partial_file.exists():
                partial_file.unlink()
            raise
        finally:
            progress_bar.close()

    print(f"Saved to {cache_file}")

    return read_h5ad(cache_file)


def from_cellxgene(
    source: str,
    *,
    name: str | None = None,
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download and load a dataset from CELLxGENE by URL or UUID.

    Parameters
    ----------
    source : str
        A CELLxGENE dataset UUID (e.g.
        `7cdea341-ca7a-40fd-8192-b8ecb2d7b91e`) or a full `.h5ad` URL (e.g.
        `https://datasets.cellxgene.cziscience.com/<uuid>.h5ad`).
    name : str, optional
        Name for the cached file (without extension). Defaults to the dataset
        UUID parsed from `source`.
    cache_directory : str | Path | PathLike
        Directory where the `.h5ad` file is cached.
    use_cache : bool, default=True
        If True, load from the cached file when present. If False, remove any
        existing cached file and regenerate from scratch.
    bring : bool, default=True
        If True and `cache_directory` is not the default global cache, copy the
        file from the global cache when it already exists there. Set to False
        to disable this behavior.

    Returns
    -------
    AnnData
        The downloaded dataset.

    Raises
    ------
    ValueError
        If `source` is neither a CELLxGENE dataset UUID nor a URL containing one.
    OSError
        If the downloaded file size does not match the expected size.

    Notes
    -----
    Thin wrapper around `from_url` that accepts a CELLxGENE dataset UUID
    directly and constructs the canonical download URL.
    """
    match = _UUID_PATTERN.search(source)
    if match is None:
        message = (
            f"source must be a CELLxGENE dataset UUID or a URL containing one, got {source!r}"
        )
        raise ValueError(message)

    uuid = match.group(0)
    url = (
        source
        if source.startswith(("http://", "https://"))
        else f"https://datasets.cellxgene.cziscience.com/{uuid}.h5ad"
    )

    return from_url(
        url,
        name=name or uuid,
        cache_directory=cache_directory,
        use_cache=use_cache,
        bring=bring,
    )


def breast_cancer_atlas(
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    *,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download and load the breast cancer cell atlas from CELLxGENE.

    Parameters
    ----------
    cache_directory : str | Path | PathLike
        Directory where the `.h5ad` file is cached.
    use_cache : bool, default=True
        If True, load from the cached file when present. If False, remove any
        existing cached file and regenerate from scratch.
    bring : bool, default=True
        If True and `cache_directory` is not the default global cache, copy the
        file from the global cache when it already exists there. Set to False
        to disable this behavior.

    Returns
    -------
    AnnData
        The breast cancer atlas dataset.

    Raises
    ------
    OSError
        If the downloaded file size does not match the expected size.

    Notes
    -----
    Thin wrapper around `from_cellxgene` with the atlas UUID baked in.
    """
    return from_cellxgene(
        "7cdea341-ca7a-40fd-8192-b8ecb2d7b91e",
        name="breast_cancer_atlas",
        cache_directory=cache_directory,
        use_cache=use_cache,
        bring=bring,
    )


def human_lymph_node(
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    *,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download, preprocess, and load the V1 Human Lymph Node Visium spatial dataset.

    Parameters
    ----------
    cache_directory : str | Path | PathLike
        Directory where the preprocessed `.h5ad` file is cached.
    use_cache : bool, default=True
        If True, load from the cached file when present. If False, remove any
        existing cached file and regenerate from scratch.
    bring : bool, default=True
        If True and `cache_directory` is not the default global cache, copy the
        file from the global cache when it already exists there. Set to False
        to disable this behavior.

    Returns
    -------
    AnnData
        The preprocessed V1 Human Lymph Node Visium dataset with clusters and UMAP.
    """
    _require_extras("human_lymph_node", "scanpy")

    import scanpy as sc

    cache_file = _resolve_cache_file(
        cache_directory,
        "V1_Human_Lymph_Node_pped.h5ad",
        use_cache=use_cache,
        bring=bring,
    )

    if cache_file.exists():
        return read_h5ad(cache_file)

    print(f"No cache at {cache_file} !")
    print("Downloading and preprocessing V1 Human Lymph Node...")

    adata: Any = None

    def step_download() -> None:
        nonlocal adata
        adata = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")
        adata.var_names_make_unique()

    def step_annotate_mt() -> None:
        adata.var["mt"] = adata.var_names.str.startswith("MT-")

    def step_filter_mt_pct() -> None:
        nonlocal adata
        adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

    steps: list[tuple[str, str, Callable[[], object]]] = [
        (
            "downloading dataset",
            "sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')",
            step_download,
        ),
        (
            "annotating mt genes",
            "adata.var['mt'] = adata.var_names.str.startswith('MT-')",
            step_annotate_mt,
        ),
        (
            "computing QC metrics",
            "sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'])",
            lambda: sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True),
        ),
        (
            "filtering cells (min_counts)",
            "sc.pp.filter_cells(adata, min_counts=5000)",
            lambda: sc.pp.filter_cells(adata, min_counts=5000),
        ),
        (
            "filtering cells (max_counts)",
            "sc.pp.filter_cells(adata, max_counts=35000)",
            lambda: sc.pp.filter_cells(adata, max_counts=35000),
        ),
        (
            "filtering by mt %",
            "adata = adata[adata.obs['pct_counts_mt'] < 20].copy()",
            step_filter_mt_pct,
        ),
        (
            "filtering genes",
            "sc.pp.filter_genes(adata, min_cells=10)",
            lambda: sc.pp.filter_genes(adata, min_cells=10),
        ),
        (
            "normalizing totals",
            "sc.pp.normalize_total(adata, inplace=True)",
            lambda: sc.pp.normalize_total(adata, inplace=True),
        ),
        ("log1p transform", "sc.pp.log1p(adata)", lambda: sc.pp.log1p(adata)),
        (
            "highly variable genes",
            "sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)",
            lambda: sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000),
        ),
        ("PCA", "sc.pp.pca(adata)", lambda: sc.pp.pca(adata)),
        ("neighbors graph", "sc.pp.neighbors(adata)", lambda: sc.pp.neighbors(adata)),
        ("UMAP", "sc.tl.umap(adata)", lambda: sc.tl.umap(adata)),
        (
            "leiden clustering",
            "sc.tl.leiden(adata, key_added='clusters', flavor='igraph', directed=False, n_iterations=2)",
            lambda: sc.tl.leiden(
                adata,
                key_added="clusters",
                flavor="igraph",
                directed=False,
                n_iterations=2,
            ),
        ),
    ]

    _run_steps(steps, desc="Preprocessing V1 Human Lymph Node")

    adata.write(cache_file)

    print(f"Saved to {cache_file}")

    return adata
