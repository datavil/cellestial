import shutil
from os import PathLike
from pathlib import Path

from anndata import AnnData, read_h5ad

_GLOBAL_CACHE = Path.home() / ".cache" / "cellestial" / "datasets"


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

    if (
        use_cache
        and bring
        and cache_directory != _GLOBAL_CACHE
        and not cache_file.exists()
    ):
        global_cache_file = _GLOBAL_CACHE / filename
        if global_cache_file.exists():
            print(f"Bringing cached file from {global_cache_file} to {cache_file}")
            shutil.copy(global_cache_file, cache_file)

    if not use_cache and cache_file.exists():
        print(f"Removing existing cache: {cache_file}")
        cache_file.unlink()

    return cache_file


def pbmc3k(
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    *,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download and preprocess the pbmc3k dataset.

    Adopted from
    https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html.

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
    """
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

    example_data = pooch.create(
        path=pooch.os_cache("scverse_tutorials"),
        base_url="doi:10.6084/m9.figshare.22716739.v1/",
    )
    example_data.load_registry_from_doi()

    samples = {
        "s1d1": "s1d1_filtered_feature_bc_matrix.h5",
        "s1d3": "s1d3_filtered_feature_bc_matrix.h5",
    }
    sample_adatas = {}

    for sample_id, filename in samples.items():
        path = example_data.fetch(filename)
        sample_adata = sc.read_10x_h5(path)
        sample_adata.var_names_make_unique()
        sample_adatas[sample_id] = sample_adata

    adata = ad.concat(sample_adatas, label="sample")

    adata.raw = adata.copy()

    adata.obs_names_make_unique()

    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

    # basic filtering of cells and genes
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)

    # doublet detection with scrublet
    sc.pp.scrublet(adata, batch_key="sample")

    # before normalization, save raw counts in a separate layer
    adata.layers["counts"] = adata.X.copy()
    # normalize total counts per cell to median total counts
    sc.pp.normalize_total(adata)
    # logarithmize the data
    sc.pp.log1p(adata)

    # identify highly variable genes (HVGs)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")

    # pca for dimensionality reduction
    sc.tl.pca(adata)

    # dimensionality reduction with UMAP
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    sc.tl.tsne(adata)

    # clustering
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

    for resolution in [0.02, 0.5, 2.0]:
        sc.tl.leiden(
            adata,
            key_added=f"leiden_res_{resolution:4.2f}",
            resolution=resolution,
            flavor="igraph",
        )

    adata.obs["cell_type_lvl1"] = adata.obs["leiden_res_0.02"].map(
        {
            "0": "Lymphocytes",
            "1": "Monocytes",
            "2": "Erythroid",
            "3": "B Cells",
        }
    )

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
    Download and preprocess the endocrinogenesis day-15 pancreas dataset.

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

    adata = scv.datasets.pancreas(raw_file)
    scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    scv.pp.moments(adata)
    scv.tl.velocity(adata, mode="deterministic")
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_embedding(adata, basis="umap")

    adata.write(cache_file)

    print(f"Saved to {cache_file}")

    return adata


def breast_cancer_atlas(
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    *,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download the breast cancer cell atlas from cellxgene.

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
    """
    import requests
    from tqdm import tqdm

    cache_file = _resolve_cache_file(
        cache_directory, "breast_cancer_atlas.h5ad", use_cache=use_cache, bring=bring
    )

    if cache_file.exists():
        return read_h5ad(cache_file)

    print(f"No cache at {cache_file} !")
    print("Downloading breast cancer atlas...")
    url = "https://datasets.cellxgene.cziscience.com/7cdea341-ca7a-40fd-8192-b8ecb2d7b91e.h5ad"

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading Atlas")

    with cache_file.open("wb") as file:
        for chunk in response.iter_content(block_size):
            progress_bar.update(len(chunk))
            file.write(chunk)

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        message = f"Download incomplete: expected {total_size} bytes, got {progress_bar.n}."
        raise OSError(message)

    print(f"Saved to {cache_file}")

    return read_h5ad(cache_file)


def human_lymph_node(
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    *,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download and preprocess the V1 Human Lymph Node Visium spatial dataset.

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
    adata = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node")
    adata.var_names_make_unique()

    # mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # filtering
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, max_counts=35000)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    sc.pp.filter_genes(adata, min_cells=10)

    # normalization
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

    # dimensionality reduction
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    # clustering
    sc.tl.leiden(
        adata,
        key_added="clusters",
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )

    adata.write(cache_file)

    print(f"Saved to {cache_file}")

    return adata
