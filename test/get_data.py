# code to get h5ad data for testing
# Core scverse libraries
from pathlib import Path

import anndata as ad
import pooch
import scanpy as sc
import scvelo as scv


def get_pbmc3k()-> None:

    if Path("data/pbmc3k_pped.h5ad").exists():
        return
    EXAMPLE_DATA = pooch.create(
        path=pooch.os_cache("scverse_tutorials"),
        base_url="doi:10.6084/m9.figshare.22716739.v1/",
    )
    EXAMPLE_DATA.load_registry_from_doi()

    samples = {
        "s1d1": "s1d1_filtered_feature_bc_matrix.h5",
        "s1d3": "s1d3_filtered_feature_bc_matrix.h5",
    }
    adatas = {}

    for sample_id, filename in samples.items():
        path = EXAMPLE_DATA.fetch(filename)
        sample_adata = sc.read_10x_h5(path)
        sample_adata.var_names_make_unique()
        adatas[sample_id] = sample_adata

    adata = ad.concat(adatas, label="sample")

    #adata.write("../data/pbmc3k.h5ad")

    adata.obs_names_make_unique()

    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

    # 2. Basic filtering of cells and genes
    sc.pp.filter_cells(adata, min_genes=200)  # filter cells with at least 200 genes
    sc.pp.filter_genes(adata, min_cells=3)  # filter genes expressed in at least 3 cells

    # 5. Normalize total counts per cell
    sc.pp.normalize_total(adata, target_sum=1e4)

    # 6. Log-transform the data
    sc.pp.log1p(adata)

    # 7. Identify highly variable genes (HVGs)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]

    # 8. Scale the data to unit variance and mean zero
    sc.pp.scale(adata, max_value=10)

    # 9. Perform PCA for dimensionality reduction
    sc.tl.pca(adata, n_comps=50)

    # 10. Compute the neighborhood graph for clustering
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)

    # 11. Compute UMAP for visualization
    sc.tl.umap(adata)
    sc.tl.tsne(adata)

    # 12. Cluster cells
    sc.tl.leiden(adata, resolution=0.5)  # or use sc.tl.louvain

    adata.write("data/pbmc3k_pped.h5ad")

    return

def get_pancreas() -> None:

    if Path("data/endocrinogenesis_day15_pped.h5ad").exists():
        return

    adata = scv.datasets.pancreas("data/endocrinogenesis_day15.h5ad")
    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)
    scv.pp.log1p(adata)
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    scv.tools.velocity_embedding(adata, basis="umap")

    adata.write("data/endocrinogenesis_day15_pped.h5ad")

    return

def get_breast_cancer_atlas() -> None:

    import requests
    from tqdm import tqdm

    url = "https://datasets.cellxgene.cziscience.com/7cdea341-ca7a-40fd-8192-b8ecb2d7b91e.h5ad"
    local_filename = "data/breast_cancer_atlas.h5ad"

    # 1. Use stream=True to avoid loading the whole file into RAM at once
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # 2. Get the total file size from headers (in bytes)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte

    # 3. Initialize the progress bar
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading Atlas")

    with open(local_filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Something went wrong with the download.")
    else:
        print(f"Successfully saved to {local_filename}")

    return

def main() -> None:
    print("1 - getting and pre-processing pbmc3k")
    get_pbmc3k()
    print("2 - getting and pre-processing pancreas")
    get_pancreas()
    print("3 - getting breast cancer cell atlas")
    get_breast_cancer_atlas()
    return

if __name__ == "__main__":
    import os
    print(os.getcwd())
    main()
