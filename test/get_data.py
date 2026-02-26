# code to get h5ad data for testing
# Core scverse libraries
from pathlib import Path

import anndata as ad
import pooch
import scanpy as sc
import scvelo as scv


def get_pbmc3k() -> None:

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

    adata.raw = adata.copy()

    adata.obs_names_make_unique()

    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

    # 2. Basic filtering of cells and genes
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)

    # Doublet detection with scrublet
    sc.pp.scrublet(adata, batch_key="sample")

    # before normalization, save raw counts in a separate layer
    adata.layers["counts"] = adata.X.copy()
    # 5. Normalize total counts per cell
    # Normalizing to median total counts
    sc.pp.normalize_total(adata)
    # Logarithmize the data
    sc.pp.log1p(adata)

    # 7. Identify highly variable genes (HVGs)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")

    # pca for dimensionality reduction
    sc.tl.pca(adata)

    # dimensionality reduction with UMAP
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    #
    sc.tl.tsne(adata)

    # clusteri
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

    for res in [0.02, 0.5, 2.0]:
        sc.tl.leiden(adata, key_added=f"leiden_res_{res:4.2f}", resolution=res, flavor="igraph")

    marker_genes = {
        "CD14+ Mono": ["FCN1", "CD14"],
        "CD16+ Mono": ["TCF7L2", "FCGR3A", "LYN"],
        # Note: DMXL2 should be negative
        "cDC2": ["CST3", "COTL1", "LYZ", "DMXL2", "CLEC10A", "FCER1A"],
        "Erythroblast": ["MKI67", "HBA1", "HBB"],
        # Note HBM and GYPA are negative markers
        "Proerythroblast": ["CDK6", "SYNGR1", "HBM", "GYPA"],
        "NK": ["GNLY", "NKG7", "CD247", "FCER1G", "TYROBP", "KLRG1", "FCGR3A"],
        "ILC": ["ID2", "PLCG2", "GNLY", "SYNE1"],
        "Naive CD20+ B": ["MS4A1", "IL4R", "IGHD", "FCRL1", "IGHM"],
        # Note IGHD and IGHM are negative markers
        "B cells": [
            "MS4A1",
            "ITGB1",
            "COL4A4",
            "PRDM1",
            "IRF4",
            "PAX5",
            "BCL11A",
            "BLK",
            "IGHD",
            "IGHM",
        ],
        "Plasma cells": ["MZB1", "HSP90B1", "FNDC3B", "PRDM1", "IGKC", "JCHAIN"],
        # Note PAX5 is a negative marker
        "Plasmablast": ["XBP1", "PRDM1", "PAX5"],
        "CD4+ T": ["CD4", "IL7R", "TRBC2"],
        "CD8+ T": ["CD8A", "CD8B", "GZMK", "GZMA", "CCL5", "GZMB", "GZMH", "GZMA"],
        "T naive": ["LEF1", "CCR7", "TCF7"],
        "pDC": ["GZMB", "IL3RA", "COBLL1", "TCF4"],
    }

    adata.obs["cell_type_lvl1"] = adata.obs["leiden_res_0.02"].map(
        {
            "0": "Lymphocytes",
            "1": "Monocytes",
            "2": "Erythroid",
            "3": "B Cells",
        }
    )

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
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    # 3. Initialize the progress bar
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading Atlas")

    with open(local_filename, "wb") as file:
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
