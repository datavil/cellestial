#!/usr/bin/env bash
# Benchmark runner. Run all cases, or source this file and call a single one.
#
#   bash benchmarks/CLI.sh                 # run everything
#   source benchmarks/CLI.sh && run_heatmap   # run one case
#
# Each script appends to benchmarks/results.feather (created if absent) and
# overwrites any row with a matching (library, case, replica_id, dataset, n_cols).

# set -euo pipefail

REPLICAS=(1 2 3)
SCRNA_DATASETS=(data/atlas200k.h5ad data/pbmc3k_pped.h5ad)
SPATIAL_DATASETS=(human_lymph_node visium_hne)
GENE_COUNTS=(5 20 100 200)

# Resolve this file's location whether run or sourced, under bash or zsh.
if [ -n "${ZSH_VERSION:-}" ]; then
    _self="${(%):-%x}"
else
    _self="${BASH_SOURCE[0]}"
fi
SCRIPTS_DIR="$(cd "$(dirname "$_self")" && pwd)/scripts"

run_umap_cat() {
    for dataset in "${SCRNA_DATASETS[@]}"; do
        for replica in "${REPLICAS[@]}"; do
            poetry run python "$SCRIPTS_DIR/umap_cat.py" -d "$dataset" -r "$replica"
        done
    done
}

run_umap_var() {
    for dataset in "${SCRNA_DATASETS[@]}"; do
        for replica in "${REPLICAS[@]}"; do
            poetry run python "$SCRIPTS_DIR/umap_var.py" -d "$dataset" -r "$replica"
        done
    done
}

run_heatmap() {
    for dataset in "${SCRNA_DATASETS[@]}"; do
        for n_genes in "${GENE_COUNTS[@]}"; do
            for replica in "${REPLICAS[@]}"; do
                poetry run python "$SCRIPTS_DIR/heatmap.py" -d "$dataset" -n "$n_genes" -r "$replica"
            done
        done
    done
}

run_matrixplot() {
    for dataset in "${SCRNA_DATASETS[@]}"; do
        for n_genes in "${GENE_COUNTS[@]}"; do
            for replica in "${REPLICAS[@]}"; do
                poetry run python "$SCRIPTS_DIR/matrixplot.py" -d "$dataset" -n "$n_genes" -r "$replica"
            done
        done
    done
}

run_dotplot() {
    for dataset in "${SCRNA_DATASETS[@]}"; do
        for n_genes in "${GENE_COUNTS[@]}"; do
            for replica in "${REPLICAS[@]}"; do
                poetry run python "$SCRIPTS_DIR/dotplot.py" -d "$dataset" -n "$n_genes" -r "$replica"
            done
        done
    done
}

run_violin() {
    for dataset in "${SCRNA_DATASETS[@]}"; do
        for replica in "${REPLICAS[@]}"; do
            poetry run python "$SCRIPTS_DIR/violin.py" -d "$dataset" -r "$replica"
        done
    done
}

run_spatial_cat() {
    for dataset in "${SPATIAL_DATASETS[@]}"; do
        for replica in "${REPLICAS[@]}"; do
            poetry run python "$SCRIPTS_DIR/spatial_cat.py" -d "$dataset" -r "$replica"
        done
    done
}

run_spatial_var() {
    for dataset in "${SPATIAL_DATASETS[@]}"; do
        for replica in "${REPLICAS[@]}"; do
            poetry run python "$SCRIPTS_DIR/spatial_var.py" -d "$dataset" -r "$replica"
        done
    done
}

run_all() {
    run_umap_cat
    run_umap_var
    run_heatmap
    run_matrixplot
    run_dotplot
    run_violin
    run_spatial_cat
    run_spatial_var
}

# Execute everything only when run directly, not when sourced.
if [ -n "${ZSH_VERSION:-}" ]; then
    case "${ZSH_EVAL_CONTEXT:-}" in
        *:file) _sourced=1 ;;
        *) _sourced=0 ;;
    esac
elif [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    _sourced=1
else
    _sourced=0
fi

if [ "$_sourced" -eq 0 ]; then
    run_all
fi
