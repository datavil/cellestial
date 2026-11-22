#!/usr/bin/env python
"""
Rebuild `cell_type_lvl1` of the bundled pbmc3k dataset from the scanpy tutorial mapping.

The hosted `pbmc3k_pped.h5ad` is the bone marrow multiome sample (s1d1, s1d3) of
the scanpy "Preprocessing and clustering" tutorial. Its `cell_type_lvl1` column
was meant to copy that tutorial's coarse annotation of `leiden_res_0.02`, but
clusters 2 and 3 were swapped and cluster 4 was left unlabeled. This script sets
the column from the tutorial mapping, checks that nothing else changed, checks the
labels against canonical markers, and writes `pbmc3k_pped_v2.h5ad`.

Usage
-----
    poetry run python scripts/fix_pbmc3k_labels.py [cache_directory]

`cache_directory` defaults to `data`. The original file is loaded (or copied from
the global cache) there, and the fixed file is written next to it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad
from scipy import sparse

import cellestial as cl

ORIGINAL_URL = "https://huggingface.co/datasets/datavil/pbmc3k/resolve/main/pbmc3k_pped.h5ad"
OUTPUT_NAME = "pbmc3k_pped_v2.h5ad"
CLUSTER_KEY = "leiden_res_0.02"
LABEL_KEY = "cell_type_lvl1"
# Coarse annotation from the scanpy "Preprocessing and clustering" tutorial.
TUTORIAL_MAPPING = {
    "0": "Lymphocytes",
    "1": "Monocytes",
    "2": "B Cells",
    "3": "Erythroid",
    "4": "Erythroid",
}
# Same order as the original column, so the categorical order users see is unchanged.
CATEGORY_ORDER = ["B Cells", "Erythroid", "Lymphocytes", "Monocytes"]


def relabel(data: AnnData) -> AnnData:
    """Return a copy of `data` with `cell_type_lvl1` set from the tutorial mapping."""
    clusters = data.obs[CLUSTER_KEY].astype(str)
    unknown = sorted(set(clusters) - set(TUTORIAL_MAPPING))
    if unknown:
        message = f"Clusters {unknown} of {CLUSTER_KEY!r} have no label in the mapping."
        raise ValueError(message)

    fixed = data.copy()
    fixed.obs[LABEL_KEY] = pd.Categorical(
        clusters.map(TUTORIAL_MAPPING).to_numpy(), categories=CATEGORY_ORDER
    )
    return fixed


def assert_matrix_equal(left: object, right: object, where: str) -> None:
    """Raise when two dense or sparse matrices differ."""
    if sparse.issparse(left) or sparse.issparse(right):
        left_matrix = sparse.csr_matrix(left)
        right_matrix = sparse.csr_matrix(right)
        same = left_matrix.shape == right_matrix.shape and (left_matrix != right_matrix).nnz == 0
    else:
        same = np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
    if not same:
        message = f"{where} changed."
        raise AssertionError(message)


def assert_only_labels_changed(original: AnnData, fixed: AnnData) -> None:
    """Raise when anything but `cell_type_lvl1` differs between the two datasets."""
    assert original.shape == fixed.shape, "shape changed"
    assert original.obs_names.equals(fixed.obs_names), "cell names changed"
    assert original.var_names.equals(fixed.var_names), "gene names changed"
    assert list(original.obs.columns) == list(fixed.obs.columns), "obs columns changed"
    pd.testing.assert_frame_equal(
        original.obs.drop(columns=LABEL_KEY), fixed.obs.drop(columns=LABEL_KEY)
    )
    pd.testing.assert_frame_equal(original.var, fixed.var)
    assert_matrix_equal(original.X, fixed.X, "X")
    assert set(original.layers) == set(fixed.layers), "layer names changed"
    for name in original.layers:
        assert_matrix_equal(original.layers[name], fixed.layers[name], f"layer {name!r}")
    for slot in ("obsm", "varm", "obsp", "varp"):
        original_mapping = getattr(original, slot)
        fixed_mapping = getattr(fixed, slot)
        assert set(original_mapping) == set(fixed_mapping), f"{slot} keys changed"
        for name in original_mapping:
            assert_matrix_equal(original_mapping[name], fixed_mapping[name], f"{slot} {name!r}")
    assert set(original.uns) == set(fixed.uns), "uns keys changed"
    assert (original.raw is None) == (fixed.raw is None), "raw presence changed"
    if original.raw is not None and fixed.raw is not None:
        pd.testing.assert_frame_equal(original.raw.var, fixed.raw.var)
        assert_matrix_equal(original.raw.X, fixed.raw.X, "raw X")


def assert_labels(original: AnnData, fixed: AnnData) -> None:
    """Raise when the new labels are not the tutorial mapping or keep nulls."""
    labels = fixed.obs[LABEL_KEY]
    assert labels.isna().sum() == 0, "null labels remain"
    assert list(labels.cat.categories) == CATEGORY_ORDER, "category order changed"
    expected = original.obs[CLUSTER_KEY].astype(str).map(TUTORIAL_MAPPING)
    assert (labels.astype(str).to_numpy() == expected.to_numpy()).all(), "labels off mapping"


def marker_table(data: AnnData) -> pd.DataFrame:
    """Mean expression of canonical markers per `cell_type_lvl1` label."""
    genes = ["MS4A1", "CD79A", "CD3E", "NKG7", "LYZ", "CD14", "HBB", "HBA1", "GYPA", "CA1"]
    matrix = data[:, genes].X
    values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    frame = pd.DataFrame(values, columns=genes, index=data.obs_names)
    return frame.groupby(data.obs[LABEL_KEY].to_numpy()).mean().round(2)


def assert_markers(table: pd.DataFrame) -> None:
    """Raise when a label is not the top group for its canonical markers."""
    expectations = {
        "B Cells": ["MS4A1", "CD79A"],
        "Erythroid": ["HBB", "GYPA"],
        "Lymphocytes": ["CD3E", "NKG7"],
        "Monocytes": ["LYZ", "CD14"],
    }
    for label, genes in expectations.items():
        for gene in genes:
            top = table[gene].idxmax()
            assert top == label, f"{gene} peaks in {top!r}, expected {label!r}"


def main() -> None:
    """Load the original file, relabel, verify, and write the versioned file."""
    if len(sys.argv) > 2:
        print(__doc__)
        raise SystemExit(1)
    cache_directory = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("data")
    output_file = cache_directory / OUTPUT_NAME

    original = cl.datasets.from_url(ORIGINAL_URL, cache_directory=cache_directory)
    assert isinstance(original, AnnData)
    print("Before:")
    print(pd.crosstab(original.obs[CLUSTER_KEY], original.obs[LABEL_KEY], dropna=False))

    fixed = relabel(original)
    assert_only_labels_changed(original, fixed)
    assert_labels(original, fixed)

    table = marker_table(fixed)
    print("Mean expression per label:")
    print(table)
    assert_markers(table)

    fixed.write_h5ad(output_file)
    print(f"Wrote {output_file}")

    written = read_h5ad(output_file)
    assert_only_labels_changed(original, written)
    assert_labels(original, written)
    print("After:")
    print(pd.crosstab(written.obs[CLUSTER_KEY], written.obs[LABEL_KEY]))


if __name__ == "__main__":
    main()
