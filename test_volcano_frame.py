"""Test script for `_build_volcano_frame`.

Run with:
    poetry run python test_volcano_frame.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
import scanpy as sc

from cellestial.single.special.volcano import _build_volcano_frame


def main() -> None:
    # LOAD: data
    data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

    # PREPARE: differential expression results (scanpy expects log-normalised data;
    # pbmc3k_pped is already preprocessed, but log1p is idempotent enough here).
    if "rank_genes_groups" not in data.uns:
        sc.tl.rank_genes_groups(data, groupby="cell_type_lvl1", method="wilcoxon")

    available_groups = list(data.uns["rank_genes_groups"]["names"].dtype.names)
    print(f"Available groups: {available_groups}")
    group = available_groups[0]
    print(f"Testing with group: `{group}`\n")

    # ---------- TEST 1: defaults ----------
    print("=" * 60)
    print("TEST 1: defaults")
    print("=" * 60)
    frame = _build_volcano_frame(data, group=group)
    print(frame.head(10))
    print(f"shape: {frame.shape}")
    print(f"columns: {frame.columns}")
    print(f"dtypes:  {dict(zip(frame.columns, frame.dtypes))}")

    expected_columns = {
        "variable",
        "logfoldchange",
        "pvalue",
        "neg_log_pvalue",
        "significance",
    }
    assert expected_columns.issubset(set(frame.columns)), (
        f"Missing columns: {expected_columns - set(frame.columns)}"
    )

    # significance counts
    counts = frame.group_by("significance").len().sort("significance")
    print("\nsignificance counts:")
    print(counts)

    # ---------- TEST 2: spot-check the -log10 transform on finite rows ----------
    print("\n" + "=" * 60)
    print("TEST 2: -log10 transform sanity")
    print("=" * 60)
    finite = frame.filter(
        pl.col("pvalue").is_finite() & (pl.col("pvalue") > 0)
    ).head(5)
    expected_neg_log = -np.log10(finite["pvalue"].to_numpy())
    actual_neg_log = finite["neg_log_pvalue"].to_numpy()
    print(f"expected: {expected_neg_log}")
    print(f"actual:   {actual_neg_log}")
    assert np.allclose(expected_neg_log, actual_neg_log, equal_nan=True), (
        "neg_log_pvalue does not match -log10(pvalue)"
    )
    print("OK")

    # ---------- TEST 3: significance labels respect thresholds ----------
    print("\n" + "=" * 60)
    print("TEST 3: significance labels respect thresholds")
    print("=" * 60)
    logfoldchange_threshold = 1.0
    pvalue_threshold = 0.05
    up = frame.filter(pl.col("significance") == "up")
    down = frame.filter(pl.col("significance") == "down")
    ns = frame.filter(pl.col("significance") == "ns")
    if up.height:
        assert (up["logfoldchange"] >= logfoldchange_threshold).all()
        assert (up["pvalue"] < pvalue_threshold).all()
    if down.height:
        assert (down["logfoldchange"] <= -logfoldchange_threshold).all()
        assert (down["pvalue"] < pvalue_threshold).all()
    if ns.height:
        # ns rows fail at least one of the criteria
        fails_logfoldchange = ns["logfoldchange"].abs() < logfoldchange_threshold
        fails_pvalue = ns["pvalue"] >= pvalue_threshold
        assert (fails_logfoldchange | fails_pvalue).all()
    print(f"up={up.height}  down={down.height}  ns={ns.height}  OK")

    # ---------- TEST 4: custom column names + tighter thresholds ----------
    print("\n" + "=" * 60)
    print("TEST 4: custom column names + tighter thresholds")
    print("=" * 60)
    frame2 = _build_volcano_frame(
        data,
        group=group,
        logfoldchange_threshold=2.0,
        pvalue_threshold=0.001,
        variable_column="gene",
        logfoldchange_column="log2fc",
        pvalue_column="padj",
        neg_log_pvalue_column="minus_log10_padj",
        significance_column="status",
        up_label="UP",
        down_label="DOWN",
        nonsignificant_label="NS",
    )
    print(frame2.head(5))
    expected_columns2 = {"gene", "log2fc", "padj", "minus_log10_padj", "status"}
    assert expected_columns2.issubset(set(frame2.columns))
    statuses = set(frame2["status"].unique().to_list())
    assert statuses.issubset({"UP", "DOWN", "NS"}), f"unexpected labels: {statuses}"
    print(f"unique status labels: {statuses}  OK")

    # ---------- TEST 5: raw vs adjusted p-values produce different columns ----------
    print("\n" + "=" * 60)
    print("TEST 5: use_adjusted_pvalue switch")
    print("=" * 60)
    frame_adj = _build_volcano_frame(data, group=group, use_adjusted_pvalue=True)
    frame_raw = _build_volcano_frame(data, group=group, use_adjusted_pvalue=False)
    diff = (frame_adj["pvalue"] - frame_raw["pvalue"]).abs().sum()
    print(f"sum |adj - raw| = {diff}")
    assert diff > 0, "adjusted and raw p-values should not be identical"
    print("OK")

    # ---------- TEST 6: error paths ----------
    print("\n" + "=" * 60)
    print("TEST 6: error paths")
    print("=" * 60)

    try:
        _build_volcano_frame(data, group="not_a_real_group")
    except KeyError as exception:
        print(f"unknown group raises KeyError: {exception}")
    else:
        raise AssertionError("expected KeyError for unknown group")

    try:
        _build_volcano_frame("not an anndata", group=group)
    except TypeError as exception:
        print(f"non-AnnData raises TypeError: {exception}")
    else:
        raise AssertionError("expected TypeError for non-AnnData input")

    try:
        _build_volcano_frame(data, group=group, key="missing_de_key")
    except KeyError as exception:
        print(f"missing uns key raises KeyError: {exception}")
    else:
        raise AssertionError("expected KeyError for missing uns key")

    # ---------- TEST 7: group_by auto-computes rank_genes_groups ----------
    print("\n" + "=" * 60)
    print("TEST 7: group_by auto-computes rank_genes_groups")
    print("=" * 60)
    fresh = sc.read_h5ad("data/pbmc3k_pped.h5ad")
    assert "rank_genes_groups" not in fresh.uns, "expected fresh data to lack DE results"
    frame_auto = _build_volcano_frame(
        fresh, group=group, group_by="cell_type_lvl1"
    )
    assert "rank_genes_groups" in fresh.uns, "expected DE results to be computed in-place"
    assert frame_auto.height > 0
    print(f"computed in-place; frame shape: {frame_auto.shape}  OK")

    # ---------- TEST 8: group_by recomputes when groupby changes ----------
    print("\n" + "=" * 60)
    print("TEST 8: group_by recomputes when stored groupby differs")
    print("=" * 60)
    other_data = sc.read_h5ad("data/pbmc3k_pped.h5ad")
    sc.tl.rank_genes_groups(other_data, groupby="leiden", method="wilcoxon")
    assert other_data.uns["rank_genes_groups"]["params"]["groupby"] == "leiden"
    _build_volcano_frame(
        other_data,
        group=group,
        group_by="cell_type_lvl1",
    )
    assert other_data.uns["rank_genes_groups"]["params"]["groupby"] == "cell_type_lvl1", (
        "expected stale results to be recomputed for the new group_by"
    )
    print("recomputed for new group_by  OK")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
