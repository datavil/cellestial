"""Correctness check: build.py vs build_nextgen.py produce equivalent columns."""

from __future__ import annotations

import random

import numpy as np
import polars as pl

import cellestial as cl
from cellestial.frames.build import anndata_variable_columns as variable_columns_baseline
from cellestial.frames.build_nextgen import (
    anndata_variable_columns as variable_columns_nextgen,
)


def main() -> None:
    data = cl.datasets.pbmc3k(cache_directory="data")
    rng = random.Random(0)

    for k in [1, 10, 200, 1000]:
        keys = rng.sample(list(data.var_names), k)

        columns_baseline = variable_columns_baseline(
            data=data, column_names=[], keys=list(keys)
        )
        columns_nextgen = variable_columns_nextgen(
            data=data, column_names=[], keys=list(keys)
        )

        frame_baseline = pl.DataFrame(columns_baseline)
        frame_nextgen = pl.DataFrame(columns_nextgen)

        assert frame_baseline.columns == frame_nextgen.columns, (
            f"K={k}: column order differs"
        )
        for col in frame_baseline.columns:
            same = np.allclose(
                frame_baseline[col].to_numpy(),
                frame_nextgen[col].to_numpy(),
                rtol=1e-5,
                atol=1e-6,
            )
            assert same, f"K={k}: values differ in column {col}"
        print(f"K={k}: OK ({len(keys)} columns match)")

    print()
    print("Repeat-suppression check:")
    keys_with_repeat = ["MIR1302-2HG", "FAM138A", "MIR1302-2HG"]
    columns_baseline = variable_columns_baseline(
        data=data, column_names=[], keys=list(keys_with_repeat)
    )
    columns_nextgen = variable_columns_nextgen(
        data=data, column_names=[], keys=list(keys_with_repeat)
    )
    assert [c.name for c in columns_baseline] == [c.name for c in columns_nextgen]
    print(f"baseline: {[c.name for c in columns_baseline]}")
    print(f"nextgen:  {[c.name for c in columns_nextgen]}")

    print()
    print("Missing-key error check:")
    try:
        variable_columns_nextgen(data=data, column_names=[], keys=["__no_such_gene__"])
    except Exception as exc:
        print(f"nextgen raises: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
