"""
CLI entry point for cellestial benchmarks.

Usage
-----
    poetry run python -m benchmarks.run --quick
    poetry run python -m benchmarks.run --sizes 5000 20000 100000
    poetry run python -m benchmarks.run --cases heatmap dotplot --repeats 5

Output: ``benchmarks/results/results.csv`` with one row per
(dataset, case, framework, phase).
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from benchmarks.cases import CASES, Case, render_for
from benchmarks.datasets import (
    REAL_LOADERS,
    DatasetSpec,
    load_blobs,
)
from benchmarks.harness import measure_peak_memory_mb, time_it

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class Row:
    dataset: str
    n_cells: int
    n_vars: int
    case: str
    framework: str
    phase: str  # construct | render | total
    time_median_s: float
    time_iqr_s: float
    time_min_s: float
    peak_mem_mb: float
    n_repeats: int


def _context(ds: DatasetSpec) -> dict:
    return {
        "gene": ds.gene_keys[0],
        "genes": ds.gene_keys,
        "cluster_key": ds.cluster_key,
    }


def _run_case_framework(
    case: Case,
    ds: DatasetSpec,
    framework: str,
    *,
    warmup: int,
    repeats: int,
) -> list[Row]:
    ctx = _context(ds)
    adata = ds.adata
    construct = case.construct_cellestial if framework == "cellestial" else case.construct_scanpy
    render = render_for(framework)

    # Construct-only: rebuild each repeat so we measure steady-state cost.
    construct_stats = time_it(lambda: construct(adata, ctx), warmup=warmup, repeats=repeats)

    # For render, pre-build once per repeat so we isolate serialization cost.
    def _render_once() -> None:
        plot = construct(adata, ctx)
        render(plot)

    total_stats = time_it(_render_once, warmup=warmup, repeats=repeats)

    # Render-only = total - construct (approx); measure directly instead:
    prebuilt = [construct(adata, ctx) for _ in range(repeats + warmup)]
    idx = {"i": 0}

    def _render_prebuilt() -> None:
        render(prebuilt[idx["i"]])
        idx["i"] += 1

    render_stats = time_it(_render_prebuilt, warmup=warmup, repeats=repeats)

    # Peak memory: measure a single full construct+render.
    peak_mem = measure_peak_memory_mb(_render_once)

    base = {
        "dataset": ds.name,
        "n_cells": ds.n_cells,
        "n_vars": ds.n_vars,
        "case": case.name,
        "framework": framework,
        "n_repeats": repeats,
    }
    return [
        Row(
            **base,
            phase="construct",
            time_median_s=construct_stats.median,
            time_iqr_s=construct_stats.iqr,
            time_min_s=construct_stats.min,
            peak_mem_mb=0.0,
        ),
        Row(
            **base,
            phase="render",
            time_median_s=render_stats.median,
            time_iqr_s=render_stats.iqr,
            time_min_s=render_stats.min,
            peak_mem_mb=0.0,
        ),
        Row(
            **base,
            phase="total",
            time_median_s=total_stats.median,
            time_iqr_s=total_stats.iqr,
            time_min_s=total_stats.min,
            peak_mem_mb=peak_mem,
        ),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run cellestial benchmarks.")
    p.add_argument(
        "--cases",
        nargs="+",
        default=list(CASES.keys()),
        choices=list(CASES.keys()),
    )
    p.add_argument(
        "--real",
        nargs="*",
        default=list(REAL_LOADERS.keys()),
        choices=list(REAL_LOADERS.keys()),
        help="Real scanpy datasets to include. Pass --real (no args) to skip.",
    )
    p.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=[5_000, 20_000, 100_000],
        help="blobs dataset sizes (n_observations). Pass --sizes with no args to skip.",
    )
    p.add_argument("--n-vars", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--quick", action="store_true", help="Fast smoke: one real dataset, 2k blobs, 2 repeats."
    )
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "results.csv")
    return p.parse_args(argv)


def _datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    if args.quick:
        return [
            REAL_LOADERS["pbmc68k_reduced"](),
            load_blobs(2_000, n_variables=args.n_vars),
        ]
    out: list[DatasetSpec] = []
    for name in args.real:
        out.append(REAL_LOADERS[name]())
    for n in args.sizes:
        out.append(load_blobs(n, n_variables=args.n_vars))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.repeats = 2
        args.warmup = 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    datasets = _datasets(args)
    for ds in datasets:
        print(f"[dataset] {ds.name}  n_cells={ds.n_cells} n_vars={ds.n_vars}")
        for case_name in args.cases:
            case = CASES[case_name]
            for framework in ("cellestial", "scanpy"):
                print(f"  [{framework:10s}] {case_name} ... ", end="", flush=True)
                try:
                    case_rows = _run_case_framework(
                        case,
                        ds,
                        framework,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    rows.extend(case_rows)
                    total = next(r for r in case_rows if r.phase == "total")
                    print(
                        f"total={total.time_median_s * 1000:7.1f}ms  "
                        f"peak={total.peak_mem_mb:6.1f}MB"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"FAILED: {exc!r}")

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    print(f"\nWrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
