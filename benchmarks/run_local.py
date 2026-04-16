"""
CLI entry point for local-dataset cellestial benchmarks.

Kept separate from ``run.py`` because these rely on h5ad files that live
outside the repo (under ``~/data``), so they can't run in CI or on a fresh
clone. Uses the same cases + harness as the main runner, but only iterates
over datasets registered in ``LOCAL_LOADERS``.

Usage
-----
    poetry run python -m benchmarks.run_local
    poetry run python -m benchmarks.run_local --datasets breast_cancer_atlas
    poetry run python -m benchmarks.run_local --cases umap_gene dotplot

Output: ``benchmarks/results/results_local.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from benchmarks.cases import CASES
from benchmarks.datasets import LOCAL_LOADERS, DatasetSpec
from benchmarks.run import RESULTS_DIR, Row, _run_case_framework

# Atlas-scale datasets run only scatter-style cases: the point of these
# benchmarks is to prove cellestial scales to hundreds of thousands of cells
# on the UMAP path. Dotplot/heatmap/violin aggregate across many genes and
# groups and balloon the runtime without adding signal.
ATLAS_CASES = ["umap_gene", "umap_cluster"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cellestial benchmarks on local datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(LOCAL_LOADERS.keys()),
        choices=list(LOCAL_LOADERS.keys()),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=ATLAS_CASES,
        choices=list(CASES.keys()),
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[50_000, 80_000],
        help="Subsample sizes to benchmark for each local dataset.",
    )
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "results_local.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    datasets: list[DatasetSpec] = [
        LOCAL_LOADERS[name](subsample=size)
        for name in args.datasets
        for size in args.sizes
    ]
    for dataset in datasets:
        print(f"[dataset] {dataset.name}  n_cells={dataset.n_cells} n_vars={dataset.n_vars}")
        for case_name in args.cases:
            case = CASES[case_name]
            for framework in ("cellestial", "scanpy"):
                print(f"  [{framework:10s}] {case_name} ... ", end="", flush=True)
                try:
                    case_rows = _run_case_framework(
                        case,
                        dataset,
                        framework,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    rows.extend(case_rows)
                    total = next(r for r in case_rows if r.phase == "total")
                    total_html = next(
                        (r for r in case_rows if r.phase == "total_html"), None
                    )
                    extra = (
                        f"  total_html={total_html.time_median_s * 1000:7.1f}ms"
                        if total_html
                        else ""
                    )
                    print(
                        f"total={total.time_median_s * 1000:7.1f}ms  "
                        f"peak={total.peak_mem_mb:6.1f}MB{extra}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"FAILED: {exc!r}")

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print(f"\nWrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
