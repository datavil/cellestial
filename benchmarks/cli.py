"""
CLI: `poetry run python -m benchmarks [...]`.

Default behavior: run every case on every dataset for which it is valid, with
3 replicas each, write the results to `benchmarks/results/results.feather`
(plus a timestamped snapshot under `runs/`), and rebuild the four figures
under `benchmarks/figures/`.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime

from benchmarks import _console, cases, datasets, figures, io, runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks",
        description="Benchmark cellestial plots against scanpy.pl counterparts.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=datasets.available(),
        default=None,
        help="Subset of datasets to run (default: all).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=cases.available(),
        default=None,
        help="Subset of cases to run (default: all).",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=3,
        help="Number of measured replicas per (case, dataset, params) tuple (default: 3).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the warmup pass before measured replicas.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Time only the construction phase; skip SVG export.",
    )
    parser.add_argument(
        "--only-figures",
        action="store_true",
        help="Skip the run; rebuild figures from the existing results.feather.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only load each selected dataset and report metadata. Useful for "
        "verifying the environment without running plots.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and cases, then exit.",
    )
    return parser


def _setup_environment() -> None:
    """Configure matplotlib (Agg) and lets-plot (no JS) for headless SVG export."""
    import matplotlib

    matplotlib.use("Agg")

    from lets_plot import LetsPlot

    try:
        LetsPlot.setup_html(no_js=True, isolated_frame=True)
    except TypeError:
        # Older lets-plot versions don't expose no_js; fall back to setup_html().
        LetsPlot.setup_html()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)

    if namespace.list:
        _print_listing()
        return 0

    _setup_environment()

    _console.banner("cellestial vs scanpy.pl — benchmark suite")
    _console.kv("started_at", datetime.now(UTC).isoformat())
    _console.kv("replicas", namespace.replicas)
    _console.kv(
        "datasets",
        ", ".join(namespace.datasets) if namespace.datasets else "(all)",
    )
    _console.kv(
        "cases",
        ", ".join(namespace.cases) if namespace.cases else "(all)",
    )
    _console.kv("warmup", not namespace.no_warmup)
    _console.kv("render", not namespace.skip_render)

    selected_datasets = datasets.resolve(namespace.datasets)
    selected_cases = cases.resolve(namespace.cases)

    if namespace.check:
        _console.section("dataset check")
        for dataset in selected_datasets:
            metadata = dataset.metadata()
            _console.ok(
                f"{dataset.name}: {metadata.n_cells} cells × {metadata.n_vars} vars "
                f"(kind={metadata.kind})"
            )
        return 0

    if namespace.only_figures:
        _console.section("rebuilding figures only")
        outputs = figures.build_all()
        _console.kv("figures written", len(outputs))
        return 0

    overall_start = time.perf_counter()

    _console.section("preloading dataset metadata")
    for dataset in selected_datasets:
        try:
            dataset.metadata()
        except Exception as exc:
            _console.fail(f"failed to load '{dataset.name}': {exc}")
            return 2

    frame = runner.run(
        cases=selected_cases,
        datasets=selected_datasets,
        replicas=namespace.replicas,
        warmup=not namespace.no_warmup,
        skip_render=namespace.skip_render,
    )

    _console.section("writing results")
    consolidated = io.write_results(frame)
    _console.ok(f"wrote {consolidated} ({frame.height} rows)")

    _console.section("building figures")
    outputs = figures.build_all()
    _console.kv("figures written", len(outputs))

    elapsed = time.perf_counter() - overall_start
    _console.banner(f"done in {elapsed:.1f}s")
    return 0


def _print_listing() -> None:
    _console.section("datasets")
    for name in datasets.available():
        entry = datasets.get(name)
        _console.kv(name, entry.kind)
    _console.section("cases")
    for case in cases.CASES:
        supports = "+".join(case.supports)
        _console.kv(case.name, supports)


if __name__ == "__main__":
    sys.exit(main())
