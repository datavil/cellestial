"""
Spatial benchmark with a categorical key: cellestial vs scanpy.

Measures peak memory and time for building and rendering a spatial scatter
colored by a categorical column, then appends the results to
``benchmarks/results.feather``.

``-d`` selects the spatial dataset by name (not a path): ``human_lymph_node``
or ``visium_hne``.

Run from the repo root::

    poetry run python benchmarks/scripts/spatial_cat.py -d human_lymph_node -r 1
"""

from __future__ import annotations

import argparse
import gc
import tempfile
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import polars as pl
import psutil
import scanpy as sc

import cellestial as cl

CASE = "spatial"
CATEGORICAL = 1
RESULTS = Path(__file__).resolve().parent.parent / "results.feather"
SCHEMA: dict[str, pl.DataType] = {
    "library": pl.Utf8,
    "case": pl.Utf8,
    "categorical": pl.Int64,
    "replica_id": pl.Int64,
    "dataset": pl.Utf8,
    "n_obs": pl.Int64,
    "n_cols": pl.Int64,
    "n_items": pl.Int64,
    "memory(MB)": pl.Float64,
    "time(s)": pl.Float64,
}
DEDUP_KEYS = ["library", "case", "categorical", "replica_id", "dataset", "n_cols"]


def _load_spatial(name: str) -> tuple[object, str]:
    """Return (data, categorical_key) for the named spatial dataset."""
    if name == "human_lymph_node":
        return cl.datasets.human_lymph_node(), "clusters"
    if name == "visium_hne":
        import squidpy as sq

        return sq.datasets.visium_hne_adata(), "leiden"
    message = f"unknown spatial dataset {name!r}; expected human_lymph_node or visium_hne"
    raise ValueError(message)


def _measure(function) -> tuple[float, float]:
    """Run ``function`` while sampling RSS; return (peak_delta_MB, seconds)."""
    process = psutil.Process()
    gc.collect()
    baseline = process.memory_info().rss
    peak = baseline
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, process.memory_info().rss)
            time.sleep(0.005)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    function()
    elapsed = time.perf_counter() - start
    stop.set()
    sampler.join()
    peak = max(peak, process.memory_info().rss)
    return (peak - baseline) / 1e6, elapsed


def _render_cellestial(plot) -> None:
    """Render a cellestial plot to a throwaway SVG."""
    with tempfile.TemporaryDirectory() as directory:
        cl.save(plot, "render.svg", path=directory)


def _render_scanpy(figure) -> None:
    """Render a matplotlib figure to a throwaway SVG and close it."""
    import matplotlib.pyplot as plt

    if figure is None:
        figure = plt.gcf()
    with tempfile.TemporaryDirectory() as directory:
        figure.savefig(Path(directory) / "render.svg", format="svg")
    plt.close(figure)


def _cl_spatial_cat(data, key: str) -> None:
    plot = cl.spatial(data, key=key, tooltips="none", sampling="none")
    _render_cellestial(plot)


def _sc_spatial_cat(data, key: str) -> None:
    sc.pl.spatial(data, color=key, show=False)
    _render_scanpy(None)


def _upsert(rows: list[dict[str, object]]) -> None:
    """Overwrite rows matching the dedup keys and append the rest."""
    incoming = pl.DataFrame(rows, schema=SCHEMA)
    if RESULTS.exists():
        existing = pl.read_ipc(RESULTS)
        kept = existing.join(incoming.select(DEDUP_KEYS), on=DEDUP_KEYS, how="anti")
        merged = pl.concat([kept, incoming], how="vertical_relaxed")
    else:
        merged = incoming
    partial = RESULTS.with_suffix(".feather.part")
    merged.write_ipc(partial)
    partial.replace(RESULTS)


def main() -> None:
    """Parse arguments, benchmark both libraries, and append the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--replica", type=int, required=True)
    parser.add_argument("-d", "--dataset", required=True, help="human_lymph_node or visium_hne")
    args = parser.parse_args()

    data, key = _load_spatial(args.dataset)
    dataset = args.dataset
    n_obs = int(data.n_obs)
    n_cols = 1
    n_items = n_obs * n_cols

    cellestial_memory, cellestial_time = _measure(lambda: _cl_spatial_cat(data, key))
    scanpy_memory, scanpy_time = _measure(lambda: _sc_spatial_cat(data, key))

    rows = [
        {
            "library": "cellestial",
            "case": CASE,
            "categorical": CATEGORICAL,
            "replica_id": args.replica,
            "dataset": dataset,
            "n_obs": n_obs,
            "n_cols": n_cols,
            "n_items": n_items,
            "memory(MB)": cellestial_memory,
            "time(s)": cellestial_time,
        },
        {
            "library": "scanpy",
            "case": CASE,
            "categorical": CATEGORICAL,
            "replica_id": args.replica,
            "dataset": dataset,
            "n_obs": n_obs,
            "n_cols": n_cols,
            "n_items": n_items,
            "memory(MB)": scanpy_memory,
            "time(s)": scanpy_time,
        },
    ]
    _upsert(rows)
    print(
        f"{CASE} {dataset} r{args.replica} key={key} "
        f"cellestial {cellestial_time:.3f}s {cellestial_memory:.1f}MB | "
        f"scanpy {scanpy_time:.3f}s {scanpy_memory:.1f}MB"
    )


if __name__ == "__main__":
    main()
