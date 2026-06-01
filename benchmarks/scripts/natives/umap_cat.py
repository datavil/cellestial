"""
UMAP benchmark with a categorical key: cellestial vs scanpy.

Measures peak memory and time for building and rendering a UMAP colored by a
categorical observation column, then appends the results to
``benchmarks/results.native.feather``.

Run from the repo root::

    poetry run python benchmarks/scripts/natives/umap_cat.py -d data/pbmc3k_pped.h5ad -r 1
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

CASE = "umap"
CATEGORICAL = 1
RESULTS = Path(__file__).resolve().parent.parent.parent / "results.native.feather"
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


def _pick_group_by(data) -> str:
    """Return the first available categorical column to color/group by."""
    for name in ("leiden", "cell_type_lvl1", "cell_type", "clusters"):
        if name in data.obs.columns:
            return name
    message = f"no categorical group_by column found in {list(data.obs.columns)}"
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
        cl.save(plot, "render.html", path=directory)


def _render_scanpy(figure) -> None:
    """Render a matplotlib figure to a throwaway SVG and close it."""
    import matplotlib.pyplot as plt

    if figure is None:
        figure = plt.gcf()
    with tempfile.TemporaryDirectory() as directory:
        figure.savefig(Path(directory) / "render.png", format="png")
    plt.close(figure)


def _cl_umap_cat(data, group_by: str) -> None:
    plot = cl.umap(data, key=group_by, tooltips="none", sampling="none")
    _render_cellestial(plot)


def _sc_umap_cat(data, group_by: str) -> None:
    figure = sc.pl.umap(data, color=group_by, show=False, return_fig=True)
    _render_scanpy(figure)


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
    parser.add_argument("-d", "--dataset", required=True, help="path to an .h5ad file")
    args = parser.parse_args()

    data = sc.read_h5ad(args.dataset)
    dataset = Path(args.dataset).stem
    group_by = _pick_group_by(data)
    n_obs = int(data.n_obs)
    n_cols = 1
    n_items = n_obs * n_cols

    cellestial_memory, cellestial_time = _measure(lambda: _cl_umap_cat(data, group_by))
    scanpy_memory, scanpy_time = _measure(lambda: _sc_umap_cat(data, group_by))

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
        f"{CASE} {dataset} r{args.replica} "
        f"cellestial {cellestial_time:.3f}s {cellestial_memory:.1f}MB | "
        f"scanpy {scanpy_time:.3f}s {scanpy_memory:.1f}MB"
    )


if __name__ == "__main__":
    main()
