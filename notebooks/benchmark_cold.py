"""Cold-start benchmark: one fresh Python process per measurement.

Run from repo root:

    poetry run python notebooks/benchmark_cold.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CASES: list[tuple[str, str]] = [
    ("cellestial", "html"),
    ("cellestial", "svg"),
    ("cellestial", "pdf"),
    ("cellestial", "png"),
    ("scanpy", "png"),
    ("scanpy", "svg"),
    ("scanpy", "pdf"),
]

CHILD = r"""
import sys
import time

library, output_format = sys.argv[1], sys.argv[2]

import cellestial as cl
import scanpy as sc

data = sc.read_h5ad("data/atlas200k.h5ad")

start = time.perf_counter()
if library == "cellestial":
    plot = cl.umap(data, "cell_type", sampling="none", tooltips="none")
    getattr(plot, f"to_{output_format}")(f"figures/umap.{output_format}")
elif library == "scanpy":
    figure = sc.pl.umap(data, color="cell_type", return_fig=True)
    figure.savefig(f"figures/umap_scanpy.{output_format}")
else:
    raise ValueError(library)
elapsed = time.perf_counter() - start

print(f"PLOT_SECONDS={elapsed:.6f}")
"""


def run_case(library: str, output_format: str) -> tuple[float, float]:
    wall_start = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", CHILD, library, output_format],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    wall_seconds = time.perf_counter() - wall_start

    plot_seconds = float("nan")
    for line in completed.stdout.splitlines():
        if line.startswith("PLOT_SECONDS="):
            plot_seconds = float(line.split("=", 1)[1])
    return plot_seconds, wall_seconds


def main() -> None:
    print(f"{'library':12} {'format':6} {'plot (s)':>10} {'total (s)':>11}")
    print("-" * 42)
    for library, output_format in CASES:
        plot_seconds, wall_seconds = run_case(library, output_format)
        print(f"{library:12} {output_format:6} {plot_seconds:10.4f} {wall_seconds:11.4f}")


if __name__ == "__main__":
    main()
