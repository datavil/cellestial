"""Cold-start benchmark: one fresh Python process per measurement.

Run from repo root:

    poetry run python notebooks/benchmark_cold.py
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

OUTPUT_CSV = Path("benchmarks") / "results" / "benchmark_cold.csv"
REPEATS = 5

CASES: list[tuple[str, str]] = [
    ("cellestial", "html"),
    ("cellestial", "svg"),
    # ("cellestial", "pdf"),
    # ("cellestial", "png"),
    ("scanpy", "png"),
    ("scanpy", "svg"),
    # ("scanpy", "pdf"),
]

CHILD = r"""
import resource
import sys
import time

library, output_format = sys.argv[1], sys.argv[2]

import cellestial as cl
import scanpy as sc

data = sc.read_h5ad("data/atlas200k.h5ad")

start = time.perf_counter()
if library == "cellestial":
    plot = cl.umap(data, "cell_type", sampling="none", tooltips="none")
    cl.save(plot, f"figures/umap.{output_format}")
elif library == "scanpy":
    figure = sc.pl.umap(data, color="cell_type", return_fig=True)
    figure.savefig(f"figures/umap_scanpy.{output_format}")
else:
    raise ValueError(library)
elapsed = time.perf_counter() - start

ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mb = ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else ru_maxrss / 1024

print(f"PLOT_SECONDS={elapsed:.6f}")
print(f"PEAK_MEMORY_MB={peak_mb:.6f}")
"""


def run_case(library: str, output_format: str) -> tuple[float, float]:
    completed = subprocess.run(
        [sys.executable, "-c", CHILD, library, output_format],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    plot_seconds = float("nan")
    peak_memory_mb = float("nan")
    for line in completed.stdout.splitlines():
        if line.startswith("PLOT_SECONDS="):
            plot_seconds = float(line.split("=", 1)[1])
        elif line.startswith("PEAK_MEMORY_MB="):
            peak_memory_mb = float(line.split("=", 1)[1])
    return plot_seconds, peak_memory_mb


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["library", "format", "run", "plot_seconds", "peak_memory_mb"])
        for library, output_format in CASES:
            for run in range(1, REPEATS + 1):
                plot_seconds, peak_memory_mb = run_case(library, output_format)
                writer.writerow(
                    [library, output_format, run, f"{plot_seconds:.6f}", f"{peak_memory_mb:.6f}"]
                )
                handle.flush()
                print(
                    f"{library:12} {output_format:6} run {run} "
                    f"{plot_seconds:10.4f}s {peak_memory_mb:10.1f} MB"
                )
    print(f"wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
