"""Benchmark anndata_variable_columns: build.py (baseline) vs build_nextgen.py."""

from __future__ import annotations

import gc
import random
import timeit
import tracemalloc

import cellestial as cl
from cellestial.frames.build import anndata_variable_columns as variable_columns_baseline
from cellestial.frames.build_nextgen import (
    anndata_variable_columns as variable_columns_nextgen,
)

RANDOM_SEED = 42
NUM_REPEATS = 5
K_TARGETS = [10, 200, 500, 1000, 5000]


def peak_memory_mb(callable_fn) -> float:
    gc.collect()
    tracemalloc.start()
    callable_fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def best_time_seconds(callable_fn) -> float:
    gc.collect()
    times = timeit.repeat(callable_fn, number=1, repeat=NUM_REPEATS)
    return min(times)


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    print("Loading pbmc3k...")
    data = cl.datasets.pbmc3k(cache_directory="data")
    n_vars_total = data.n_vars
    print(f"AnnData shape: {data.shape}")
    print(f"n_vars (total): {n_vars_total}")
    print(f"X backend: {type(data.X).__name__}")
    print()

    all_var_names = list(data.var_names)

    k_values = sorted({k for k in K_TARGETS if k <= n_vars_total} | {n_vars_total})

    header = (
        f"{'K':>8} | {'baseline (s)':>14} | {'nextgen (s)':>14} | "
        f"{'speedup':>10} | {'baseline peak (MB)':>20} | {'nextgen peak (MB)':>20}"
    )
    print(header)
    print("-" * len(header))

    for k in k_values:
        if k < n_vars_total:
            keys = rng.sample(all_var_names, k)
        else:
            keys = list(all_var_names)

        baseline_call = lambda: variable_columns_baseline(
            data=data, column_names=[], keys=list(keys)
        )
        nextgen_call = lambda: variable_columns_nextgen(
            data=data, column_names=[], keys=list(keys)
        )

        baseline_call()
        nextgen_call()

        time_baseline = best_time_seconds(baseline_call)
        time_nextgen = best_time_seconds(nextgen_call)
        speedup = time_baseline / time_nextgen if time_nextgen > 0 else float("inf")

        peak_baseline = peak_memory_mb(baseline_call)
        peak_nextgen = peak_memory_mb(nextgen_call)

        print(
            f"{k:>8} | {time_baseline:>14.4f} | {time_nextgen:>14.4f} | "
            f"{speedup:>9.2f}x | {peak_baseline:>20.2f} | {peak_nextgen:>20.2f}"
        )


if __name__ == "__main__":
    main()
