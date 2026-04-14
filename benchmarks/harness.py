"""
Timing + memory harness.

Design notes:
- Warmup runs are discarded (import caches, JIT-ish effects in lets-plot).
- Repeats are measured with `time.perf_counter`; we report median and IQR.
- Peak memory is measured on a separate dedicated run via `tracemalloc`
  (tracemalloc adds overhead, so we don't mix it into timing runs).
- A `gc.collect()` is forced between runs to reduce carry-over noise.
"""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimingStats:
    samples: list[float] = field(default_factory=list)

    @property
    def median(self) -> float:
        return statistics.median(self.samples) if self.samples else float("nan")

    @property
    def iqr(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        q = statistics.quantiles(self.samples, n=4)
        return q[2] - q[0]

    @property
    def min(self) -> float:
        return min(self.samples) if self.samples else float("nan")


def time_it(
    fn: Callable[[], Any],
    *,
    warmup: int = 1,
    repeats: int = 5,
) -> TimingStats:
    for _ in range(warmup):
        fn()
    stats = TimingStats()
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        stats.samples.append(time.perf_counter() - t0)
    return stats


def measure_peak_memory_mb(fn: Callable[[], Any]) -> float:
    """Run fn once under tracemalloc and return peak allocated MB."""
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / (1024 * 1024)
