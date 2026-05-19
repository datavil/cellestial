"""
Timing harness.

For every (case × dataset × params) tuple, runs one warmup pass plus
`replicas` measured passes. Each pass is split into two phases:

- `construct` — calling the library function. For cellestial this returns a
  PlotSpec (cheap, deferred). For scanpy this builds the matplotlib Figure
  (includes layout work that lets-plot defers to render).
- `render` — serializing to a real `.svg` file under `benchmarks/svg/...`.

Results are appended to a polars frame matching the schema in
`plans/benchmarks.md`.
"""

from __future__ import annotations

import gc
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks import _console, io

if TYPE_CHECKING:
    import polars as pl
    from anndata import AnnData

    from benchmarks.cases import Case, Params
    from benchmarks.datasets import DatasetEntry, DatasetMetadata


@dataclass
class Row:
    """One result row before being assembled into the polars frame."""

    case: str
    library: str
    dataset: str
    n_cells: int
    n_keys: int
    key_kind: str
    phase: str
    replica: int
    seconds: float | None
    svg_bytes: int
    status: str
    started_at: datetime


# --- Library-specific renderers ---------------------------------------------


def _render_cellestial(plot: object, path: Path) -> int:
    from lets_plot.export import ggsave

    ggsave(plot, filename=path.name, path=str(path.parent), iframe=False)
    return path.stat().st_size


def _render_scanpy(figure_or_axes: object, path: Path) -> int:
    import matplotlib.pyplot as plt

    # scanpy's DotPlot / MatrixPlot / StackedViolin are BasePlot subclasses
    # that defer figure construction; they expose `.savefig()` which calls
    # `.show()` internally.
    if (
        hasattr(figure_or_axes, "savefig")
        and hasattr(figure_or_axes, "make_figure")
        and hasattr(figure_or_axes, "show")
    ):
        figure_or_axes.savefig(path, bbox_inches="tight")
        plt.close("all")
        return path.stat().st_size

    figure = _resolve_figure(figure_or_axes)
    if figure is None:
        message = f"could not resolve a matplotlib Figure from {type(figure_or_axes)!r}"
        raise RuntimeError(message)
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return path.stat().st_size


def _resolve_figure(value: object):
    """Pull a matplotlib Figure out of a scanpy return value."""
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    if value is None:
        # scanpy.pl.heatmap returns a dict[str, Axes] when show=False; matplotlib's
        # current figure is the one we want.
        import matplotlib.pyplot as plt

        return plt.gcf()
    if isinstance(value, Figure):
        return value
    if isinstance(value, Axes):
        return value.figure
    if isinstance(value, dict):
        for candidate in value.values():
            if isinstance(candidate, Axes):
                return candidate.figure
            if isinstance(candidate, Figure):
                return candidate
    if hasattr(value, "get_axes"):
        axes_list = value.get_axes()  # type: ignore[attr-defined]
        if axes_list:
            return axes_list[0].figure
    if hasattr(value, "fig"):
        return value.fig  # type: ignore[attr-defined]
    if hasattr(value, "figure"):
        return value.figure  # type: ignore[attr-defined]
    import matplotlib.pyplot as plt

    return plt.gcf()


# --- Phase runners -----------------------------------------------------------


@dataclass
class ReplicaTiming:
    construct_seconds: float | None
    render_seconds: float | None
    svg_bytes: int
    error: str | None


def _time_call(
    *,
    library: str,
    construct: Callable[[], object],
    render: Callable[[object], int],
    svg_target: Path,
) -> ReplicaTiming:
    """Run construct + render for one library, returning timings or an error."""
    gc.collect()
    try:
        construct_start = time.perf_counter()
        artifact = construct()
        construct_seconds = time.perf_counter() - construct_start

        render_start = time.perf_counter()
        svg_bytes = render(artifact)
        render_seconds = time.perf_counter() - render_start

        return ReplicaTiming(
            construct_seconds=construct_seconds,
            render_seconds=render_seconds,
            svg_bytes=svg_bytes,
            error=None,
        )
    except Exception as exc:
        message = f"{exc.__class__.__name__}: {exc}"
        if os.environ.get("BENCHMARKS_DEBUG"):
            traceback.print_exc()
        # Clean up any half-written SVG so we don't mistake it for valid output.
        if svg_target.exists():
            svg_target.unlink()
        return ReplicaTiming(
            construct_seconds=None,
            render_seconds=None,
            svg_bytes=0,
            error=message,
        )
    finally:
        # Aggressively drop matplotlib state between replicas.
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except ImportError:
            pass


def _make_rows(
    *,
    case: Case,
    dataset: DatasetEntry,
    metadata: DatasetMetadata,
    params: Params,
    library: str,
    replica: int,
    timing: ReplicaTiming,
    started_at: datetime,
) -> list[Row]:
    status_construct = "ok" if timing.error is None else f"error:{timing.error}"
    status_render = "ok" if timing.error is None else f"error:{timing.error}"
    common = dict(
        case=case.name,
        library=library,
        dataset=dataset.name,
        n_cells=metadata.n_cells,
        n_keys=params.n_keys,
        key_kind=params.key_kind,
        replica=replica,
        started_at=started_at,
    )
    return [
        Row(
            **common,
            phase="construct",
            seconds=timing.construct_seconds,
            svg_bytes=0,
            status=status_construct,
        ),
        Row(
            **common,
            phase="render",
            seconds=timing.render_seconds,
            svg_bytes=timing.svg_bytes,
            status=status_render,
        ),
    ]


def _supports_dataset(case: Case, metadata: DatasetMetadata) -> bool:
    return metadata.kind in case.supports


def _skip_row(
    *,
    case: Case,
    dataset: DatasetEntry,
    metadata: DatasetMetadata,
    reason: str,
) -> list[Row]:
    started_at = datetime.now(UTC)
    common = dict(
        case=case.name,
        dataset=dataset.name,
        n_cells=metadata.n_cells,
        n_keys=0,
        key_kind="",
        replica=0,
        started_at=started_at,
        seconds=None,
        svg_bytes=0,
        status=f"skip:{reason}",
    )
    return [
        Row(library="cellestial", phase="construct", **common),
        Row(library="cellestial", phase="render", **common),
        Row(library="scanpy", phase="construct", **common),
        Row(library="scanpy", phase="render", **common),
    ]


def run(
    *,
    cases: list[Case],
    datasets: list[DatasetEntry],
    replicas: int,
    warmup: bool = True,
    skip_render: bool = False,
) -> pl.DataFrame:
    """Run the full grid and return the result frame."""
    all_rows: list[Row] = []
    total_combos = sum(
        len(case.expand(dataset.metadata()))
        for case in cases
        for dataset in datasets
        if _supports_dataset(case, dataset.metadata())
    )
    _console.section(
        f"running {len(cases)} cases × {len(datasets)} datasets → {total_combos} parameter combos"
    )

    io.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = io.RUNS_DIR / "_partial.feather"

    combo_index = 0
    for case in cases:
        _console.section(f"case '{case.name}'")
        for dataset in datasets:
            metadata = dataset.metadata()
            if not _supports_dataset(case, metadata):
                _console.warn(
                    f"skipping dataset '{dataset.name}' for case '{case.name}' "
                    f"(supports={case.supports}, dataset kind={metadata.kind})"
                )
                continue
            params_list = case.expand(metadata)
            if not params_list:
                _console.warn(
                    f"dataset '{dataset.name}' produces no params for case '{case.name}' (skipped)"
                )
                all_rows.extend(
                    _skip_row(
                        case=case,
                        dataset=dataset,
                        metadata=metadata,
                        reason="no_params",
                    )
                )
                continue

            data = dataset.load()
            for params in params_list:
                combo_index += 1
                _console.step(
                    f"[{combo_index}/{total_combos}] '{case.name}' on '{dataset.name}' / "
                    f"{params.slug} (n_keys={params.n_keys}, kind={params.key_kind})"
                )
                rows = _run_combo(
                    case=case,
                    dataset=dataset,
                    metadata=metadata,
                    params=params,
                    data=data,
                    replicas=replicas,
                    warmup=warmup,
                    skip_render=skip_render,
                )
                all_rows.extend(rows)
                _save_partial(all_rows, partial_path)

    frame = _rows_to_frame(all_rows)
    _console.section("results summary")
    _print_summary(frame)
    return frame


def _run_combo(
    *,
    case: Case,
    dataset: DatasetEntry,
    metadata: DatasetMetadata,
    params: Params,
    data: AnnData,
    replicas: int,
    warmup: bool,
    skip_render: bool,
) -> list[Row]:
    rows: list[Row] = []

    # warmup — discard results, but if it fails we record one error row per
    # library so the user can see the case is broken without running 3 replicas.
    if warmup:
        for library in ("cellestial", "scanpy"):
            _run_one(
                case=case,
                dataset=dataset,
                metadata=metadata,
                params=params,
                data=data,
                library=library,
                replica=-1,
                skip_render=skip_render,
                record_into=None,
            )

    for replica in range(replicas):
        for library in ("cellestial", "scanpy"):
            _run_one(
                case=case,
                dataset=dataset,
                metadata=metadata,
                params=params,
                data=data,
                library=library,
                replica=replica,
                skip_render=skip_render,
                record_into=rows,
            )

    _print_replica_summary(rows)
    return rows


def _run_one(
    *,
    case: Case,
    dataset: DatasetEntry,
    metadata: DatasetMetadata,
    params: Params,
    data: AnnData,
    library: str,
    replica: int,
    skip_render: bool,
    record_into: list[Row] | None,
) -> None:
    started_at = datetime.now(UTC)
    svg_path = io.svg_path(case.name, dataset.name, params.slug, library, max(replica, 0))

    if library == "cellestial":
        construct_fn = lambda: case.cellestial_fn(data, params, metadata)
        render_fn = lambda artifact: _render_cellestial(artifact, svg_path)
    else:
        construct_fn = lambda: case.scanpy_fn(data, params, metadata)
        render_fn = lambda artifact: _render_scanpy(artifact, svg_path)

    if skip_render:
        # Replace the render with a no-op so we still record a row, but no SVG.
        original_render = render_fn

        def _noop_render(artifact: object) -> int:
            _ = original_render  # static-analyzer placeholder
            return 0

        render_fn = _noop_render

    timing = _time_call(
        library=library,
        construct=construct_fn,
        render=render_fn,
        svg_target=svg_path,
    )

    if record_into is None:
        # warmup
        return

    rows = _make_rows(
        case=case,
        dataset=dataset,
        metadata=metadata,
        params=params,
        library=library,
        replica=replica,
        timing=timing,
        started_at=started_at,
    )
    record_into.extend(rows)


def _print_replica_summary(rows: list[Row]) -> None:
    """Print the mean ± sd construct/render per library for one combo."""
    import statistics

    by_library: dict[str, dict[str, list[float]]] = {}
    errors: dict[str, list[str]] = {}
    for row in rows:
        bucket = by_library.setdefault(row.library, {"construct": [], "render": []})
        if row.seconds is None:
            errors.setdefault(row.library, []).append(row.status)
            continue
        bucket[row.phase].append(row.seconds)

    for library in ("cellestial", "scanpy"):
        if library in errors:
            _console.fail(f"{library}: {errors[library][0]}")
            continue
        if library not in by_library:
            continue
        construct_list = by_library[library]["construct"]
        render_list = by_library[library]["render"]
        construct_mean = statistics.fmean(construct_list) if construct_list else float("nan")
        render_mean = statistics.fmean(render_list) if render_list else float("nan")
        total_mean = construct_mean + render_mean
        _console.kv(
            library,
            f"construct {_console.render_seconds(construct_mean)} | "
            f"render {_console.render_seconds(render_mean)} | "
            f"total {_console.render_seconds(total_mean)}",
        )


def _print_summary(frame: pl.DataFrame) -> None:
    """Print an end-of-run overview by case × library."""
    import polars as pl

    ok_only = frame.filter(pl.col("seconds").is_not_null())
    if ok_only.is_empty():
        _console.warn("no successful runs to summarize")
        return
    grouped = (
        ok_only.group_by(["case", "library", "phase"])
        .agg(pl.col("seconds").mean().alias("mean_seconds"))
        .sort(["case", "library", "phase"])
    )
    case_count = grouped.select("case").n_unique()
    library_count = grouped.select("library").n_unique()
    _console.kv("ok rows", ok_only.height)
    _console.kv("cases summarized", case_count)
    _console.kv("libraries summarized", library_count)
    error_rows = frame.filter(pl.col("status").str.starts_with("error"))
    if error_rows.height:
        _console.warn(f"{error_rows.height} error rows")
    skip_rows = frame.filter(pl.col("status").str.starts_with("skip"))
    if skip_rows.height:
        _console.info(f"{skip_rows.height} skip rows")


def _save_partial(rows: list[Row], path: Path) -> None:
    """Write a rolling partial snapshot so an interrupt does not lose progress."""
    try:
        frame = _rows_to_frame(rows)
        tmp = path.with_name(path.name + ".part")
        frame.write_ipc(tmp)
        tmp.replace(path)
    except Exception:
        # Partial saves are best-effort; never let them kill the run.
        pass


def _rows_to_frame(rows: list[Row]) -> pl.DataFrame:
    import polars as pl

    return pl.DataFrame(
        {
            "case": [row.case for row in rows],
            "library": [row.library for row in rows],
            "dataset": [row.dataset for row in rows],
            "n_cells": [row.n_cells for row in rows],
            "n_keys": [row.n_keys for row in rows],
            "key_kind": [row.key_kind for row in rows],
            "phase": [row.phase for row in rows],
            "replica": [row.replica for row in rows],
            "seconds": [row.seconds for row in rows],
            "svg_bytes": [row.svg_bytes for row in rows],
            "status": [row.status for row in rows],
            "started_at": [row.started_at for row in rows],
        },
        schema={
            "case": pl.String,
            "library": pl.String,
            "dataset": pl.String,
            "n_cells": pl.Int64,
            "n_keys": pl.Int64,
            "key_kind": pl.String,
            "phase": pl.String,
            "replica": pl.Int64,
            "seconds": pl.Float64,
            "svg_bytes": pl.Int64,
            "status": pl.String,
            "started_at": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )
