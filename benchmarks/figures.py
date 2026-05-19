"""
Build the comparison figures from `results/results.feather`.

All figures use `theme_bw()` and the project's brand colors. Output goes to
`benchmarks/figures/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks import _console, io

if TYPE_CHECKING:
    import polars as pl


_LIBRARY_COLORS = {"cellestial": "#1f9e89", "scanpy": "#D2042D"}


def _load_results() -> "pl.DataFrame":
    import polars as pl

    path = io.RESULTS_DIR / "results.feather"
    if not path.exists():
        message = f"no results at {path}; run the benchmarks first"
        raise FileNotFoundError(message)
    frame = pl.read_ipc(path)
    return frame.filter(pl.col("seconds").is_not_null())


def _mean_frame(frame: "pl.DataFrame", *, group_cols: list[str]) -> "pl.DataFrame":
    import polars as pl

    return (
        frame.group_by(group_cols)
        .agg(
            pl.col("seconds").mean().alias("mean_seconds"),
            pl.col("seconds").std().alias("sd_seconds"),
            pl.col("seconds").count().alias("replicas"),
        )
        .sort(group_cols)
    )


def build_all() -> list[Path]:
    """Build all comparison figures. Returns a list of output paths."""
    from lets_plot.export import ggsave

    io.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame = _load_results()
    if frame.is_empty():
        _console.warn("results frame is empty; nothing to plot")
        return []

    outputs: list[Path] = []
    for name, plot in _build_plots(frame).items():
        path = io.FIGURES_DIR / f"{name}.svg"
        ggsave(plot, filename=path.name, path=str(path.parent), iframe=False)
        _console.ok(f"wrote {path}")
        outputs.append(path)
    return outputs


def _build_plots(frame: "pl.DataFrame") -> dict[str, object]:
    return {
        "scaling_by_data_size": _scaling_by_data_size(frame),
        "scaling_by_key_count": _scaling_by_key_count(frame),
        "construct_vs_render": _construct_vs_render(frame),
        "overall_summary": _overall_summary(frame),
    }


def _scaling_by_data_size(frame: "pl.DataFrame"):
    """Wall time vs n_cells, faceted by case, colored by library, linetype = phase."""
    import polars as pl
    from lets_plot import (
        aes,
        facet_wrap,
        geom_line,
        geom_point,
        ggplot,
        ggtitle,
        labs,
        scale_color_manual,
        scale_x_log10,
        scale_y_log10,
        theme_bw,
    )

    aggregated = _mean_frame(
        frame,
        group_cols=["case", "library", "phase", "dataset", "n_cells"],
    )
    # default param per case: smallest n_keys
    default_keys = (
        frame.group_by("case")
        .agg(pl.col("n_keys").min().alias("default_keys"))
    )
    # Filter to only those (case, n_keys) pairs to keep the plot interpretable
    filtered = aggregated.join(default_keys, on="case", how="inner").filter(
        pl.col("n_keys") == pl.col("default_keys")
    ) if "n_keys" in aggregated.columns else aggregated
    # The join above silently no-ops because n_keys was aggregated out — redo properly:
    base = frame.join(default_keys, on="case", how="inner").filter(
        pl.col("n_keys") == pl.col("default_keys")
    )
    aggregated = _mean_frame(
        base,
        group_cols=["case", "library", "phase", "dataset", "n_cells"],
    )
    pandas_frame = aggregated.to_pandas()

    return (
        ggplot(pandas_frame, aes(x="n_cells", y="mean_seconds", color="library"))
        + geom_line(aes(linetype="phase"))
        + geom_point(aes(shape="phase"), size=2)
        + facet_wrap("case", scales="free_y")
        + scale_x_log10()
        + scale_y_log10()
        + scale_color_manual(values=_LIBRARY_COLORS)
        + labs(x="cells (log10)", y="seconds (log10)")
        + ggtitle("Scaling by dataset size (default n_keys per case)")
        + theme_bw()
    )


def _scaling_by_key_count(frame: "pl.DataFrame"):
    """Wall time vs n_keys for heatmap-variant cases."""
    import polars as pl
    from lets_plot import (
        aes,
        facet_grid,
        geom_line,
        geom_point,
        ggplot,
        ggtitle,
        labs,
        scale_color_manual,
        theme_bw,
    )

    key_varying = ("dotplot", "heatmap", "matrixplot", "stacked_violin")
    base = frame.filter(pl.col("case").is_in(list(key_varying)))
    if base.is_empty():
        return _empty_plot("no key-varying cases ran")

    aggregated = _mean_frame(
        base,
        group_cols=["case", "library", "phase", "dataset", "n_keys"],
    )
    pandas_frame = aggregated.to_pandas()

    return (
        ggplot(pandas_frame, aes(x="n_keys", y="mean_seconds", color="library"))
        + geom_line(aes(linetype="phase"))
        + geom_point(aes(shape="phase"), size=2)
        + facet_grid(y="dataset", x="case", scales="free_y")
        + scale_color_manual(values=_LIBRARY_COLORS)
        + labs(x="number of keys", y="mean seconds (3 replicas)")
        + ggtitle("Scaling by key count (heatmap variants)")
        + theme_bw()
    )


def _construct_vs_render(frame: "pl.DataFrame"):
    """Stacked bar of construct vs render time per case, dodged by library."""
    import polars as pl
    from lets_plot import (
        aes,
        facet_wrap,
        geom_bar,
        ggplot,
        ggtitle,
        labs,
        scale_fill_manual,
        theme,
        theme_bw,
        element_text,
    )

    # Pick a single representative dataset per case (the smallest one we ran on)
    smallest = (
        frame.group_by(["case"])
        .agg(pl.col("n_cells").min().alias("smallest_n_cells"))
    )
    representative = frame.join(smallest, on="case").filter(
        pl.col("n_cells") == pl.col("smallest_n_cells")
    )
    # Within each case, pick the smallest n_keys
    smallest_keys = (
        representative.group_by("case")
        .agg(pl.col("n_keys").min().alias("smallest_keys"))
    )
    representative = representative.join(smallest_keys, on="case").filter(
        pl.col("n_keys") == pl.col("smallest_keys")
    )

    aggregated = _mean_frame(
        representative,
        group_cols=["case", "library", "phase"],
    )
    pandas_frame = aggregated.to_pandas()

    return (
        ggplot(pandas_frame, aes(x="library", y="mean_seconds", fill="phase"))
        + geom_bar(stat="identity")
        + facet_wrap("case", scales="free_y")
        + scale_fill_manual(values={"construct": "#377eb8", "render": "#e6550d"})
        + labs(x="library", y="mean seconds")
        + ggtitle("Construct vs render time (smallest dataset / n_keys per case)")
        + theme_bw()
        + theme(axis_text_x=element_text(angle=30))
    )


def _overall_summary(frame: "pl.DataFrame"):
    """Total time per case dodged by library, error bars from replicas."""
    import polars as pl
    from lets_plot import (
        aes,
        geom_bar,
        geom_errorbar,
        ggplot,
        ggtitle,
        labs,
        position_dodge,
        scale_fill_manual,
        theme,
        theme_bw,
        element_text,
    )

    by_replica = (
        frame.group_by(["case", "library", "dataset", "replica"])
        .agg(pl.col("seconds").sum().alias("total_seconds"))
    )
    aggregated = (
        by_replica.group_by(["case", "library"])
        .agg(
            pl.col("total_seconds").mean().alias("mean_total"),
            pl.col("total_seconds").std().alias("sd_total"),
        )
        .with_columns(
            (pl.col("mean_total") - pl.col("sd_total").fill_null(0)).alias("lower"),
            (pl.col("mean_total") + pl.col("sd_total").fill_null(0)).alias("upper"),
        )
        .sort(["case", "library"])
    )
    pandas_frame = aggregated.to_pandas()

    dodge = position_dodge(width=0.8)
    return (
        ggplot(pandas_frame, aes(x="case", y="mean_total", fill="library"))
        + geom_bar(stat="identity", position=dodge, width=0.7)
        + geom_errorbar(
            aes(ymin="lower", ymax="upper"),
            position=dodge,
            width=0.3,
        )
        + scale_fill_manual(values=_LIBRARY_COLORS)
        + labs(x="case", y="mean total seconds")
        + ggtitle("Overall: construct + render per case (averaged across datasets)")
        + theme_bw()
        + theme(axis_text_x=element_text(angle=40))
    )


def _empty_plot(message: str):
    from lets_plot import (
        aes,
        geom_text,
        ggplot,
        theme_bw,
    )
    import pandas as pd

    return (
        ggplot(pd.DataFrame({"x": [0], "y": [0], "label": [message]}), aes("x", "y"))
        + geom_text(aes(label="label"), size=8)
        + theme_bw()
    )
