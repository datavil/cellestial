"""
Plot cellestial (ggrust) vs scanpy benchmark results with ggrust.

Run from the repo root:

    poetry run python benchmarks/plot_results.py

Writes PNGs to benchmarks/results/.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from ggrust import (
    aes,
    coord_flip,
    element_blank,
    facet_wrap,
    geom_col,
    geom_hline,
    geom_line,
    geom_point,
    geom_text,
    ggplot,
    ggsave,
    ggsize,
    labs,
    position_dodge,
    scale_color_manual,
    scale_fill_manual,
    scale_x_log10,
    scale_y_log10,
    theme,
    theme_bw,
)

BENCHMARKS = Path(__file__).resolve().parent
OUTPUT = BENCHMARKS / "results"
COLORS = {"cellestial": "#3B6FB6", "scanpy": "#E07B39"}
GRID_CASES = ["heatmap", "dotplot", "matrixplot"]


def _load() -> pl.DataFrame:
    """Return per-configuration medians for cellestial (ggrust) and scanpy."""
    frame = (
        pl.read_ipc(BENCHMARKS / "results.feather", memory_map=False)
        .filter(pl.col("library").is_in(["cellestial-ggrust", "scanpy"]))
        .with_columns(pl.col("library").str.replace("-ggrust", ""))
    )
    keys = ["library", "case", "categorical", "dataset", "n_obs", "n_cols"]
    return frame.group_by(keys).agg(
        pl.col("time(s)").median().alias("time"),
        pl.col("memory(MB)").median().alias("memory"),
    )


def _label(frame: pl.DataFrame) -> pl.DataFrame:
    """Add a readable configuration label: case, key kind or gene count."""
    detail = (
        pl.when(pl.col("case").is_in(GRID_CASES))
        .then(pl.format("{} genes", pl.col("n_cols")))
        .when(pl.col("categorical") == 1)
        .then(pl.lit("categorical"))
        .when(pl.col("case") == "violin")
        .then(pl.lit("1 gene"))
        .otherwise(pl.lit("gene"))
    )
    return frame.with_columns(pl.format("{} ({})", pl.col("case"), detail).alias("configuration"))


def speedup_plot(frame: pl.DataFrame):
    """Bar chart of scanpy time divided by cellestial time, per configuration."""
    wide = (
        _label(frame)
        .pivot("library", index=["configuration", "case", "n_cols", "dataset"], values="time")
        .with_columns((pl.col("scanpy") / pl.col("cellestial")).alias("speedup"))
        .with_columns(pl.format("{}x", pl.col("speedup").round(1)).alias("speedup_label"))
        .sort("case", "n_cols", descending=[True, True])
    )
    order = wide["configuration"].unique(maintain_order=True).to_list()
    return (
        ggplot(wide, aes(x="configuration", y="speedup"))
        + geom_col(aes(fill="dataset"), position=position_dodge(width=0.85), width=0.8)
        + geom_text(
            aes(label="speedup_label", group="dataset"),
            position=position_dodge(width=0.85),
            hjust=-0.15,
            size=4,
        )
        + geom_hline(yintercept=1, linetype="dashed", color="#555555")
        + scale_y_log10(limits=[1, 50], expand=[0, 0])
        + coord_flip()
        + labs(
            x="",
            y="speedup (scanpy time / cellestial time, log scale)",
            fill="dataset",
            title="cellestial vs scanpy: time speedup",
            subtitle="median of 3 replicas, SVG render; right of the dashed line = cellestial faster",
        )
        + _base_theme()
        + theme(panel_grid_major_y=element_blank())
        + ggsize(1000, 900)
        + _x_discrete(order)
    )


def _x_discrete(order: list[str]):
    """Keep configurations in the given order (cases grouped, genes ascending)."""
    from ggrust import scale_x_discrete

    return scale_x_discrete(limits=order)


def scaling_plot(frame: pl.DataFrame, value: str, axis_title: str, title: str):
    """Time or memory against number of genes for the multi-gene cases."""
    grid = frame.filter(pl.col("case").is_in(GRID_CASES))
    return (
        ggplot(grid, aes(x="n_cols", y=value, color="library"))
        + geom_line(aes(linetype="dataset"), size=1)
        + geom_point(aes(shape="dataset"), size=3)
        + facet_wrap("case", ncol=3)
        + scale_x_log10(breaks=[5, 20, 100, 200], limits=[4, 260])
        + scale_y_log10()
        + scale_color_manual(values=COLORS)
        + labs(x="number of genes (log scale)", y=axis_title, title=title)
        + _base_theme()
        + theme(legend_position="right")
        + ggsize(1150, 450)
    )


def single_key_plot(frame: pl.DataFrame, value: str, axis_title: str, title: str):
    """Side-by-side bars for the one-key cases (umap, spatial, violin)."""
    single = _label(frame.filter(~pl.col("case").is_in(GRID_CASES)))
    return (
        ggplot(single, aes(x="configuration", y=value, fill="library"))
        + geom_col(position=position_dodge(width=0.85), width=0.8)
        + facet_wrap("dataset", ncol=2, scales="free")
        + scale_fill_manual(values=COLORS)
        + labs(x="", y=axis_title, title=title)
        + _base_theme()
        + ggsize(1000, 650)
    )


def _base_theme():
    """Shared theme: white background, legend at the bottom."""
    return theme_bw() + theme(legend_position="bottom")


def main() -> None:
    """Render every benchmark figure to PNG."""
    OUTPUT.mkdir(exist_ok=True)
    frame = _load()
    figures = {
        "speedup.png": speedup_plot(frame),
        "scaling_time.png": scaling_plot(frame, "time", "time (s, log scale)", "Time vs genes"),
        "scaling_memory.png": scaling_plot(
            frame, "memory", "peak memory (MB, log scale)", "Peak memory vs genes"
        ),
        "single_key_time.png": single_key_plot(frame, "time", "time (s)", "Time, one-key plots"),
        "single_key_memory.png": single_key_plot(
            frame, "memory", "peak memory (MB)", "Peak memory, one-key plots"
        ),
    }
    for name, plot in figures.items():
        print(ggsave(plot, name, path=str(OUTPUT)))


if __name__ == "__main__":
    main()
