"""
Plot benchmark results with lets-plot: scaling curves (log-log) and bars.

poetry run python -m benchmarks.plot_results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from lets_plot import (
    aes,
    facet_wrap,
    geom_bar,
    geom_line,
    geom_point,
    ggplot,
    ggsave,
    ggsize,
    ggtitle,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_log10,
    scale_y_log10,
    theme_classic,
)

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

FRAMEWORK_COLORS = {"cellestial": "#377eb8", "scanpy": "#e41a1c"}


def _scaling_plot(df: pl.DataFrame, phase: str, out_path: Path) -> None:
    blobs = df.filter(pl.col("dataset").str.starts_with("blobs_") & (pl.col("phase") == phase))
    if blobs.is_empty():
        print(f"no blobs rows for phase={phase}; skipping")
        return

    plot = (
        ggplot(blobs, aes(x="n_cells", y="time_median_s", color="framework"))
        + geom_line(size=1)
        + geom_point(size=3)
        + scale_x_log10()
        + scale_y_log10()
        + scale_color_manual(
            values=[FRAMEWORK_COLORS["cellestial"], FRAMEWORK_COLORS["scanpy"]],
            breaks=["cellestial", "scanpy"],
        )
        + facet_wrap("case", scales="free_y")
        + labs(x="n cells", y=f"{phase} time (s)")
        + ggtitle(f"Scaling, {phase}")
        + theme_classic()
        + ggsize(1100, 450)
    )
    ggsave(plot, filename=out_path.name, path=str(out_path.parent))
    print(f"wrote {out_path}")


def _bar_plot(df: pl.DataFrame, dataset: str, out_path: Path) -> None:
    sub = df.filter((pl.col("dataset") == dataset) & (pl.col("phase") == "total"))
    if sub.is_empty():
        return

    plot = (
        ggplot(sub, aes(x="case", y="time_median_s", fill="framework"))
        + geom_bar(stat="identity", position="dodge")
        + scale_fill_manual(
            values=[FRAMEWORK_COLORS["cellestial"], FRAMEWORK_COLORS["scanpy"]],
            breaks=["cellestial", "scanpy"],
        )
        + labs(x="", y="total time (s)")
        + ggtitle(f"Total time per case, {dataset}")
        + theme_classic()
        + ggsize(800, 420)
    )
    ggsave(plot, filename=out_path.name, path=str(out_path.parent))
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=RESULTS_DIR / "results.csv")
    ap.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    args = ap.parse_args()

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_csv(args.input)

    for phase in ("construct", "render", "total"):
        _scaling_plot(df, phase, args.plots_dir / f"scaling_{phase}.svg")

    for dataset in df["dataset"].unique().to_list():
        if not dataset.startswith("blobs_"):
            _bar_plot(df, dataset, args.plots_dir / f"bars_{dataset}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
