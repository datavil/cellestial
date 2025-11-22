from __future__ import annotations

import polars as pl
from lets_plot import aes, geom_text, theme
from lets_plot.plot.core import FeatureSpec, FeatureSpecArray


def _legend_ondata(
    *,
    frame: pl.DataFrame,
    x: str,
    y: str,
    cluster_name: str,
    size: float = 12,
    color: str = "#3f3f3f",
    fontface: str = "bold",
    family: str = "sans",
    alpha: float = 1,
    weighted: bool = True,
) -> FeatureSpec | FeatureSpecArray:
    # group by cluster names and find X and Y mean for midpoints

    if weighted:
        group_means = frame.group_by(cluster_name).agg(
            pl.col(x).mean().alias("mean_x"), pl.col(y).mean().alias("mean_y")
        )
        # join the group means to the frame
        frame = frame.join(group_means, on=cluster_name, how="left")
        # calculate the distance between the group means and the frame
        frame = frame.with_columns(
            ((pl.col(x) - pl.col("mean_x")) ** 2 + (pl.col(y) - pl.col("mean_y")) ** 2)
            .sqrt()
            .alias("distance")
        )
        # assign weights to the individual points
        frame = frame.with_columns((1 / pl.col("distance").sqrt()).alias("weight"))
        # calculate the weighted mean of the group means
        grouped = frame.group_by(cluster_name).agg(
            (pl.col(x) * pl.col("weight")).sum() / pl.col("weight").sum(),
            (pl.col(y) * pl.col("weight")).sum() / pl.col("weight").sum(),
        )
    else:
        grouped = frame.group_by(cluster_name).agg(pl.col(x).mean(), pl.col(y).mean())
    return geom_text(
        data=grouped,
        mapping=aes(x=x, y=y, label=cluster_name),
        size=size,
        color=color,
        fontface=fontface,
        family=family,
        alpha=alpha,
    ) + theme(legend_position="none")
