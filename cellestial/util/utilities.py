import polars as pl
from lets_plot import *
from lets_plot import (
    LetsPlot,
    aes,
    arrow,
    element_blank,
    element_line,
    element_text,
    geom_blank,
    geom_jitter,
    geom_point,
    geom_segment,
    geom_violin,
    gggrid,
    ggplot,
    ggsize,
    ggtb,
    guide_colorbar,
    guide_legend,
    guides,
    labs,
    layer_tooltips,
    scale_color_brewer,
    scale_color_continuous,
    scale_color_gradient,
    scale_color_hue,
    scale_color_viridis,
    scale_shape,
    theme,
    theme_classic,
)


def _add_arrow_axis(
    frame: pl.DataFrame,
    *,
    axis_type: str | None,
    arrow_size: float,
    arrow_color: str,
    arrow_angle: float,
    arrow_length:float,
    dimensions:str
):
    if axis_type is None:
        return theme(
            # remove axis elements
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks_y=element_blank(),
            axis_ticks_x=element_blank(),
            axis_line=element_blank(),
        )

    elif axis_type == "axis":
        return geom_blank()

    elif axis_type == "arrow":
        new_layer = theme(
            # remove axis elements
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks_y=element_blank(),
            axis_ticks_x=element_blank(),
            axis_line=element_blank(),
            # position axis titles according to arrow size
            axis_title_x=element_text(hjust=arrow_length / 2),
            axis_title_y=element_text(hjust=arrow_length / 2),
        )
        x_max = frame.select(f"{dimensions}1").max().item()
        x_min = frame.select(f"{dimensions}1").min().item()
        y_max = frame.select(f"{dimensions}2").max().item()
        y_min = frame.select(f"{dimensions}2").min().item()

        # find total difference between the max and min for both axis
        x_diff = x_max - x_min
        y_diff = y_max - y_min

        # find the ends of the arrows
        xend = x_min + arrow_length * x_diff
        yend = y_min + arrow_length * y_diff

        # adjust bottom ends of arrows
        adjust_rate = 0.025
        x0 = x_min - x_diff * adjust_rate
        y0 = y_min - y_diff * adjust_rate

        # X axis
        new_layer += geom_segment(
            x=x0,
            y=y0,
            xend=xend,
            yend=y0,
            color=arrow_color,
            size=arrow_size,
            arrow=arrow(arrow_angle),
        )
        # Y axis
        new_layer += geom_segment(
            x=x0,
            y=y0,
            xend=x0,
            yend=yend,
            color=arrow_color,
            size=arrow_size,
            arrow=arrow(arrow_angle),
        )
    else:
        msg = f"expected 'axis' or 'arrow' for 'axis_type' argument, but received {axis_type}"
        raise ValueError(msg)

    return new_layer
