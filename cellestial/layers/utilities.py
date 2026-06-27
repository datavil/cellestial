from __future__ import annotations

from typing import TYPE_CHECKING

from lets_plot import element_rect, theme

if TYPE_CHECKING:
    from lets_plot.plot.core import FeatureSpec


def borders(
    panel_color="green",
    legend_color="blue",
    plot_color="red",
    *,
    panel_size=4,
    legend_size=4,
    plot_size=4,
) -> FeatureSpec:
    """Add colored rectangles around panel, legend, and plot areas."""
    return theme(
        panel_background=element_rect(color=panel_color, size=panel_size),
        legend_background=element_rect(color=legend_color, size=legend_size),
        plot_background=element_rect(color=plot_color, size=plot_size),
    )
