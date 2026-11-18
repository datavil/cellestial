from __future__ import annotations

__lazy_modules__ = ["ggrust"]

from typing import TYPE_CHECKING

from ggrust import element_rect, theme

from cellestial.util.errors import InvalidArgumentError

if TYPE_CHECKING:
    from ggrust.plot.core import FeatureSpec


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


def _reject_mapping_clash(
    mapping: FeatureSpec | None, *, function: str, set_by: dict[str, str]
) -> None:
    """
    Raise `InvalidArgumentError` when `mapping` sets an aesthetic the function sets itself.

    `set_by` maps each aesthetic the function sets to the parameter (or role)
    that sets it, e.g. `{"color": "key"}`.
    """
    if mapping is None:
        return
    clashes = [aesthetic for aesthetic in mapping.as_dict() if aesthetic in set_by]
    if clashes:
        aesthetic = clashes[0]
        msg = (
            f"`mapping` sets `{aesthetic}`, which `{function}` already sets from "
            f"{set_by[aesthetic]}.\nRemove `{aesthetic}` from `mapping`"
            f"; the aesthetics `{function}` sets are {sorted(set_by)}."
        )
        raise InvalidArgumentError(msg)
