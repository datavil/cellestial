from __future__ import annotations

__lazy_modules__ = ["ggrust"]

from typing import TYPE_CHECKING, Any

from ggrust import element_rect, theme

from cellestial.util.errors import InvalidArgumentError

if TYPE_CHECKING:
    from ggrust.plot.core import FeatureSpec


_MAPPING_COLOR_HINT = (
    "Use `geom_color=` / `geom_fill=` for a constant, or map a column in `mapping`, "
    "e.g. `aes(color='leiden')`."
)


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


def _reject_constant_color_fill(geom_kwargs: dict[str, Any], *, function: str, hint: str) -> None:
    """
    Raise `InvalidArgumentError` when `color` / `fill` reach the geom as constants.

    `color` / `fill` name columns in cellestial; constants go through
    `geom_color` / `geom_fill`. `hint` is the second line of the message,
    saying what to use instead.
    """
    for aesthetic in ("color", "fill"):
        if aesthetic in geom_kwargs:
            msg = (
                f"`{aesthetic}={geom_kwargs[aesthetic]!r}` is not a parameter of `{function}`: "
                f"`color` / `fill` never set a constant here.\n{hint}"
            )
            raise InvalidArgumentError(msg)


def _reject_constant_over_mapping(
    *,
    geom_color: str | None,
    geom_fill: str | None,
    mapped_by: dict[str, str],
) -> None:
    """
    Raise `InvalidArgumentError` when a `geom_*` constant would hide a mapped aesthetic.

    `mapped_by` maps each aesthetic mapped to a column to what maps it, e.g.
    `{"color": "`key` 'leiden'"}`.
    """
    for aesthetic, constant in (("color", geom_color), ("fill", geom_fill)):
        if constant is not None and aesthetic in mapped_by:
            msg = (
                f"`geom_{aesthetic}={constant!r}` would override the `{aesthetic}` mapped "
                f"from {mapped_by[aesthetic]}, hiding that coloring.\n"
                f"Drop `geom_{aesthetic}` to color by the column, or drop the mapping "
                "to use the constant."
            )
            raise InvalidArgumentError(msg)
