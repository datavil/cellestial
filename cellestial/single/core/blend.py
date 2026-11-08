from __future__ import annotations

__lazy_modules__ = ["anndata", "ggrust", "ggrust.plot.core", "mudata"]

import re
from difflib import get_close_matches
from typing import TYPE_CHECKING, Literal

import polars as pl
from anndata import AnnData
from ggrust import (
    aes,
    coord_fixed,
    geom_point,
    geom_tile,
    gggrid,
    ggplot,
    ggtb,
    labs,
    scale_color_identity,
    scale_fill_identity,
    scale_x_continuous,
    scale_y_continuous,
)
from mudata import MuData

from cellestial.frames import build_frame
from cellestial.frames._container import _container
from cellestial.layers import _modify_axis
from cellestial.single.core.subdimensional import expression
from cellestial.themes import _THEME_DIMENSION
from cellestial.util import _drop_nonfinite_rows, _resolve_embedding_key, _resolve_tooltips
from cellestial.util.errors import (
    ConflictingLengthError,
    DuplicateKeysError,
    InvalidColorError,
    VariableNotFoundError,
    _unsupported_data_type,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ggrust.plot.core import FeatureSpec, PlotSpec
    from ggrust.plot.subplots import SupPlotsSpec

_BLEND_COLUMN = "blend_color"
_KEY_LEVELS = 32
_HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_HEX_DIGITS = [f"{value:02x}" for value in range(256)]


def blend(
    data: AnnData | MuData,
    keys: tuple[str, str] | Sequence[str],
    *,
    dimensions: Literal["umap", "pca", "tsne"] | str = "umap",
    use_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    size: float | None = 0.8,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observation_id_column: str = "Barcode",
    color_low: str = "#e6e6e6",
    colors: tuple[str, str] | Sequence[str] = ("#ff0000", "#00ff00"),
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    ncol: int | None = None,
    **geom_kwargs,
) -> SupPlotsSpec:
    """
    Co-expression plot that blends the colors of two features on an embedding.

    Parameters
    ----------
    data : AnnData | MuData
        The single-cell data object.
    keys : tuple[str, str] | Sequence[str]
        The two variable (gene) names to blend, e.g. `("CD3E", "CD8A")`.
    dimensions : str, default='umap'
        The dimensionality reduction to plot, named without the `X_` prefix.
        e.g., 'umap', 'pca', 'tsne', or a joint embedding such as 'wnn_umap'.
    use_key : str, default=None
        The specific key to use for the desired dimensions.
        e.g., 'X_umap_2d' or 'X_pca_2d'.
        Otherwise, the function will decide on the key based on the dimensions.
    xy : tuple[int, int] | Sequence[int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g., (1, 2) for UMAP1 and UMAP2.
    size : float | None, default=0.8
        The size of the points.
    tooltips : {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the points.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    interactive : bool, default=False
        Whether to make the plots interactive.
    observation_id_column : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    color_low : str, default='#e6e6e6'
        Hex color of cells that express neither feature.
    colors : tuple[str, str] | Sequence[str], default=('#ff0000', '#00ff00')
        Hex colors of the first and the second feature at their highest expression.
    axis_type : {'axis', 'arrow'} | None
        Whether to use regular axis or arrows as the axis.
    arrow_length : float, default=0.25
        Span of each axis line as a fraction of its data range (0.25 covers 25%).
    arrow_size : float, default=1
        Width of the axis lines.
    arrow_color : str, default='#3f3f3f'
        Color of the arrows.
    arrow_angle : float, default=10
        Angle of the arrow head in degrees.
    ncol : int | None, default=None
        Number of columns in the grid. All four panels sit in one row by default.
    **geom_kwargs
        Additional parameters for the `geom_point` layers.

    Returns
    -------
    SupPlotsSpec
        Grid of four panels: the first feature, the second feature, the blend, and
        the two-dimensional color key.

    Raises
    ------
    ConflictingLengthError
        If `keys` or `colors` does not hold exactly two values.
    DuplicateKeysError
        If both `keys` name the same feature.
    VariableNotFoundError
        If a key is not a variable (gene) name.
    InvalidColorError
        If `color_low` or a value of `colors` is not a hex color.

    Notes
    -----
    Each feature is scaled to 0 (its minimum) to 1 (its maximum), giving levels `a`
    and `b`. Following Seurat's blend, each cell's red, green and blue channels are
    `low + a * (first - low) + b * (second - low) + a * b * low`, clipped to the
    color range, where `low`, `first` and `second` are the channels of `color_low`
    and `colors`. The levels stay continuous (no binning or threshold). With the
    default red and green, cells expressing only the first feature are red, only
    the second green, both yellow, and neither grey.
    Cells with a missing or non-finite value for either feature are left out of
    every panel.

    Examples
    --------
    Two co-expressed T cell markers.

    .. jupyter-execute::

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.blend(data, keys=("CD3E", "CD8A"))

    A T cell marker and a B cell marker, rarely expressed together.

    .. jupyter-execute::

        cl.blend(data, keys=("CD3E", "MS4A1"), colors=("#ff0000", "#0000ff"))
    """
    # HANDLE: Data types
    if not isinstance(data, (AnnData, MuData)):
        raise _unsupported_data_type(data, AnnData, MuData)

    # HANDLE: keys and colors
    keys = [keys] if isinstance(keys, str) else list(keys)
    if len(keys) != 2:
        msg = f"`keys` needs exactly two features to blend, got {len(keys)}: {keys}. Pass two names, e.g. `keys=('CD3E', 'CD8A')`."
        raise ConflictingLengthError(msg)
    first_key, second_key = keys
    if first_key == second_key:
        msg = f"`keys` names '{first_key}' twice. Pass two different features, e.g. `keys=('CD3E', 'CD8A')`."
        raise DuplicateKeysError(msg)
    for key in keys:
        _require_variable(data, key)
    colors = [colors] if isinstance(colors, str) else list(colors)
    if len(colors) != 2:
        msg = f"`colors` needs exactly two colors, one per key, got {len(colors)}: {colors}. Pass e.g. `colors=('#ff0000', '#00ff00')`."
        raise ConflictingLengthError(msg)
    first_color, second_color = colors
    low_rgb = _hex_to_rgb(color_low, parameter="color_low")
    first_rgb = _hex_to_rgb(first_color, parameter="colors")
    second_rgb = _hex_to_rgb(second_color, parameter="colors")

    # HANDLE: XY
    if len(xy) != 2:
        msg = f"xy MUST be of length 2, (len(xy)=={len(xy)})"
        raise KeyError(msg)
    prefix = _resolve_embedding_key(data=data, dimensions=dimensions, use_key=use_key, xy=xy)
    x = f"{prefix}{xy[0]}"
    y = f"{prefix}{xy[1]}"

    # HANDLE: tooltips
    variable_keys = list(keys)
    metadata_columns: list[str] = []
    tooltips = _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=[observation_id_column, first_key, second_key],
        metadata_columns=metadata_columns,
        axis=0,
    )

    # BUILD: dataframe
    observation_column_name = None if tooltips == "none" else observation_id_column
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=0,
        observation_id_column=observation_column_name,
        include_dimensions=max(xy),
        metadata_columns=metadata_columns,
    )
    frame = _drop_nonfinite_rows(frame, [x, y, first_key, second_key])
    frame = frame.with_columns(
        _blend_color(
            _scaled(pl.col(first_key)),
            _scaled(pl.col(second_key)),
            low_rgb=low_rgb,
            first_rgb=first_rgb,
            second_rgb=second_rgb,
        ).alias(_BLEND_COLUMN)
    )

    # BUILD: feature panels
    first_panel, second_panel = (
        expression(
            data,
            key,
            frame=frame,
            dimensions=dimensions,
            use_key=use_key,
            xy=xy,
            size=size,
            tooltips=tooltips,
            observation_id_column=observation_id_column,
            color_low=color_low,
            color_high=color,
            axis_type=axis_type,
            arrow_length=arrow_length,
            arrow_size=arrow_size,
            arrow_color=arrow_color,
            arrow_angle=arrow_angle,
            **geom_kwargs,
        )
        for key, color in ((first_key, first_color), (second_key, second_color))
    )

    # BUILD: blend panel
    axis_labels = {"x": x.replace("X_", ""), "y": y.replace("X_", "")}
    if dimensions == "tsne":
        axis_labels = {"x": f"tSNE{xy[0]}", "y": f"tSNE{xy[1]}"}
    blend_panel = (
        ggplot(data=frame)
        + geom_point(
            mapping=aes(x=x, y=y, color=_BLEND_COLUMN),
            size=size,
            tooltips=tooltips,
            **geom_kwargs,
        )
        + scale_color_identity()
        + _THEME_DIMENSION
        + labs(title=f"{first_key} + {second_key}", **axis_labels)
        + _modify_axis(
            axis_type=axis_type,
            arrow_size=arrow_size,
            arrow_color=arrow_color,
            arrow_angle=arrow_angle,
            arrow_length=arrow_length,
        )
    )

    # BUILD: color key
    color_key = _color_key(
        first_key,
        second_key,
        low_rgb=low_rgb,
        first_rgb=first_rgb,
        second_rgb=second_rgb,
    )

    grid = gggrid([first_panel, second_panel, blend_panel, color_key], ncol=ncol)
    if interactive:
        grid += ggtb(size_zoomin=-1)
    return grid


def _require_variable(data: AnnData | MuData, key: str) -> None:
    """Raise `VariableNotFoundError` with a close match when `key` is not a variable name."""
    container = _container(data)
    if container.owns_variable(key):
        return
    names = [str(name) for name in container.variable_names()]
    matches = get_close_matches(key, names, n=1)
    hint = f" Did you mean '{matches[0]}'?" if matches else ""
    msg = f"`keys` entry '{key}' is not a variable (gene) name.{hint} `blend` accepts two variable names."
    raise VariableNotFoundError(msg)


def _hex_to_rgb(color: str, *, parameter: str) -> tuple[int, int, int]:
    """Parse a '#rgb' or '#rrggbb' color into its red, green and blue channels."""
    if not isinstance(color, str) or not _HEX_COLOR.match(color):
        msg = (
            f"`{parameter}` got {color!r}, but `blend` mixes hex colors only. Pass e.g. '#ff0000'."
        )
        raise InvalidColorError(msg)
    digits = color[1:]
    if len(digits) == 3:
        digits = "".join(digit * 2 for digit in digits)
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def _scaled(value: pl.Expr) -> pl.Expr:
    """Min-max scale to 0..1; a constant feature scales to 0."""
    value = value.cast(pl.Float64)
    value_range = value.max() - value.min()
    return pl.when(value_range > 0).then((value - value.min()) / value_range).otherwise(0.0)


def _blend_color(
    first_level: pl.Expr,
    second_level: pl.Expr,
    *,
    low_rgb: tuple[int, int, int],
    first_rgb: tuple[int, int, int],
    second_rgb: tuple[int, int, int],
) -> pl.Expr:
    """Hex color adding both feature colors onto the low color, clipped per channel."""
    hex_digits = pl.Series(_HEX_DIGITS)
    channels = []
    for low, first, second in zip(low_rgb, first_rgb, second_rgb, strict=True):
        channel = (
            low
            + first_level * (first - low)
            + second_level * (second - low)
            + first_level * second_level * low
        )
        channel = channel.clip(0, 255).round().cast(pl.Int64)
        channels.append(pl.lit(hex_digits).gather(channel))
    return pl.concat_str([pl.lit("#"), *channels])


def _color_key(
    first_key: str,
    second_key: str,
    *,
    low_rgb: tuple[int, int, int],
    first_rgb: tuple[int, int, int],
    second_rgb: tuple[int, int, int],
) -> PlotSpec:
    """Two-dimensional legend: the blend color of every pair of scaled levels."""
    levels = pl.Series("level", [index / (_KEY_LEVELS - 1) for index in range(_KEY_LEVELS)])
    key_frame = (
        levels.to_frame("first_level")
        .join(levels.to_frame("second_level"), how="cross")
        .with_columns(
            _blend_color(
                pl.col("first_level"),
                pl.col("second_level"),
                low_rgb=low_rgb,
                first_rgb=first_rgb,
                second_rgb=second_rgb,
            ).alias(_BLEND_COLUMN)
        )
    )
    breaks = [0, 0.5, 1]
    return (
        ggplot(data=key_frame)
        + geom_tile(
            mapping=aes(x="first_level", y="second_level", fill=_BLEND_COLUMN),
            tooltips="none",
        )
        + scale_fill_identity()
        + scale_x_continuous(breaks=breaks, expand=[0, 0])
        + scale_y_continuous(breaks=breaks, expand=[0, 0])
        + coord_fixed()
        + _THEME_DIMENSION
        + labs(title="Color key", x=first_key, y=second_key)
    )
