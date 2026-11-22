from __future__ import annotations

__lazy_modules__ = ["anndata", "ggrust", "ggrust.plot.core", "mudata"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

# Core scverse libraries
import polars as pl
from anndata import AnnData

# Data retrieval
from ggrust import (
    aes,
    geom_point,
    ggplot,
    ggtb,
    labs,
    scale_color_brewer,
)
from ggrust.plot.core import FeatureSpec, LayerSpec, PlotSpec
from mudata import MuData

from cellestial.frames import build_frame
from cellestial.layers import _modify_axis, ondata_legend
from cellestial.layers.utilities import (
    _reject_constant_color_fill,
    _reject_constant_over_mapping,
    _reject_mapping_clash,
)
from cellestial.themes import _THEME_DIMENSION
from cellestial.util import (
    _collect_aes_columns,
    _color_gradient,
    _drop_nonfinite_rows,
    _filter_groups,
    _reject_renamed_embedding_kwargs,
    _reject_sequence_key,
    _require_feature_key,
    _resolve_embedding_key,
    _resolve_tooltips,
    _validate_tooltips,
    _warn,
)
from cellestial.util.errors import InvalidArgumentError, _unsupported_data_type

if TYPE_CHECKING:
    from ggrust.plot.core import PlotSpec
    from polars import DataFrame


_KEY_COLOR_HINT = (
    "Use `geom_color=` / `geom_fill=` for a constant, or `key=` to color by a column."
)


def _outline_layers(
    *,
    layer_mapping: FeatureSpec,
    size: float | None,
    outline_width: tuple[float, float] | Sequence[float],
    outline_color: tuple[str, str] | Sequence[str],
    geom_kwargs: dict,
) -> list[LayerSpec]:
    """
    Build the two point layers drawn under the points to outline the point cloud.

    The outline (bottom) and gap (middle) layers are the same points drawn
    larger, so only their rim shows around the cloud. Sizes follow scanpy:
    widths are fractions of the point size, added on each side. Both layers
    carry the points' mapping (constants override color, fill and size), so
    layers added with `+` that read the first layer's mapping still find it.
    """
    if size is None:
        msg = (
            "`outline=True` needs a constant `size`, but `size` is None or mapped by "
            "`mapping`.\nPass a number, e.g. `size=0.8`, and drop `size` from `mapping`."
        )
        raise InvalidArgumentError(msg)
    if size <= 0:
        msg = (
            f"`outline=True` needs a positive `size`, but received {size!r}.\n"
            "Pass a number above 0, e.g. `size=0.8`."
        )
        raise InvalidArgumentError(msg)
    if (
        not isinstance(outline_width, Sequence)
        or isinstance(outline_width, str)
        or len(outline_width) != 2
    ):
        msg = (
            f"`outline_width` must hold 2 numbers (outline, gap), but received "
            f"{outline_width!r}.\nPass e.g. `outline_width=(0.3, 0.05)`."
        )
        raise InvalidArgumentError(msg)
    outline_fraction, gap_fraction = outline_width
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
        for value in (outline_fraction, gap_fraction)
    ):
        msg = (
            f"`outline_width` values must be non-negative numbers, but received "
            f"{tuple(outline_width)!r}.\nPass e.g. `outline_width=(0.3, 0.05)`."
        )
        raise InvalidArgumentError(msg)
    if (
        not isinstance(outline_color, Sequence)
        or isinstance(outline_color, str)
        or len(outline_color) != 2
    ):
        msg = (
            f"`outline_color` must hold 2 colors (outline, gap), but received "
            f"{outline_color!r}.\nPass e.g. `outline_color=('black', 'white')`."
        )
        raise InvalidArgumentError(msg)
    if not all(isinstance(color, str) for color in outline_color):
        msg = (
            f"`outline_color` values must be color strings, but received "
            f"{tuple(outline_color)!r}.\nPass e.g. `outline_color=('black', 'white')`."
        )
        raise InvalidArgumentError(msg)
    outline_color_value, gap_color_value = outline_color

    gap_size = size + 2 * size * gap_fraction
    outline_size = gap_size + 2 * size * outline_fraction
    # Match the point shape so the rim follows the points' silhouette.
    shared = {k: v for k, v in geom_kwargs.items() if k in ("shape", "stroke")}
    return [
        geom_point(
            mapping=layer_mapping,
            size=layer_size,
            color=layer_color,
            fill=layer_color,
            # Opaque even when the points map `alpha`, like scanpy's outline.
            alpha=1,
            tooltips="none",
            show_legend=False,
            **shared,
        )
        for layer_size, layer_color in (
            (outline_size, outline_color_value),
            (gap_size, gap_color_value),
        )
    ]


def dimensional(
    data: AnnData | MuData,
    key: str | None = None,
    *,
    frame: DataFrame | None = None,
    mapping: FeatureSpec | None = None,
    dimensions: Literal["umap", "pca", "tsne"] | str = "umap",
    embedding_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    size: float | None = 0.8,
    geom_color: str | None = None,
    geom_fill: str | None = None,
    variable_keys: Sequence[str] | str | None = None,
    add_keys: Sequence[str] | str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observation_id_column: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    midpoint: Literal["mean", "median", "mid"] | float = "median",
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    legend_ondata: bool = False,
    ondata_size: float = 12,
    ondata_color: str = "#3f3f3f",
    ondata_fontface: str = "bold",
    ondata_family: str = "sans",
    ondata_alpha: float = 1,
    ondata_label: bool = False,
    halo_width: float | None = 0.5,
    halo_color: str | None = None,
    outline: bool = False,
    outline_width: tuple[float, float] = (0.3, 0.05),
    outline_color: tuple[str, str] = ("black", "white"),
    **geom_kwargs,
) -> PlotSpec:
    """
    Dimensionality reduction plot.

    Parameters
    ----------
    data : AnnData | MuData
        The single-cell data object.
    key : str, default=None
        The key (cell feature) to color the points by.
        e.g., 'leiden' or 'louvain' to color by clusters or gene name for expression.
    frame : DataFrame | None, default=None
        A prebuilt frame to plot from. If provided, the frame is used directly and
        building from `data` is skipped. Must contain the embedding and key columns.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    dimensions : str, default='umap'
        The dimensionality reduction to plot, named without the `X_` prefix.
        e.g., 'umap', 'pca', 'tsne', or a joint embedding such as 'wnn_umap'.
    xy : tuple[int, int] | Sequence[int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g., (1, 2) for UMAP1 and UMAP2.
    embedding_key : str | None, default=None
        Name of the stored embedding to plot, e.g. `X_umap_harmony`.
        If None, the embedding matching the plot type is used.
    size : float | None, default=0.8
        The size of the points.
    variable_keys : str | Sequence[str] | None, default=None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    add_keys : str | Sequence[str] | None, default=None
        Extra metadata columns or variable names to materialise into the frame,
        on top of those inferred from `key`, `mapping`, and `tooltips`. Useful
        when an added layer reads a column the plot itself does not reference.
    groups : str | Sequence[str] | None, default=None
        Show only specific groups, keeping points where `key` matches any of
        them. Categorical keys only.
    drop : str | Sequence[str] | None, default=None
        Drop specific groups, filtering out points where `key` matches any of
        them. Categorical keys only.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    observation_id_column : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    color_low : str, default='#e6e6e6'
        The color to use for the low end of the color gradient.
    color_mid : str, default=None
        The color to use for the middle part of the color gradient.
    color_high : str, default='#377EB8'
        The color to use for the high end of the color gradient.
    midpoint : {'mean', 'median', 'mid'} | float, default='median'
        The midpoint (in data value) of the color gradient.
        Can be 'mean', 'median' and 'mid' or a number (float or int).
        - If 'mean', the midpoint is the mean of the data.
        - If 'median', the midpoint is the median of the data.
        - If 'mid', the midpoint is the mean of 'min' and 'max' of the data.

    axis_type : {'axis', 'arrow'} | None
        Whether to use regular axis or arrows as the axis.
    arrow_length : float, default=0.25
        Span of each axis line as a fraction of the panel (0.25 covers 25%).
    arrow_size : float, default=1
        Width of the axis lines.
    arrow_color : str, default='#3f3f3f'
        Color of the arrows.
    arrow_angle : float, default=10
        Angle of the arrow head in degrees.
    legend_ondata: bool, default=False
        whether to show legend on data
    ondata_size: float, default=12
        size of the legend (text) on data.
    ondata_color: str, default='#3f3f3f'
        color of the legend (text) on data
    ondata_fontface: str, default='bold'
        fontface of the legend (text) on data.
        https://lets-plot.org/python/pages/aesthetics.html#font-face
    ondata_family: str, default='sans'
        family of the legend (text) on data.
        https://lets-plot.org/python/pages/aesthetics.html#font-family
    ondata_alpha: float, default=1
        alpha (transparency) of the legend on data.
    ondata_label: bool, default=False
        Whether to draw on-data legends with a filled label background.
    halo_width : float | None, default = 0.5
        Width of the text halo (text outline), not rendered when 0.
        Only applicable when on-data legend is used without background (i.e ondata_label=False).
    halo_color : str | None, default = None
        Color of the text halo (text outline).
        Only applicable when on-data legend is used without background (i.e ondata_label=False).
    outline : bool, default=False
        Whether to outline the point cloud: points are drawn on top of a gap
        layer and an outline layer, so only the rim of the cloud shows a border.
        Outlined points get `alpha=0.7` unless `alpha` is passed.
    outline_width : tuple[float, float], default=(0.3, 0.05)
        Widths of the outline and of the gap between outline and points, as
        fractions of the point size.
    outline_color : tuple[str, str], default=('black', 'white')
        Colors of the outline and of the gap.
    geom_color : str | None, default=None
        Color for all points in the plot. To color by a column, use `key`.
    geom_fill : str | None, default=None
        Fill color for all points in the plot (filled shapes 21 to 25 only).
    **geom_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensionality reduction plot.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    TypeError
        If `key` is a sequence of keys; use `dimensionals` for multiple keys.
    KeyNotFoundError
        If `key` is not found in observation metadata or variable names.
    InvalidArgumentError
        If `xy` does not hold exactly two components, selects a component the
        embedding does not have (components are 1-based), if `mapping` sets
        `x`, `y`, or `color` (with `key`), which the function sets itself, if
        `color` / `fill` are passed as constants (use `geom_color` /
        `geom_fill`), or if a `geom_color` / `geom_fill` constant would hide
        the coloring by `key` or `mapping`.
    UnknownParameterError
        If the removed `use_key` or `basis` is passed; use `embedding_key`.

    Examples
    --------
    Dimensionality reduction plot with categorical data.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.dimensional(data,key="cell_type_lvl1",axis_type="arrow",legend_ondata=True)

    With continuous data.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.dimensional(data,key="CD14",axis_type="arrow",color_high="red")

    Highlight a few groups over grey cells.

    .. jupyter-execute::

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.dimensional(data,key="leiden") + cl.highlight(["0","3"])

    """
    # HANDLE: Data types
    if not isinstance(data, (AnnData, MuData)):
        raise _unsupported_data_type(data, AnnData, MuData)

    # HANDLE: key
    _reject_sequence_key(key, singular="dimensional", plural="dimensionals")
    if frame is None or (key is not None and key not in frame.columns):
        _require_feature_key(data, key)

    # HANDLE: mapping
    _reject_mapping_clash(
        mapping,
        function="dimensional",
        set_by={"x": "`xy`", "y": "`xy`", **({"color": "`key`"} if key is not None else {})},
    )
    mapping = mapping or aes()
    _reject_constant_color_fill(
        geom_kwargs,
        function="dimensional",
        hint=_KEY_COLOR_HINT,
    )
    _reject_renamed_embedding_kwargs(geom_kwargs)
    mapped_by = {
        aesthetic: f"`mapping` column {column!r}"
        for aesthetic, column in mapping.as_dict().items()
        if aesthetic in ("color", "fill")
    }
    if key is not None:
        mapped_by["color"] = f"`key` {key!r}"
    _reject_constant_over_mapping(geom_color=geom_color, geom_fill=geom_fill, mapped_by=mapped_by)

    #  HANDLE: XY
    if len(xy) != 2:
        msg = (
            f"`xy` must hold exactly 2 components, but received {len(xy)}: {tuple(xy)}.\n"
            "Pass two 1-based components, e.g. `xy=(1, 2)`."
        )
        raise InvalidArgumentError(msg)
    prefix = _resolve_embedding_key(
        data=data, dimensions=dimensions, embedding_key=embedding_key, xy=xy
    )
    if min(xy) < 1:
        msg = (
            f"`xy={tuple(xy)}` selects a component below 1 of `{prefix}`.\n"
            "Components are 1-based: pass e.g. `xy=(1, 2)` for the first two."
        )
        raise InvalidArgumentError(msg)
    x = f"{prefix}{xy[0]}"  # e.g. X_UMAP1
    y = f"{prefix}{xy[1]}"  # e.g. X_UMAP2

    point_mapping = aes(
        x=x, y=y, **({"color": key} if key is not None else {}), **mapping.as_dict()
    )

    # HANDLE: outline (fail early, before building the frame)
    outline_layers: list[LayerSpec] = []
    if outline:
        outline_layers = _outline_layers(
            layer_mapping=point_mapping,
            size=None if "size" in mapping.as_dict() else size,
            outline_width=outline_width,
            outline_color=outline_color,
            geom_kwargs=geom_kwargs,
        )
        # Like scanpy, outlined points are slightly transparent unless set.
        if "alpha" not in mapping.as_dict():
            geom_kwargs.setdefault("alpha", 0.7)

    # HANDLE: variable_keys
    if variable_keys is None:
        variable_keys = []
    elif isinstance(variable_keys, str):
        variable_keys = [variable_keys]
    elif isinstance(variable_keys, Sequence):
        variable_keys = list(variable_keys)

    # Collect the frame columns from the color key, aes refs, and `add_keys`.
    if isinstance(add_keys, str):
        add_keys = [add_keys]
    metadata_columns: list[str] = []
    _collect_aes_columns(
        data,
        keys=[key, *(add_keys or [])],
        mapping=mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=0,
    )

    # HANDLE: tooltips
    tooltips = _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=[observation_id_column, *([key] if key is not None else [])],
        metadata_columns=metadata_columns,
        axis=0,
    )

    # BUILD: dataframe
    # All embeddings are materialised (not just the plotted one) so deferred
    # layers like `stream` can read velocity embeddings from the frame.
    if frame is None:
        # Skip the observation identifier column when no tooltip can reference it.
        observation_column_name = None if tooltips == "none" else observation_id_column
        frame = build_frame(
            data=data,
            variable_keys=variable_keys,
            axis=0,
            observation_id_column=observation_column_name,
            include_dimensions=max(xy),
            metadata_columns=metadata_columns,
        )

    # HANDLE: groups / drop filters (categorical-only)
    if key is not None:
        frame = _filter_groups(frame, column=key, groups=groups, drop=drop, column_parameter="key")

    # VALIDATE: both components exist
    if x not in frame.columns or y not in frame.columns:
        n_components = sum(
            1
            for column in frame.columns
            if column.startswith(prefix) and column.removeprefix(prefix).isdigit()
        )
        msg = (
            f"`xy={tuple(xy)}` selects components outside `{prefix}`, which has "
            f"{n_components} components.\nComponents are 1-based: pass values from 1 to "
            f"{n_components}, e.g. `xy=(1, 2)`."
            if n_components
            else f"Embedding `{prefix}` has no components in the plotted data.\n"
            "Check `embedding_key` / `dimensions`, or that a passed `frame` holds the embedding."
        )
        raise InvalidArgumentError(msg)

    frame = _drop_nonfinite_rows(frame, [x, y])
    _validate_tooltips(tooltips, frame)

    # BUILD: scatter plot
    if "size" in mapping.as_dict():
        size = None
    scttr = ggplot(data=frame)
    for layer in outline_layers:
        scttr += layer
    scttr = (
        scttr
        + geom_point(
            mapping=point_mapping,
            size=size,
            color=geom_color,
            fill=geom_fill,
            tooltips=tooltips,
            **geom_kwargs,
        )
        + _THEME_DIMENSION
    )

    if key is not None:
        # CASE1 ---------------------- CATEGORICAL DATA ----------------------
        if isinstance(frame[key].dtype, (pl.Categorical, pl.Enum)):
            scttr += scale_color_brewer(palette="Set2")

        # CASE2 ---------------------- CONTINUOUS DATA ----------------------
        elif frame[key].dtype.is_numeric():
            scttr += _color_gradient(
                frame[key],
                color_low=color_low,
                color_mid=color_mid,
                color_high=color_high,
                midpoint=midpoint,
            )
        # else: let letsplot handle it

    # HANDLE: tSNE label, a special case for labels
    if dimensions == "tsne":
        x_label = f"tSNE{xy[0]}"
        y_label = f"tSNE{xy[1]}"
        scttr += labs(x=x_label, y=y_label)
    elif dimensions == "pca":
        scttr += labs(x=f"PC{xy[0]}", y=f"PC{xy[1]}")
    else:
        # UMAP1 and UMAP2 rather than X_UMAP1 and X_UMAP2 etc.,
        scttr += labs(
            x=x.replace("X_", ""),
            y=y.replace("X_", ""),
        )

    # HANDLE: arrow axis
    scttr += _modify_axis(
        axis_type=axis_type,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        arrow_length=arrow_length,
    )
    # HANDLE: interactive
    if interactive:
        scttr += ggtb(size_zoomin=-1)

    # HANDLE: legend on data
    if key is not None and legend_ondata:
        if isinstance(frame[key].dtype, (pl.Categorical, pl.Enum)):
            scttr += ondata_legend(
                size=ondata_size,
                color=ondata_color,
                fontface=ondata_fontface,
                family=ondata_family,
                alpha=ondata_alpha,
                label=ondata_label,
                halo_width=halo_width,
                halo_color=halo_color,
            )
        else:
            msg = f"key `{key}` is not categorical, legend on data will not be added"
            _warn(msg)

    return scttr
