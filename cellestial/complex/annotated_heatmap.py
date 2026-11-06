from __future__ import annotations

__lazy_modules__ = ["anndata", "ggrust", "mudata", "numpy"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from anndata import AnnData
from ggrust import (
    aes,
    coord_cartesian,
    element_text,
    geom_raster,
    geom_segment,
    geom_tile,
    geom_xsideline,
    geom_xsidetile,
    geom_ysideline,
    geom_ysidetile,
    ggplot,
    ggside,
    ggtb,
    new_scale_fill,
    scale_fill_brewer,
    scale_fill_gradient,
    scale_fill_gradientn,
    scale_fill_manual,
    scale_x_continuous,
    scale_xsidey_discrete,
    scale_y_continuous,
    scale_ysidex_discrete,
    theme,
)
from mudata import MuData

from cellestial.frames import build_frame
from cellestial.single.heatmap.utilities import _scale_values
from cellestial.themes import _THEME_HEATMAP
from cellestial.util import _fill_gradient, _get_dendrogram, _warn
from cellestial.util.errors import _unsupported_data_type
from cellestial.util.utilities import _modality_source

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ggrust.plot.core import FeatureSpec, PlotSpec

    AnnotationColors = Mapping[str, str] | Sequence[str]

# Width of the dendrogram, relative to the heatmap.
_DENDROGRAM_WIDTH = 0.15


# ColorBrewer palettes cycled across annotation tracks so each gets a distinct one.
# Qualitative palettes come first (better for categorical tracks), then diverging.
_BREWER_PALETTES = (
    "Set1",
    "Set2",
    "Set3",
    "Spectral",
    "Accent",
    "Dark2",
    "Paired",
    "Pastel1",
    "Pastel2",
    "BrBG",
    "PiYG",
    "PRGn",
    "PuOr",
    "RdBu",
    "RdGy",
    "RdYlBu",
    "RdYlGn",
)

# Legend element sizing. `title` carries the bold: lets-plot resolves a legend title's `face` from the
# parent `title` element, so setting `face` on `legend_title` alone has no effect.
# Axis titles are blanked here, so this only bolds the (legend) titles.
_LEGEND_THEME = theme(
    legend_key_size=10,
    legend_text=element_text(size=9),
    legend_title=element_text(size=10),
    title=element_text(face="bold"),
)


def _as_list(keys: str | Sequence[str] | None) -> list[str]:
    """Normalize a single key, a sequence of keys, or `None` into a list."""
    if keys is None:
        return []
    if isinstance(keys, str):
        return [keys]
    return list(keys)


def _as_layers(layers: FeatureSpec | Sequence[FeatureSpec] | None) -> list[FeatureSpec]:
    """Normalize a single layer, a sequence of layers, or `None` into a list."""
    if layers is None:
        return []
    if isinstance(layers, (list, tuple)):
        return list(layers)  # ty:ignore[invalid-return-type]
    return [layers]  # ty:ignore[invalid-return-type]


def _apply_layers(plot: PlotSpec, layers: Sequence[FeatureSpec]) -> PlotSpec:
    """Append each layer to `plot`, returning the extended spec."""
    for layer in layers:
        plot = plot + layer
    return plot


def _order_observations(
    frame: pl.DataFrame, *, group_by: str | None, group_order: list[str] | None = None
) -> tuple[pl.DataFrame, list[str] | None]:
    """
    Sort rows into contiguous `group_by` blocks.

    Notes
    -----
    Returns the sorted frame and the group order (first group ends up at the
    top of the heatmap). When `group_by` is `None` the frame is returned in its
    original order with no group order. An explicit `group_order` (e.g. from a
    dendrogram) is used as-is; otherwise groups keep their first-appearance order.
    """
    if group_by is None:
        return frame, None
    if group_order is None:
        group_order = (
            frame.select(group_by).unique(maintain_order=True)[group_by].cast(pl.String).to_list()
        )
    rank = {group: index for index, group in enumerate(group_order)}
    ordered = (
        frame.with_columns(pl.col(group_by).cast(pl.String).replace_strict(rank).alias("_rank"))
        .sort("_rank", maintain_order=True)
        .drop("_rank")
    )
    return ordered, group_order


def _bin_observations(
    frame: pl.DataFrame, *, group_by: str | None, max_rows: int | None
) -> pl.DataFrame:
    """
    Collapse rows into at most ~`max_rows` virtual rows.

    Notes
    -----
    Bins are allocated per group proportionally to group size (at least one bin
    per group), mirroring the heatmap's `_bin_within_groups`, so group
    boundaries are preserved and the total stays near `max_rows`. Numeric columns
    are averaged within a bin; categorical columns take their most frequent value.
    Returns the frame unchanged when it already fits in `max_rows`.
    """
    n_rows = frame.height
    if max_rows is None or n_rows <= max_rows:
        return frame

    def _aggregations(columns: list[str]) -> list[pl.Expr]:
        return [
            (
                pl.col(column).mean()
                if frame.schema[column].is_numeric()
                else pl.col(column).mode().first()
            ).alias(column)
            for column in columns
        ]

    if group_by is None:
        return (
            frame.with_row_index("_rank")
            .with_columns((pl.col("_rank") * max_rows // n_rows).alias("_bin"))
            .group_by("_bin", maintain_order=True)
            .agg(_aggregations(frame.columns))
            .sort("_bin")
            .drop("_bin")
        )

    # Per-group bin allocation, matching heatmap's `_bin_within_groups` formula.
    group_meta = (
        frame.group_by(group_by, maintain_order=True)
        .agg(pl.len().alias("_n_cells"))
        .with_columns(
            pl.max_horizontal(
                pl.lit(1, dtype=pl.Int64),
                (pl.col("_n_cells") * max_rows / n_rows).round().cast(pl.Int64),
            ).alias("_bin_count")
        )
        .with_columns((pl.col("_bin_count").cum_sum() - pl.col("_bin_count")).alias("_bin_offset"))
    )
    value_columns = [column for column in frame.columns if column != group_by]
    return (
        frame.with_columns(pl.int_range(pl.len()).over(group_by).alias("_rank_in_group"))
        .join(group_meta, on=group_by)
        .with_columns(
            (
                (pl.col("_rank_in_group") * pl.col("_bin_count") // pl.col("_n_cells"))
                + pl.col("_bin_offset")
            ).alias("_bin")
        )
        .group_by("_bin", group_by, maintain_order=True)
        .agg(_aggregations(value_columns))
        .sort("_bin")
        .drop("_bin")
    )


def _assign_position_y(frame: pl.DataFrame, *, descending: bool) -> pl.DataFrame:
    """
    Attach `position_y`, the observation index of each row.

    Notes
    -----
    `descending` places the first row at the highest position (the top of an
    upright heatmap); otherwise the first row sits at position 0 (the left of a
    transposed heatmap).
    """
    n_rows = frame.height
    position = n_rows - 1 - pl.col("_row") if descending else pl.col("_row")
    return frame.with_row_index("_row").with_columns(position.alias("position_y")).drop("_row")


def _run_boundaries(labels: Sequence) -> list[int]:
    """Indices where `labels` changes value: each starts a new contiguous run."""
    return [index for index in range(1, len(labels)) if labels[index] != labels[index - 1]]


def _annotation_scale(
    series: pl.Series, colors: AnnotationColors | None, palette: str
) -> FeatureSpec | None:
    """
    Resolve the fill scale for one annotation track.

    Notes
    -----
    `colors` may be a category -> color mapping, a list of colors, or `None`.
    Numeric tracks default to a gray gradient; categorical tracks default to the
    track's assigned ColorBrewer `palette`.
    """
    numeric = series.dtype.is_numeric()
    if colors is None:
        if numeric:
            return scale_fill_gradient(low="#f0f0f0", high="#1a1a1a", name=series.name)
        return scale_fill_brewer(palette=palette, name=series.name)
    if isinstance(colors, Mapping):
        return scale_fill_manual(values=dict(colors), name=series.name)
    colors = list(colors)
    if numeric:
        if len(colors) == 2:
            return scale_fill_gradient(low=colors[0], high=colors[1], name=series.name)
        return scale_fill_gradientn(colors=colors, name=series.name)
    return scale_fill_manual(values=colors, name=series.name)


def _track_layers(
    frame: pl.DataFrame,
    annotations: Sequence[str],
    *,
    band: Literal["x", "y"],
    far: bool,
    colors: Mapping[str, AnnotationColors],
    palette_of: Mapping[str, str],
    titles: bool,
) -> list[FeatureSpec]:
    """
    Build one band's annotation tracks, each with its own fill scale.

    Notes
    -----
    `frame` holds the band's position column (shared with the heatmap axis) and
    the annotation columns. Each track is a level of the band's discrete cross
    axis, named after its column. The first listed is the outermost, so on a
    `far` band (right or top, where the cross axis grows away from the heatmap)
    the level order is reversed. Without `titles` the band axis has no breaks,
    so the track names take no room.
    """
    side_tile = geom_xsidetile if band == "x" else geom_ysidetile
    layers: list[FeatureSpec] = []
    for annotation in annotations:
        layers.append(new_scale_fill())
        layers.append(side_tile(aes(fill=annotation), data=frame))
        layers.append(
            _annotation_scale(frame[annotation], colors.get(annotation), palette_of[annotation])
        )
    if annotations:
        limits = list(reversed(annotations)) if far else list(annotations)
        # Track names on the right of an x band, clear of the heatmap's y axis
        # labels, and on top of a y band: turned names at the bottom would make the
        # bottom axis area as tall as the longest name. Hidden names keep the band
        # axis on the main axes' sides, where its line joins theirs.
        if band == "x":
            names = {"position": "right"} if titles else {"breaks": []}
            layers.append(scale_xsidey_discrete(limits=limits, expand=[0, 0], **names))
        else:
            names = {"position": "top"} if titles else {"breaks": []}
            layers.append(scale_ysidex_discrete(limits=limits, expand=[0, 0], **names))
    return layers


def _dendrogram_layer(
    paths: pl.DataFrame,
    *,
    band: Literal["x", "y"],
    group_centers: list[float],
    leaf_edge: float,
    extent: float,
    color: str,
    size: float,
    kwargs: dict,
) -> FeatureSpec:
    """
    Draw the group dendrogram as line segments in a side band.

    Notes
    -----
    Leaves sit at the group centers along the heatmap's observation axis, at
    cross position `leaf_edge` next to the tracks; the root lies `extent` cross
    units away (negative `extent` points toward the left or bottom). Each path
    is split into two-point segments, since a side line orders its points along
    the shared axis.
    """
    leaf_positions = np.arange(len(group_centers), dtype=float)
    leaf = np.interp(paths["x"].to_numpy(), leaf_positions, group_centers)
    height = leaf_edge + paths["y"].to_numpy() * extent
    path_group = paths["group"].to_numpy()
    starts = np.flatnonzero(path_group[:-1] == path_group[1:])
    segment = np.repeat(np.arange(starts.size), 2)
    index = np.column_stack([starts, starts + 1]).ravel()
    frame = pl.DataFrame({"leaf": leaf[index], "height": height[index], "segment": segment})
    if band == "x":
        mapping, side_line = aes(x="leaf", y="height", group="segment"), geom_xsideline
    else:
        mapping, side_line = aes(x="height", y="leaf", group="segment"), geom_ysideline
    return side_line(mapping, data=frame, inherit_aes=False, color=color, size=size, **kwargs)


def annotated_heatmap(
    data: AnnData | MuData,
    keys: Sequence[str],
    group_by: str | None = None,
    *,
    variable_group_by: str | None = None,
    modality: str | None = None,
    column_annotations: str | Sequence[str] | None = None,
    row_annotations: str | Sequence[str] | None = None,
    annotation_colors: Mapping[str, AnnotationColors] | None = None,
    mapping: FeatureSpec | None = None,
    geom: Literal["raster", "tile"] = "raster",
    layers: FeatureSpec | Sequence[FeatureSpec] | None = None,
    layers_all: FeatureSpec | Sequence[FeatureSpec] | None = None,
    scale_axis: Literal[0, 1] | None = None,
    transpose: bool = False,
    dendrogram: bool = False,
    group_lines: bool = True,
    group_lines_color: str = "white",
    group_lines_size: float = 0.6,
    group_lines_kwargs: dict | None = None,
    dendrogram_color: str = "black",
    dendrogram_size: float = 0.5,
    dendrogram_key: str | None = None,
    dendrogram_kwargs: dict | None = None,
    column_annotation_height: float = 0.025,
    row_annotation_width: float = 0.025,
    column_annotation_position: Literal["top", "bottom"] = "top",
    row_annotation_position: Literal["left", "right"] = "left",
    track_titles: bool = False,
    color_low: str = "#0000ff",
    color_mid: str = "#ffffff",
    color_high: str = "#ff0000",
    midpoint: Literal["mean", "median", "mid"] | float = "mid",
    max_rows: int | None = 1000,
    value_column: str = "value",
    variable_column: str = "variable",
    variable_id_column: str = "Variable",
    interactive: bool = False,
    legend: bool = True,
    **geom_kwargs,
) -> PlotSpec:
    """
    Annotated heatmap.

    Parameters
    ----------
    data : AnnData | MuData
        The single-cell data object.
    keys : Sequence[str]
        Variable keys to include, placed on the x-axis.
    variable_group_by : str | None, default=None
        Variable metadata key whose contiguous blocks (over `keys` in order) are
        separated by group lines, like `group_by` for observations. Drawn with
        the `group_lines_*` styling; needs `group_lines`.
    modality : str | None, default=None
        Which modality's stored analysis results to use when `markers` or
        `dendrogram` is enabled. Required for a multimodal object holding more
        than one modality, and not accepted otherwise.
    group_by : str | None, default=None
        The observation key to group the rows by. When `None`, rows keep their
        original order and no group separators are drawn.
    column_annotations : str | Sequence[str] | None, default=None
        Variable metadata key(s) to draw as tracks above the heatmap.
    row_annotations : str | Sequence[str] | None, default=None
        Observation metadata key(s) to draw as tracks to the left of the heatmap.
    annotation_colors : Mapping[str, Mapping[str, str] | Sequence[str]] | None, default=None
        Per-track color overrides, keyed by annotation name. The value is either a
        category -> color mapping, or a list of colors (two colors define a
        gradient for numeric tracks). Tracks without an entry use the defaults.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the heatmap, the result of `aes()`.
    geom : {'raster', 'tile'}, default='raster'
        The geom to use for the heatmap cells. Use 'raster' for performance.
        Use 'tile' to enable tooltips.
    layers : FeatureSpec | Sequence[FeatureSpec] | None, default=None
        Extra lets-plot layers to add to the heatmap, such as a scale or theme; a
        single spec or a sequence of them. A `scale_fill_*` (e.g.
        `scale_fill_viridis()`) overrides the default value gradient.
    layers_all : FeatureSpec | Sequence[FeatureSpec] | None, default=None
        Extra lets-plot layers added after `layers`. The heatmap and its tracks
        are one plot, so this now acts as `layers`; kept for compatibility.
    scale_axis : {0, 1} | None, default=None
        Standardize values with min-max scaling. If 0, standardize each
        variable (column); if 1, standardize each observation (row).
    transpose : bool, default=False
        Swap the axes so observations run along the x-axis and the variable
        `keys` along the y-axis. Annotation tracks and the dendrogram move with
        their data: variable tracks to the left, observation tracks and the
        dendrogram above the heatmap.
    dendrogram : bool, default=False
        Whether to cluster the groups and draw a dendrogram beside the
        observation tracks, outermost in their side band.
        The clustering is computed if not already available. Requires `group_by`.
    group_lines : bool, default=True
        Whether to draw separator lines between `group_by` groups (and between
        `variable_group_by` blocks).
    group_lines_color : str, default='white'
        Color of the group separator lines.
    group_lines_size : float, default=0.6
        Size (thickness) of the group separator lines.
    group_lines_kwargs : dict | None, default=None
        Additional parameters for the group separator `geom_segment`.
    dendrogram_color : str, default='black'
        Color of the dendrogram segments.
    dendrogram_size : float, default=0.5
        Size (thickness) of the dendrogram segments.
    dendrogram_key : str | None, default=None
        Specific key holding the precomputed dendrogram.
    dendrogram_kwargs : dict | None, default=None
        Additional parameters for the dendrogram `geom_path`.
    column_annotation_height : float, default=0.025
        Height of each column annotation track, relative to the heatmap.
    row_annotation_width : float, default=0.025
        Width of each row annotation track, relative to the heatmap.
    column_annotation_position : {'top', 'bottom'}, default='top'
        Which side of the heatmap the column (variable) tracks sit on. Under
        `transpose` this reflects to the left/right side (top to left, bottom
        to right).
    row_annotation_position : {'left', 'right'}, default='left'
        Which side of the heatmap the row (observation) tracks and the
        dendrogram sit on. Under `transpose` this reflects to the top/bottom
        side (left to top, right to bottom).
    track_titles : bool, default=False
        Whether to name each annotation track beside its band. Hidden by
        default, since the legends already name the tracks; set to `True` to
        show them (a top or bottom band's names on its right, a left or right
        band's names above it).
    color_low : str, default='#0000ff'
        Color for low values in the heatmap gradient.
    color_mid : str, default='#ffffff'
        Color for mid values in the heatmap gradient.
    color_high : str, default='#ff0000'
        Color for high values in the heatmap gradient.
    midpoint : {'mean', 'median', 'mid'} | float, default='mid'
        Midpoint for the heatmap color gradient.
    max_rows : int | None, default=1000
        Cap the number of plotted rows by averaging contiguous observations
        (within each group) into virtual bins. Numeric annotations are averaged
        and categorical annotations take their most frequent value per bin. Set
        to `None` to plot every observation.
    value_column : str, default='value'
        Name for the value column after unpivoting.
    variable_column : str, default='variable'
        Name for the variable column after unpivoting.
    variable_id_column : str, default='Variable'
        Name for the variables identifier column.
    interactive : bool, default=False
        Whether to make the heatmap panel interactive (zoom and pan).
    legend : bool, default=True
        Whether to show the legends for the heatmap and the annotation tracks.
    **geom_kwargs
        Additional parameters for the heatmap geom layer.

    Returns
    -------
    PlotSpec
        The annotated heatmap, one plot with its tracks in side bands.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    ValueError
        If `dendrogram` is set without `group_by`.

    Notes
    -----
    Rows are observations and columns are the variable `keys`. Variable tracks
    sit above the heatmap and observation tracks to its left, in ggside side
    bands that share the heatmap's axes (and its zoom, with `interactive`).
    Each track has its own fill scale (`new_scale_fill`) and legend, and is
    labelled on its band's axis.

    When `dendrogram=True` and the requested dendrogram is absent from
    `data.uns`, plot construction calls `scanpy.tl.dendrogram` and stores the
    result in `data.uns`. This mutates the input `AnnData`. Precompute the
    dendrogram under `dendrogram_key` to avoid mutation during plotting.

    Examples
    --------
    .. jupyter-execute::

        import cellestial as cl

        data = cl.datasets.pbmc3k()
        markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]

        cl.annotated_heatmap(
            data,
            keys=markers,
            group_by="cell_type_lvl1",
            column_annotations=["highly_variable"],
            row_annotations=["leiden", "pct_counts_mt"],
            scale_axis=0,
        )
    """
    if not isinstance(data, (AnnData, MuData)):
        raise _unsupported_data_type(data, AnnData, MuData)

    keys = list(keys)
    column_annotations = _as_list(column_annotations)
    row_annotations = _as_list(row_annotations)
    annotation_colors = annotation_colors or {}

    if dendrogram and group_by is None:
        msg = "`group_by` is required when `dendrogram` is set."
        raise ValueError(msg)

    # CLUSTER: a dendrogram dictates the group order; otherwise first-appearance.
    if dendrogram:
        assert group_by is not None  # guaranteed by the guard above
        dendrogram_order, dendrogram_paths = _get_dendrogram(
            *_modality_source(data, modality, group_by), use_key=dendrogram_key
        )
    else:
        dendrogram_order, dendrogram_paths = None, None

    # BUILD: observation frame with expression and row metadata, order, then bin.
    metadata = [group_by, *row_annotations] if group_by is not None else list(row_annotations)
    observations = build_frame(
        data=data,
        variable_keys=keys,
        axis=0,
        observation_id_column=None,
        metadata_columns=list(dict.fromkeys(metadata)),
    )
    observations, group_order = _order_observations(
        observations, group_by=group_by, group_order=dendrogram_order
    )
    observations = _bin_observations(observations, group_by=group_by, max_rows=max_rows)
    # Transposed, the first observation sits at the left; upright, at the top.
    observations = _assign_position_y(observations, descending=not transpose)

    # BUILD: long-form heatmap frame on shared x / y positions. Transposed, the
    # first key sits at the top; upright, at the left.
    n_keys = len(keys)
    position_x = {
        key: (n_keys - 1 - index if transpose else index) for index, key in enumerate(keys)
    }
    frame = (
        observations.unpivot(
            on=keys,
            index="position_y",
            variable_name=variable_column,
            value_name=value_column,
        )
        .with_columns(pl.col(variable_column).replace_strict(position_x).alias("position_x"))
        .drop_nulls(value_column)
    )
    if scale_axis is not None:
        partition_key = variable_column if scale_axis == 0 else "position_y"
        frame = _scale_values(frame, value_column=value_column, partition_key=partition_key)

    n_observations = observations.height
    key_limits = (-0.5, n_keys - 0.5)
    observation_limits = (-0.5, n_observations - 0.5)
    mapping = mapping or aes()

    if "tooltips" in geom_kwargs and geom == "raster":
        _warn(
            "\nWarning: tooltips are not supported for 'raster' geom and will be ignored."
            "\nUse 'tile' geom to enable tooltips."
        )
        geom_kwargs.pop("tooltips")

    # Axis roles. `position_x` holds the variable-key index, `position_y` the
    # observation index; `transpose` swaps which is drawn on the visual x-axis.
    # The variable axis carries the tick labels; the observation axis has no breaks
    # (a theme blank would also hide the side band's track names on that axis).
    if transpose:
        heatmap_x, heatmap_y = "position_y", "position_x"
        key_scale = scale_y_continuous(
            breaks=[position_x[key] for key in keys],
            labels=keys,
            limits=list(key_limits),
            expand=[0, 0],
        )
        observation_scale = scale_x_continuous(
            breaks=[], limits=list(observation_limits), expand=[0, 0]
        )
    else:
        heatmap_x, heatmap_y = "position_x", "position_y"
        key_scale = scale_x_continuous(
            breaks=[position_x[key] for key in keys],
            labels=keys,
            limits=list(key_limits),
            expand=[0, 0],
        )
        observation_scale = scale_y_continuous(
            breaks=[], limits=list(observation_limits), expand=[0, 0]
        )

    # MAIN heatmap. `raster` renders the cells as one image (fast for many cells);
    # `tile` draws individual cells and supports tooltips. The plot maps only the
    # positions, which the side tracks inherit along the shared axis.
    aes_main = aes(fill=value_column, **mapping.as_dict())
    geom_layer = (
        geom_raster(aes_main, **geom_kwargs)
        if geom == "raster"
        else geom_tile(aes_main, **geom_kwargs)
    )
    plot = (
        ggplot(frame, aes(heatmap_x, heatmap_y))
        + geom_layer
        + _fill_gradient(
            frame[value_column],
            color_low=color_low,
            color_mid=color_mid,
            color_high=color_high,
            midpoint=midpoint,
        )
    )
    # Extra user layers, a single spec or a sequence, added before the tracks' new
    # fill scales so a `scale_fill_*` here overrides the default value gradient.
    plot = _apply_layers(plot, _as_layers(layers))
    plot = _apply_layers(plot, _as_layers(layers_all))

    # GROUP separator lines between contiguous blocks, spanning the variable axis.
    line_style = {
        "inherit_aes": False,
        "color": group_lines_color,
        "size": group_lines_size,
        **(group_lines_kwargs or {}),
    }
    if group_lines and group_order is not None and len(group_order) > 1:
        assert group_by is not None
        starts = _run_boundaries(observations[group_by].to_list())
        boundaries = [
            start - 0.5 if transpose else n_observations - start - 0.5 for start in starts
        ]
        if transpose:
            lines = pl.DataFrame({"x": boundaries})
            plot += geom_segment(
                data=lines,
                mapping=aes(x="x", xend="x"),
                y=-0.5,
                yend=n_keys - 0.5,
                **line_style,
            )
        else:
            lines = pl.DataFrame({"y": boundaries})
            plot += geom_segment(
                data=lines,
                mapping=aes(y="y", yend="y"),
                x=-0.5,
                xend=n_keys - 0.5,
                **line_style,
            )

    # VARIABLE group separator lines between contiguous `variable_group_by` blocks,
    # spanning the observation axis.
    if group_lines and variable_group_by is not None:
        variable_groups = build_frame(
            data=data,
            axis=1,
            variable_id_column=variable_id_column,
            metadata_columns=[variable_group_by],
        )
        group_of = dict(
            zip(
                variable_groups[variable_id_column],
                variable_groups[variable_group_by].cast(pl.String),
                strict=True,
            )
        )
        starts = _run_boundaries([group_of[key] for key in keys])
        boundaries = [n_keys - start - 0.5 if transpose else start - 0.5 for start in starts]
        if starts and transpose:
            plot += geom_segment(
                data=pl.DataFrame({"y": boundaries}),
                mapping=aes(y="y", yend="y"),
                x=-0.5,
                xend=n_observations - 0.5,
                **line_style,
            )
        elif starts:
            plot += geom_segment(
                data=pl.DataFrame({"x": boundaries}),
                mapping=aes(x="x", xend="x"),
                y=-0.5,
                yend=n_observations - 0.5,
                **line_style,
            )

    # Resolve which side each annotation group sits on. `transpose` reflects the
    # requested position across the main diagonal (left<->top, right<->bottom),
    # since it swaps which axis is horizontal. A top/bottom side is the x side
    # band (sharing the x axis); a left/right side is the y side band.
    reflect: dict[str, Literal["left", "right", "top", "bottom"]] = {
        "left": "top",
        "right": "bottom",
        "top": "left",
        "bottom": "right",
    }
    observation_side = reflect[row_annotation_position] if transpose else row_annotation_position
    variable_side = (
        reflect[column_annotation_position] if transpose else column_annotation_position
    )

    # Assign each annotation track a distinct ColorBrewer palette (cycled).
    palette_of = {
        name: _BREWER_PALETTES[index % len(_BREWER_PALETTES)]
        for index, name in enumerate([*column_annotations, *row_annotations])
    }

    # COLUMN annotation tracks (variable metadata), in the band along the variable axis.
    variable_band: Literal["x", "y"] = "x" if variable_side in ("top", "bottom") else "y"
    variable_layers: list[FeatureSpec] = []
    if column_annotations:
        variables = build_frame(
            data=data,
            axis=1,
            variable_id_column=variable_id_column,
            metadata_columns=column_annotations,
        ).filter(pl.col(variable_id_column).is_in(keys))
        variables = variables.with_columns(
            pl.col(variable_id_column).replace_strict(position_x).alias("position_x")
        ).select("position_x", *column_annotations)
        variable_layers = _track_layers(
            variables,
            column_annotations,
            band=variable_band,
            far=variable_side in ("right", "top"),
            colors=annotation_colors,
            palette_of=palette_of,
            titles=track_titles,
        )

    # ROW annotation tracks (observation metadata), in the band along the observation axis.
    observation_band: Literal["x", "y"] = "y" if variable_band == "x" else "x"
    observation_far = observation_side in ("right", "top")
    observation_layers = _track_layers(
        observations.select("position_y", *row_annotations),
        row_annotations,
        band=observation_band,
        far=observation_far,
        colors=annotation_colors,
        palette_of=palette_of,
        titles=track_titles,
    )

    # DENDROGRAM (group clustering), outermost in the observation band. Groups span
    # many rows, so an axis dendrogram (one leaf per discrete level) cannot be used.
    observation_thickness = row_annotation_width * len(row_annotations)
    if dendrogram:
        assert group_by is not None
        assert dendrogram_paths is not None
        assert group_order is not None
        centers = observations.group_by(group_by, maintain_order=True).agg(
            pl.col("position_y").mean().alias("_center")
        )
        center_of = dict(zip(centers[group_by].cast(pl.String), centers["_center"], strict=True))
        # The tree spans `_DENDROGRAM_WIDTH` of the heatmap; cross units are tracks.
        extent = _DENDROGRAM_WIDTH / row_annotation_width
        leaf_edge = len(row_annotations) - 0.5 if observation_far else -0.5
        observation_layers.append(
            _dendrogram_layer(
                dendrogram_paths,
                band=observation_band,
                group_centers=[center_of[group] for group in group_order],
                leaf_edge=leaf_edge,
                extent=extent if observation_far else -extent,
                color=dendrogram_color,
                size=dendrogram_size,
                kwargs=dendrogram_kwargs or {},
            )
        )
        observation_thickness += _DENDROGRAM_WIDTH

    # COMPOSE: both bands in one plot, each sized by its tracks (and the tree).
    thickness = {
        variable_band: column_annotation_height * len(column_annotations),
        observation_band: observation_thickness,
    }
    plot = _apply_layers(plot, variable_layers + observation_layers)
    plot += (
        key_scale
        + observation_scale
        # `geom_raster` otherwise locks to square pixels (collapsing a few-column,
        # many-row heatmap); `coord_cartesian` frees the aspect.
        + coord_cartesian()
        + ggside(
            x_pos=variable_side if variable_band == "x" else observation_side,
            y_pos=variable_side if variable_band == "y" else observation_side,
        )
        + _THEME_HEATMAP
        + theme(
            ggside_panel_scale_x=thickness["x"] or None,
            ggside_panel_scale_y=thickness["y"] or None,
            legend_position="bottom" if legend else "none",
            legend_box="horizontal",
            legend_direction="horizontal",
        )
        + _LEGEND_THEME
    )
    if interactive:
        plot += ggtb(size_zoomin=-1)
    return plot
