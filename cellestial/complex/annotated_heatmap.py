from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    coord_cartesian,
    element_blank,
    element_text,
    geom_path,
    geom_raster,
    geom_segment,
    geom_tile,
    gggrid,
    ggplot,
    ggtb,
    scale_fill_brewer,
    scale_fill_gradient,
    scale_fill_gradientn,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)

from cellestial.frames import build_frame
from cellestial.single.heatmap.utilities import _scale_values
from cellestial.themes import _THEME_HEATMAP
from cellestial.util import _fill_gradient, _get_dendrogram
from cellestial.util.errors import _unsupported_data_type

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import FeatureSpec, PlotSpec
    from lets_plot.plot.subplots import SupPlotsSpec

    AnnotationColors = Mapping[str, str] | Sequence[str]

# Width of the dendrogram panel, relative to the heatmap.
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

# Legend element sizing. Must live on the subplots that own each legend: the grid
# theme only controls legend placement/arrangement, not the collected element sizes.
# `title` carries the bold: lets-plot resolves a legend title's `face` from the
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
        return list(layers)
    return [layers]


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


def _assign_position_y(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach `position_y`, placing the first row at the top of the heatmap."""
    n_rows = frame.height
    return (
        frame.with_row_index("_row")
        .with_columns((n_rows - 1 - pl.col("_row")).alias("position_y"))
        .drop("_row")
    )


def _annotation_scale(
    series: pl.Series, colors: AnnotationColors | None, palette: str
) -> FeatureSpec | None:
    """
    Resolve the fill scale for one annotation track.

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


def _blank_strip_theme(*, legend: bool) -> FeatureSpec:
    """Void theme shared by annotation tracks, labels, and the dendrogram panel."""
    return (
        _THEME_HEATMAP
        + theme(
            axis_title=element_blank(),
            axis_text=element_blank(),
            axis_ticks=element_blank(),
            axis_line=element_blank(),
            legend_position="bottom" if legend else "none",
        )
        + _LEGEND_THEME
    )


def _annotation_strip(
    frame: pl.DataFrame,
    *,
    position: str,
    value: str,
    horizontal: bool,
    legend: bool,
    colors: AnnotationColors | None,
    palette: str,
    position_limits: tuple[float, float],
) -> PlotSpec:
    """
    Build a single one-row (or one-column) annotation track.

    `horizontal` strips vary along x (column annotations); vertical strips vary
    along y (row annotations). `position_limits` pin the varying axis so the
    track aligns with the heatmap panel. `palette` is the ColorBrewer palette used
    for a categorical track when `colors` is not given.
    """
    constant = "_strip"
    frame = frame.with_columns(pl.lit(0).alias(constant))
    mapping = (
        aes(x=position, y=constant, fill=value)
        if horizontal
        else aes(x=constant, y=position, fill=value)
    )
    plot = ggplot(frame) + geom_tile(mapping)
    scale = _annotation_scale(frame[value], colors, palette)
    if scale is not None:
        plot += scale
    if horizontal:
        plot += scale_x_continuous(limits=list(position_limits), expand=[0, 0])
        plot += scale_y_continuous(expand=[0, 0])
    else:
        plot += scale_x_continuous(expand=[0, 0])
        plot += scale_y_continuous(limits=list(position_limits), expand=[0, 0])
    plot += _blank_strip_theme(legend=legend)
    return plot


def _dendrogram_panel(
    paths: pl.DataFrame,
    *,
    group_centers: list[float],
    n_groups: int,
    position_limits: tuple[float, float],
    color: str,
    size: float,
    side: Literal["left", "right", "top", "bottom"],
    kwargs: dict,
) -> PlotSpec:
    """
    Build the group dendrogram as a panel on one `side` of the heatmap.

    Leaves sit at the group centers (height 0, next to the tracks); the root
    extends away from the heatmap toward `side`. `position_limits` match the
    heatmap observation axis so leaves line up with the group blocks. On a
    left/right side the height runs along x and leaves along y; on a top/bottom
    side the height runs along y and leaves along x.
    """
    leaf_positions = np.arange(n_groups, dtype=float)
    leaf = np.interp(paths["x"].to_numpy(), leaf_positions, group_centers)
    height = paths["y"].to_numpy()
    # Root extends away from the heatmap: negative toward left/bottom, positive
    # toward right/top. `expand` keeps a small margin so the root is not clipped.
    if side in ("left", "right"):
        signed_height = -height if side == "left" else height
        frame = pl.DataFrame({"x": signed_height, "y": leaf, "group": paths["group"]})
        height_scale = scale_x_continuous(expand=[0.05, 0])
        leaf_scale = scale_y_continuous(limits=list(position_limits), expand=[0, 0])
    else:
        signed_height = height if side == "top" else -height
        frame = pl.DataFrame({"x": leaf, "y": signed_height, "group": paths["group"]})
        height_scale = scale_y_continuous(expand=[0.05, 0])
        leaf_scale = scale_x_continuous(limits=list(position_limits), expand=[0, 0])
    return (
        ggplot(frame)
        + geom_path(aes("x", "y", group="group"), color=color, size=size, **kwargs)
        + height_scale
        + leaf_scale
        + _blank_strip_theme(legend=False)
    )


def annotated_heatmap(
    data: AnnData,
    keys: Sequence[str],
    group_by: str | None = None,
    *,
    column_annotations: str | Sequence[str] | None = None,
    row_annotations: str | Sequence[str] | None = None,
    annotation_colors: Mapping[str, AnnotationColors] | None = None,
    mapping: FeatureSpec | None = None,
    layers: FeatureSpec | Sequence[FeatureSpec] | None = None,
    layers_all: FeatureSpec | Sequence[FeatureSpec] | None = None,
    scale_axis: Literal[0, 1] | None = None,
    transpose: bool = False,
    dendrogram: bool = False,
    group_lines: bool = True,
    group_lines_color: str = "black",
    group_lines_size: float = 0.5,
    group_lines_kwargs: dict | None = None,
    dendrogram_color: str = "black",
    dendrogram_size: float = 0.5,
    dendrogram_key: str | None = None,
    dendrogram_kwargs: dict | None = None,
    column_annotation_height: float = 0.025,
    row_annotation_width: float = 0.025,
    column_annotation_position: Literal["top", "bottom"] = "top",
    row_annotation_position: Literal["left", "right"] = "left",
    color_low: str = "#0000ff",
    color_mid: str = "#ffffff",
    color_high: str = "#ff0000",
    mid_point: Literal["mean", "median", "mid"] | float = "mid",
    max_rows: int | None = 1000,
    value_column: str = "value",
    variable_column: str = "variable",
    variables_name: str = "Variable",
    interactive: bool = False,
    legend: bool = True,
    **geom_kwargs,
) -> SupPlotsSpec:
    """
    Annotated heatmap with metadata tracks along both axes.

    Rows are observations, columns are the variable `keys`. Variable-metadata
    tracks are drawn above the heatmap (over the columns) and observation-metadata
    tracks to its left (over the rows). The result is a composite assembled with
    `gggrid`; the panels are aligned and every track keeps its own legend.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    keys : Sequence[str]
        Variable keys to include, placed on the x-axis.
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
    layers : FeatureSpec | Sequence[FeatureSpec] | None, default=None
        Extra lets-plot layers to add to the heatmap, such as a scale or theme; a
        single spec or a sequence of them. A `scale_fill_*` (e.g.
        `scale_fill_viridis()`) overrides the default value gradient.
    layers_all : FeatureSpec | Sequence[FeatureSpec] | None, default=None
        Extra lets-plot layers to add to every subplot (the heatmap, each
        annotation track, and the dendrogram panel), unlike `layers` which only
        affects the heatmap; a single spec or a sequence of them.
    scale_axis : {0, 1} | None, default=None
        Standardize values with min-max scaling. If 0, standardize each
        variable (column); if 1, standardize each observation (row).
    transpose : bool, default=False
        Swap the axes so observations run along the x-axis and the variable
        `keys` along the y-axis. Annotation tracks and the dendrogram move with
        their data: variable tracks to the left, observation tracks and the
        dendrogram above the heatmap.
    dendrogram : bool, default=False
        Whether to cluster the groups and draw a dendrogram panel to the left.
        The clustering is computed if not already available. Requires `group_by`.
    group_lines : bool, default=True
        Whether to draw horizontal separator lines between groups.
    group_lines_color : str, default='black'
        Color of the group separator lines.
    group_lines_size : float, default=0.5
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
    color_low : str, default='#0000ff'
        Color for low values in the heatmap gradient.
    color_mid : str, default='#ffffff'
        Color for mid values in the heatmap gradient.
    color_high : str, default='#ff0000'
        Color for high values in the heatmap gradient.
    mid_point : {'mean', 'median', 'mid'} | float, default='mid'
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
    variables_name : str, default='Variable'
        Name for the variables identifier column.
    interactive : bool, default=False
        Whether to make the heatmap panel interactive (zoom and pan).
    legend : bool, default=True
        Whether to show the legends for the heatmap and the annotation tracks.
    **geom_kwargs
        Additional parameters for the heatmap geom layer.

    Returns
    -------
    SupPlotsSpec
        The composite annotated heatmap.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    ValueError
        If `dendrogram` is set without `group_by`.

    Examples
    --------
    .. jupyter-execute::

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")
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
    if not isinstance(data, AnnData):
        raise _unsupported_data_type(data, AnnData)

    keys = list(keys)
    column_annotations = _as_list(column_annotations)
    row_annotations = _as_list(row_annotations)
    annotation_colors = annotation_colors or {}
    shared_layers = _as_layers(layers_all)

    if dendrogram and group_by is None:
        msg = "`group_by` is required when `dendrogram` is set."
        raise ValueError(msg)

    # CLUSTER: a dendrogram dictates the group order; otherwise first-appearance.
    dendrogram_order, dendrogram_paths = (
        _get_dendrogram(data, group_by, use_key=dendrogram_key) if dendrogram else (None, None)
    )

    # BUILD: observation frame with expression and row metadata, order, then bin.
    metadata = [group_by, *row_annotations] if group_by is not None else list(row_annotations)
    observations = build_frame(
        data=data,
        variable_keys=keys,
        axis=0,
        observations_name=None,
        metadata_columns=list(dict.fromkeys(metadata)),
    )
    observations, group_order = _order_observations(
        observations, group_by=group_by, group_order=dendrogram_order
    )
    observations = _bin_observations(observations, group_by=group_by, max_rows=max_rows)
    observations = _assign_position_y(observations)

    # BUILD: long-form heatmap frame on shared x / y positions.
    position_x = {key: index for index, key in enumerate(keys)}
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

    n_keys = len(keys)
    n_obs = observations.height
    key_limits = (-0.5, n_keys - 0.5)
    obs_limits = (-0.5, n_obs - 0.5)
    mapping = mapping or aes()

    # Axis roles. `position_x` holds the variable-key index, `position_y` the
    # observation index; `transpose` swaps which is drawn on the visual x-axis.
    # The variable axis carries the tick labels; the observation axis is blanked.
    if transpose:
        heatmap_x, heatmap_y = "position_y", "position_x"
        key_scale = scale_y_continuous(
            breaks=list(range(n_keys)), labels=keys, limits=list(key_limits), expand=[0, 0]
        )
        observation_scale = scale_x_continuous(limits=list(obs_limits), expand=[0, 0])
        blank_observation_axis = theme(axis_text_x=element_blank(), axis_ticks_x=element_blank())
    else:
        heatmap_x, heatmap_y = "position_x", "position_y"
        key_scale = scale_x_continuous(
            breaks=list(range(n_keys)), labels=keys, limits=list(key_limits), expand=[0, 0]
        )
        observation_scale = scale_y_continuous(limits=list(obs_limits), expand=[0, 0])
        blank_observation_axis = theme(axis_text_y=element_blank(), axis_ticks_y=element_blank())

    # MAIN heatmap. `geom_raster` renders the cells as one image (fast for many cells);
    # aspect stays controllable through the grid widths/heights.
    htmp = (
        ggplot(frame)
        + geom_raster(
            aes(heatmap_x, heatmap_y, fill=value_column, **mapping.as_dict()), **geom_kwargs
        )
        + _fill_gradient(
            frame[value_column],
            color_low=color_low,
            color_mid=color_mid,
            color_high=color_high,
            mid_point=mid_point,
        )
        + key_scale
        + observation_scale
        # `geom_raster` otherwise locks to square pixels (collapsing a few-column,
        # many-row heatmap); `coord_cartesian` frees the aspect for the grid to set.
        + coord_cartesian()
        + _THEME_HEATMAP
        + blank_observation_axis
        + theme(legend_position="bottom" if legend else "none")
        + _LEGEND_THEME
    )
    if interactive:
        htmp += ggtb()
    # Extra user layers for the heatmap, a single spec or a sequence. A `scale_fill_*`
    # here overrides the default value gradient (lets-plot keeps the last scale).
    htmp = _apply_layers(htmp, _as_layers(layers))
    htmp = _apply_layers(htmp, shared_layers)

    # GROUP separator lines between contiguous blocks, spanning the variable axis.
    if group_lines and group_order is not None and len(group_order) > 1:
        sizes = observations.group_by(group_by, maintain_order=True).len()["len"].to_list()
        boundaries = []
        running = 0
        for size in sizes[:-1]:
            running += size
            boundaries.append(n_obs - running - 0.5)
        if transpose:
            lines = pl.DataFrame({"x": boundaries})
            htmp += geom_segment(
                data=lines,
                mapping=aes(x="x", xend="x"),
                y=-0.5,
                yend=n_keys - 0.5,
                color=group_lines_color,
                size=group_lines_size,
                **(group_lines_kwargs or {}),
            )
        else:
            lines = pl.DataFrame({"y": boundaries})
            htmp += geom_segment(
                data=lines,
                mapping=aes(y="y", yend="y"),
                x=-0.5,
                xend=n_keys - 0.5,
                color=group_lines_color,
                size=group_lines_size,
                **(group_lines_kwargs or {}),
            )

    # Resolve which side each annotation group sits on. `transpose` reflects the
    # requested position across the main diagonal (left<->top, right<->bottom),
    # since it swaps which axis is horizontal. A top/bottom side means the track
    # is a horizontal strip; a left/right side means a vertical strip.
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

    # COLUMN annotation tracks (variable metadata), drawn along the variable axis.
    column_strips: list[PlotSpec] = []
    if column_annotations:
        variables = build_frame(
            data=data,
            axis=1,
            variables_name=variables_name,
            metadata_columns=column_annotations,
        ).filter(pl.col(variables_name).is_in(keys))
        variables = variables.with_columns(
            pl.col(variables_name).replace_strict(position_x).alias("position_x")
        )
        for annotation in column_annotations:
            strip = _annotation_strip(
                variables.select("position_x", annotation),
                position="position_x",
                value=annotation,
                horizontal=variable_side in ("top", "bottom"),
                legend=legend,
                colors=annotation_colors.get(annotation),
                palette=palette_of[annotation],
                position_limits=key_limits,
            )
            column_strips.append(_apply_layers(strip, shared_layers))

    # ROW annotation tracks (observation metadata), drawn along the observation axis.
    row_strips: list[PlotSpec] = []
    for annotation in row_annotations:
        strip = _annotation_strip(
            observations.select("position_y", annotation),
            position="position_y",
            value=annotation,
            horizontal=observation_side in ("top", "bottom"),
            legend=legend,
            colors=annotation_colors.get(annotation),
            palette=palette_of[annotation],
            position_limits=obs_limits,
        )
        row_strips.append(_apply_layers(strip, shared_layers))

    # DENDROGRAM panel (group clustering), drawn alongside the observation axis.
    dendrogram_panel = None
    if dendrogram:
        centers = observations.group_by(group_by, maintain_order=True).agg(
            pl.col("position_y").mean().alias("_center")
        )
        center_of = dict(zip(centers[group_by].cast(pl.String), centers["_center"], strict=True))
        group_centers = [center_of[group] for group in group_order]
        dendrogram_panel = _apply_layers(
            _dendrogram_panel(
                dendrogram_paths,
                group_centers=group_centers,
                n_groups=len(group_order),
                position_limits=obs_limits,
                color=dendrogram_color,
                size=dendrogram_size,
                side=observation_side,
                kwargs=dendrogram_kwargs or {},
            ),
            shared_layers,
        )

    # COMPOSE: the observation group (dendrogram outermost, then observation
    # tracks) and the variable group (variable tracks). One group flanks the
    # heatmap horizontally (left/right side), the other vertically (top/bottom),
    # since their axes are perpendicular. First listed track is the outermost;
    # each keeps its per-track thickness.
    dendrogram_thickness = [_DENDROGRAM_WIDTH] if dendrogram_panel is not None else []
    observation_panels = ([dendrogram_panel] if dendrogram_panel is not None else []) + row_strips
    observation_thickness = dendrogram_thickness + [row_annotation_width] * len(row_strips)
    variable_thickness = [column_annotation_height] * len(column_strips)
    if observation_side in ("left", "right"):
        flank_panels, flank_thickness, flank_side = (
            observation_panels,
            observation_thickness,
            observation_side,
        )
        stack_panels, stack_thickness, stack_side = (
            column_strips,
            variable_thickness,
            variable_side,
        )
    else:
        flank_panels, flank_thickness, flank_side = (
            column_strips,
            variable_thickness,
            variable_side,
        )
        stack_panels, stack_thickness, stack_side = (
            observation_panels,
            observation_thickness,
            observation_side,
        )

    # Heatmap row: flank tracks sit to the left or right of the heatmap; on the
    # right the first-listed (outermost) track ends up farthest out, so reverse.
    if flank_side == "left":
        row_cells, widths, heatmap_column = (
            [*flank_panels, htmp],
            [*flank_thickness, 1.0],
            len(flank_panels),
        )
    else:
        row_cells = [htmp, *reversed(flank_panels)]
        widths = [1.0, *reversed(flank_thickness)]
        heatmap_column = 0
    n_cols = len(row_cells)

    def _stack_row(panel: PlotSpec) -> list[PlotSpec | None]:
        """A grid row holding one stacked track in the heatmap column, else empty."""
        row: list[PlotSpec | None] = [None] * n_cols
        row[heatmap_column] = panel
        return row

    # Stack tracks sit above or below the heatmap; reverse for bottom so the
    # first-listed (outermost) track ends up farthest out.
    if stack_side == "top":
        top_panels, top_thickness, bottom_panels, bottom_thickness = (
            stack_panels,
            stack_thickness,
            [],
            [],
        )
    else:
        top_panels, top_thickness = [], []
        bottom_panels, bottom_thickness = (
            list(reversed(stack_panels)),
            list(reversed(stack_thickness)),
        )

    plots: list[PlotSpec | None] = []
    for panel in top_panels:
        plots.extend(_stack_row(panel))
    plots.extend(row_cells)
    for panel in bottom_panels:
        plots.extend(_stack_row(panel))

    heights = [*top_thickness, 1.0, *bottom_thickness]
    grid = gggrid(
        plots,
        ncol=n_cols,
        widths=widths,
        heights=heights,
        align=True,
        guides="collect" if legend else None,
    )
    if legend:
        # The grid only places/arranges the collected legends (one bottom row); their
        # element sizes come from the subplots' `_LEGEND_THEME`, not from here.
        grid += theme(
            legend_position="bottom",
            legend_box="horizontal",
            legend_direction="horizontal",
        )
    return grid
