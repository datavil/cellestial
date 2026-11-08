from __future__ import annotations

__lazy_modules__ = ["anndata", "ggrust", "mudata"]

from typing import TYPE_CHECKING, Literal

import polars as pl
from anndata import AnnData
from ggrust import (
    aes,
    element_blank,
    geom_path,
    geom_rect,
    geom_ribbon,
    geom_segment,
    geom_text,
    ggplot,
    ggtb,
    guides,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)
from mudata import MuData

from cellestial.frames import build_frame
from cellestial.single.heatmap.utilities import (
    _assign_positions,
    _bin_within_groups,
    _get_group_bar_frame,
    _get_group_lines_frame,
    _label_size_absolute,
    _resolve_key_groups,
    _resolve_rank_genes_groups_args,
)
from cellestial.themes import _THEME_HEATMAP
from cellestial.util import _get_dendrogram, _get_dendrogram_path_frame
from cellestial.util.errors import (
    DendrogramMismatchError,
    DuplicateKeysError,
    InvalidArgumentError,
    _unsupported_data_type,
)
from cellestial.util.utilities import _container_column, _modality_source

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ggrust.plot.core import FeatureSpec, LayerSpec, PlotSpec

# Track geometry in y data units: track `i` (top to bottom) spans
# [n_tracks - 1 - i, n_tracks - 1 - i + _TRACK_HEIGHT].
_TRACK_HEIGHT = 0.85
_GROUP_BAR_Y = -0.2
_DENDROGRAM_GAP = 0.1
# Key-group brackets on the right, as fractions of the number of cell columns.
_BRACKET_GAP_FRACTION = 0.01
_BRACKET_TIP_FRACTION = 0.008
_BRACKET_LABEL_GAP_FRACTION = 0.012
_BRACKET_LABEL_ROOM_FRACTION = 0.03
_DEFAULT_TRACK_FILL = "#3f3f3f"


def tracksplot(
    data: AnnData | MuData,
    keys: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    group_by: str | None = None,
    *,
    modality: str | None = None,
    markers: bool | str = False,
    n_genes: int = 5,
    groups: Sequence[str] | None = None,
    mapping: FeatureSpec | None = None,
    geom: Literal["bar", "area"] = "bar",
    dendrogram: bool = False,
    group_bars: bool = True,
    group_bars_size: float = 6,
    group_bars_labels: bool = True,
    group_lines: bool = True,
    group_lines_color: str = "#bfbfbf",
    group_lines_size: float = 0.4,
    dendrogram_color: str = "black",
    dendrogram_size: float = 0.5,
    dendrogram_key: str | None = None,
    group_lines_kwargs: dict | None = None,
    key_labels: bool = True,
    key_labels_text_size: float = 1.0,
    key_labels_bracket_size: float = 0.6,
    key_labels_text_color: str = "black",
    key_labels_bracket_color: str = "black",
    key_labels_width: float = 0.6,
    value_column: str = "value",
    variable_column: str = "variable",
    observation_id_column: str = "Barcode",
    interactive: bool = False,
    max_rows: int | None = 1000,
    **geom_kwargs,
) -> PlotSpec:
    """
    Tracks plot of per-cell values, one track per key, with cells ordered by group.

    Parameters
    ----------
    data : AnnData | MuData
        The single-cell data object.
    keys : Sequence[str] | Mapping[str, Sequence[str]] | None, default=None
        Variable keys to plot, one track each, top to bottom. A mapping assigns
        keys to group labels (no key in more than one group), drawn as brackets
        on the right. Must be `None` when `markers` is set.
    group_by : str | None, default=None
        The categorical key to order and group the cells by. Inferred from a
        precomputed ranking when `markers` is set.
    modality : str | None, default=None
        Which modality's stored analysis results to use when `markers` or
        `dendrogram` is enabled. Required for a multimodal object holding more
        than one modality, and not accepted otherwise.
    markers : bool | str, default=False
        Derive `keys` from a precomputed ranking. Pass `True` to use the
        default ranking key, or a string to read a custom key (e.g.
        `"rank_genes_groups_wilcoxon"`).
    n_genes : int, default=5
        Number of top genes to take per group when `markers` is set.
    groups : Sequence[str] | None, default=None
        Subset of groups to include when `markers` is set;
        `None` keeps all groups in their stored order.
    mapping : FeatureSpec | None, default=None
        Aesthetic mappings for the tracks, the result of `aes()`.
        For example, `aes(fill=group_by)` colors the tracks by group.
    geom : {'bar', 'area'}, default='bar'
        Draw each cell as a bar, or each track as a filled area.
    dendrogram : bool, default=False
        Whether to order the groups by a dendrogram and draw it above the tracks.
        Computed from the data if not already stored.
    group_bars : bool, default=True
        Whether to draw a colored bar below the tracks marking group spans.
    group_bars_size : float, default=6
        Size (thickness) of the group bar.
    group_bars_labels : bool, default=True
        Whether to label the group spans along the x-axis.
        Removes the related legend.
    group_lines : bool, default=True
        Whether to draw vertical lines separating the groups.
    group_lines_color : str, default='#bfbfbf'
        Color of the group separator lines.
    group_lines_size : float, default=0.4
        Size (thickness) of the group separator lines.
    dendrogram_color : str, default='black'
        Color of the dendrogram segments.
    dendrogram_size : float, default=0.5
        Size (thickness) of the dendrogram segments.
    dendrogram_key : str | None, default=None
        Specific key holding the precomputed dendrogram.
        By default, `dendrogram_{group_by}` is used.
    group_lines_kwargs : dict | None, default=None
        Additional parameters to pass to the group separator lines geom_segment.
    key_labels : bool, default=True
        Whether to draw bracket labels on the right when `keys` is a mapping.
    key_labels_text_size : float, default=1.0
        Scale multiplier on the auto-computed bracket label text size.
    key_labels_bracket_size : float, default=0.6
        Size (thickness) of the bracket lines.
    key_labels_text_color : str, default='black'
        Color of the bracket label text.
    key_labels_bracket_color : str, default='black'
        Color of the bracket lines.
    key_labels_width : float, default=0.6
        Bracket extent (in track units) for a singleton group. Multi-key groups
        extend `key_labels_width / 2` past the first and last track center.
    value_column : str, default='value'
        Name for the value column after unpivoting.
    variable_column : str, default='variable'
        Name for the variable column after unpivoting.
    observation_id_column : str, default='Barcode'
        Name for the cell identifier column.
    interactive : bool, default=False
        Whether to add the interactive toolbar.
    max_rows : int | None, default=1000
        Maximum number of cell columns. Cells within a group are binned into
        about this many columns in total (at least one per group), each showing
        the mean of its cells.
        `None` draws every cell.
    **geom_kwargs
        Additional parameters passed to the track geom (`geom_rect` for bars,
        `geom_ribbon` for areas).

    Returns
    -------
    PlotSpec
        The tracks plot.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not an AnnData or MuData object.
    InvalidArgumentError
        If `keys` or `group_by` is missing without `markers`, or `geom` is
        not one of the accepted values.
    KeyError
        If `group_by` is not found.
    VariableNotFoundError
        If a key is not found.

    Notes
    -----
    Every track has its own scale: values are divided by the largest absolute
    value of that key, so the tallest cell of each track reaches the full track
    height. The unscaled values stay in `value_column` for tooltips.

    Examples
    --------
    .. jupyter-execute::

        import cellestial as cl

        data = cl.datasets.pbmc3k()
        cl.tracksplot(data, keys=["CD3D", "MS4A1", "LYZ", "NKG7"], group_by="leiden")

    Color the tracks by group and draw them as areas.

    .. jupyter-execute::

        from ggrust import aes

        cl.tracksplot(
            data,
            keys=["CD3D", "MS4A1", "LYZ", "NKG7"],
            group_by="leiden",
            mapping=aes(fill="leiden"),
            geom="area",
        )

    """
    if not isinstance(data, (AnnData, MuData)):
        raise _unsupported_data_type(data, AnnData, MuData)
    if geom not in ("bar", "area"):
        msg = f"`geom` must be 'bar' or 'area', got {geom!r}."
        raise InvalidArgumentError(msg)

    if markers:
        results, local_group_by = _modality_source(data, modality, group_by)
        keys, group_by = _resolve_rank_genes_groups_args(
            results,
            rank_genes_groups=markers,
            n_genes=n_genes,
            groups=groups,
            keys=keys,
            group_by=local_group_by,
        )
        group_by = _container_column(data, modality, group_by)
    elif keys is None or group_by is None:
        msg = (
            "`keys` and `group_by` are required for a tracks plot "
            "(or enable `markers` to derive them from a precomputed ranking)."
        )
        raise InvalidArgumentError(msg)

    mapping = mapping or aes()
    keys, key_groups = _resolve_key_groups(keys, key_labels=key_labels)
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        msg = f"`keys` contains duplicates: {duplicates}. Pass each key once."
        raise DuplicateKeysError(msg)

    # BUILD: wide frame, one row per cell and one column per key
    frame = build_frame(
        data=data,
        variable_keys=keys,
        observation_id_column=observation_id_column,
        metadata_columns=[group_by],
    )
    value_dtype = frame.head(0).select(keys).unpivot(value_name=value_column).schema[value_column]
    frame = frame.with_columns(pl.col(keys).cast(value_dtype)).filter(
        pl.col(observation_id_column).is_not_null() & pl.col(group_by).is_not_null()
    )

    # ORDER: groups left to right
    if dendrogram:
        group_order, paths = _get_dendrogram(
            *_modality_source(data, modality, group_by), use_key=dendrogram_key
        )
        present = set(frame[group_by].cast(pl.String).unique().to_list())
        absent = [group for group in group_order if group not in present]
        if absent:
            msg = (
                f"The dendrogram for `group_by={group_by!r}` includes groups with no cells "
                f"in the data: {absent}. It was likely computed before the data was "
                "subset; recompute it on this data, or set `dendrogram=False`."
            )
            raise DendrogramMismatchError(msg)
    else:
        group_order = _group_order(frame[group_by])
        paths = None

    # BIN: collapse cells into at most `max_rows` columns (within group)
    if max_rows is not None:
        frame = _bin_within_groups(
            frame,
            observation_id_column=observation_id_column,
            group_by=group_by,
            keys=keys,
            y_order_groups=group_order,
            max_rows=max_rows,
        )

    # SCALE: each track to [0, 1] by its largest absolute value
    scaled_column = "__cellestial_track_value"
    frame = (
        frame.unpivot(
            on=keys,
            index=[observation_id_column, group_by],
            variable_name=variable_column,
            value_name=value_column,
        )
        .drop_nulls(value_column)
        .with_columns(
            (pl.col(value_column) / pl.col(value_column).abs().max().over(variable_column))
            .fill_nan(0.0)
            .alias(scaled_column)
        )
    )

    # ASSIGN: positions. The heatmap helper places keys on x and cells on y,
    # so its axes are swapped here: cells on x, keys (tracks) on y.
    frame, cell_frame, n_tracks, n_cells, group_centers = _assign_positions(
        frame,
        aggregate=False,
        group_by=group_by,
        observation_id_column=observation_id_column,
        variable_column=variable_column,
        x_keys=keys,
        y_order_groups=group_order,
    )
    assert cell_frame is not None
    frame = frame.with_columns(
        pl.col("position_y").alias("position_x"),
        (n_tracks - 1 - pl.col("position_x")).alias("track_base"),
    ).with_columns(
        (pl.col("track_base") + pl.col(scaled_column).clip(lower_bound=0) * _TRACK_HEIGHT).alias(
            "track_top"
        ),
        (pl.col("track_base") + pl.col(scaled_column).clip(upper_bound=0) * _TRACK_HEIGHT).alias(
            "track_bottom"
        ),
        # One area per track and group, so a group-mapped fill varies along the track.
        pl.concat_str(
            pl.col(variable_column), pl.col(group_by).cast(pl.String), separator="\x1f"
        ).alias("track_group"),
        (pl.col("position_x") - 0.5).alias("position_xmin"),
        (pl.col("position_x") + 0.5).alias("position_xmax"),
    )

    # BUILD: track layer
    if "fill" not in mapping.as_dict() and "fill" not in geom_kwargs:
        geom_kwargs["fill"] = _DEFAULT_TRACK_FILL
    # Outlines would cover the fill of one-pixel-wide cell columns.
    geom_kwargs.setdefault("size", 0)
    if geom == "bar":
        track_layer = geom_rect(
            aes(
                xmin="position_xmin",
                xmax="position_xmax",
                ymin="track_bottom",
                ymax="track_top",
                **mapping.as_dict(),
            ),
            **geom_kwargs,
        )
    else:
        track_layer = geom_ribbon(
            aes(
                x="position_x",
                ymin="track_bottom",
                ymax="track_top",
                group="track_group",
                **mapping.as_dict(),
            ),
            **geom_kwargs,
        )
    plot = ggplot(frame.sort("position_x")) + track_layer + _THEME_HEATMAP

    # Y scale: one break per track, labelled with its key
    x_min_limit = -0.5
    x_max_limit = n_cells - 0.5
    y_min_limit = _GROUP_BAR_Y - 0.15 if group_bars else -0.05
    y_max_limit = n_tracks - 1 + _TRACK_HEIGHT + 0.05

    # GROUP separator lines (vertical, between groups)
    if group_lines and len(group_order) > 1:
        lines_frame = _get_group_lines_frame(
            cell_frame,
            aggregate=False,
            group_by=group_by,
            n_x=n_tracks,
            n_groups=len(group_order),
        )
        plot += geom_segment(
            data=lines_frame.select(
                pl.col("y").alias("x"),
                pl.col("y").alias("xend"),
                pl.lit(y_min_limit).alias("y"),
                pl.lit(n_tracks - 1 + _TRACK_HEIGHT).alias("yend"),
            ),
            mapping=aes(x="x", xend="xend", y="y", yend="yend"),
            color=group_lines_color,
            size=group_lines_size,
            **(group_lines_kwargs or {}),
        )

    # GROUP color bar below the tracks
    if group_bars:
        group_bar_frame = _get_group_bar_frame(cell_frame, group_by=group_by, n_x=n_tracks)
        plot += geom_segment(
            data=group_bar_frame.with_columns(
                (pl.col("y_min") - 0.5).alias("x"),
                (pl.col("y_max") + 0.5).alias("xend"),
                pl.lit(_GROUP_BAR_Y).alias("y"),
            ),
            mapping=aes(x="x", xend="xend", y="y", yend="y", color=group_by),
            size=group_bars_size,
        )

    # DENDROGRAM above the tracks
    if dendrogram:
        assert paths is not None
        dendrogram_frame = _get_dendrogram_path_frame(
            paths, n_x=n_tracks, n_groups=len(group_order), group_centers=group_centers
        )
        dendrogram_frame = dendrogram_frame.select(
            pl.col("y").alias("x"),
            (pl.col("x") + 0.5 - (1 - _TRACK_HEIGHT) + _DENDROGRAM_GAP).alias("y"),
            pl.col("group"),
        )
        y_max_limit = dendrogram_frame["y"].max()
        plot += geom_path(
            data=dendrogram_frame,
            mapping=aes(x="x", y="y", group="group"),
            color=dendrogram_color,
            size=dendrogram_size,
        )

    # KEY-GROUP brackets on the right when `keys` was a mapping
    if key_labels and key_groups is not None:
        bracket_layers, x_max_limit = _key_groups_track_layers(
            key_groups,
            n_tracks=n_tracks,
            n_cells=n_cells,
            text_color=key_labels_text_color,
            bracket_color=key_labels_bracket_color,
            bracket_size=key_labels_bracket_size,
            width=key_labels_width,
            label_size_scale=key_labels_text_size,
        )
        for layer in bracket_layers:
            plot += layer

    plot += scale_y_continuous(
        breaks=[n_tracks - 1 - index + _TRACK_HEIGHT / 2 for index in range(n_tracks)],
        labels=keys,
        limits=[y_min_limit, y_max_limit],
        expand=[0, 0],
    )
    if group_bars and group_bars_labels:
        plot += scale_x_continuous(
            breaks=group_centers,
            labels=group_order,
            limits=[x_min_limit, x_max_limit],
            expand=[0, 0],
        )
        plot += guides(color="none")
    else:
        plot += scale_x_continuous(limits=[x_min_limit, x_max_limit], expand=[0, 0])
        plot += theme(axis_text_x=element_blank(), axis_ticks_x=element_blank())

    if interactive:
        plot += ggtb(size_zoomin=-1)

    return plot


def _group_order(groups: pl.Series) -> list[str]:
    """
    Order the groups present in `groups` from left to right.

    Notes
    -----
    Integer-like labels (cluster ids) sort numerically; otherwise categories
    keep their stored order and other labels sort alphabetically.
    """
    present = groups.cast(pl.String).unique()
    numeric = present.cast(pl.Int64, strict=False)
    if numeric.null_count() == 0:
        return [present[index] for index in numeric.arg_sort()]
    if isinstance(groups.dtype, (pl.Categorical, pl.Enum)):
        present_set = set(present.to_list())
        return [label for label in groups.cat.get_categories().to_list() if label in present_set]
    return present.sort().to_list()


def _key_groups_track_layers(
    groups: dict[str, list[str]],
    *,
    n_tracks: int,
    n_cells: int,
    text_color: str,
    bracket_color: str,
    bracket_size: float,
    width: float,
    label_size_scale: float,
) -> tuple[list[LayerSpec], float]:
    """
    Build the brackets spanning track groups on the right of the tracks.

    Returns
    -------
    layers : list[LayerSpec]
        The bracket path and label layers.
    x_max_limit : float
        The right x limit that keeps the brackets and labels inside the panel.
    """
    bracket_x = n_cells - 0.5 + _BRACKET_GAP_FRACTION * n_cells
    tip_length = _BRACKET_TIP_FRACTION * n_cells
    label_x = bracket_x + _BRACKET_LABEL_GAP_FRACTION * n_cells
    extension = width / 2
    path_records: list[dict] = []
    label_records: list[dict] = []
    cursor = 0
    for group_index, (label, values) in enumerate(groups.items()):
        # Track centers run top to bottom: track `i` is centered at n_tracks - 1 - i + h / 2.
        top = n_tracks - 1 - cursor + _TRACK_HEIGHT / 2 + extension
        bottom = n_tracks - 1 - (cursor + len(values) - 1) + _TRACK_HEIGHT / 2 - extension
        path_records.extend(
            [
                {"x": bracket_x - tip_length, "y": top, "g": group_index},
                {"x": bracket_x, "y": top, "g": group_index},
                {"x": bracket_x, "y": bottom, "g": group_index},
                {"x": bracket_x - tip_length, "y": bottom, "g": group_index},
            ]
        )
        label_records.append({"x": label_x, "y": (top + bottom) / 2, "label": label})
        cursor += len(values)
    layers = [
        geom_path(
            data=pl.DataFrame(path_records),
            mapping=aes(x="x", y="y", group="g"),
            color=bracket_color,
            size=bracket_size,
        ),
        geom_text(
            data=pl.DataFrame(label_records),
            mapping=aes(x="x", y="y", label="label"),
            color=text_color,
            size=_label_size_absolute(groups, scale=label_size_scale),
            angle=-90,
            hjust=0.5,
            vjust=0,
        ),
    ]
    return layers, label_x + _BRACKET_LABEL_ROOM_FRACTION * n_cells
