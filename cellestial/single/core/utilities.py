from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    geom_boxplot,
    geom_histogram,
    geom_jitter,
    geom_point,
    geom_sina,
    geom_violin,
    ggplot,
    ggtb,
    position_dodge,
    position_jitterdodge,
)
from lets_plot.plot.core import FeatureSpec, PlotSpec

from cellestial.frames import build_frame
from cellestial.themes import _THEME_DIST
from cellestial.util import (
    _collect_aes_columns,
    _determine_axis,
    _drop_nonfinite_rows,
    _resolve_tooltips,
    _validate_tooltips,
    _warn,
)
from cellestial.util.errors import _unsupported_data_type

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec


def _distribution(
    data: AnnData,
    key: str | Sequence[str],
    *,
    frame: pl.DataFrame | None = None,
    group_by: str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    mapping: FeatureSpec | None = None,
    geom: Literal["violin", "boxplot", "histogram"] = "violin",
    axis: Literal[0, 1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    threshold: float | None = None,
    add_keys: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    geom_fill: str | None = None,
    geom_color: str | None = None,
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    point_geom: Literal["jitter", "point", "sina"] = "jitter",
    point_mapping: FeatureSpec | None = None,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    show_points: bool = True,
    interactive: bool = False,
    value_column: str = "value",
    variable_column: str = "variable",
    point_kwargs: dict[str, Any] | None = None,
    **geom_kwargs,
) -> PlotSpec:
    # Handling Data types
    if not isinstance(data, AnnData):
        raise _unsupported_data_type(data, AnnData)

    # HANDLE: mapping
    mapping = mapping or aes()
    point_mapping = point_mapping or aes()

    # merge standalone color/fill into mapping
    _mapping = mapping.as_dict()
    if color is not None and "color" not in _mapping:
        _mapping["color"] = color
    if fill is not None and "fill" not in _mapping:
        _mapping["fill"] = fill
    mapping = aes(**_mapping)

    mapping_color = _mapping.get("color")
    mapping_fill = _mapping.get("fill")

    # convert to list if single string
    if isinstance(key, str):
        keys = [key]
    elif isinstance(key, Sequence):
        keys = list(key)

    # handle value_column if single key is provided
    if len(keys) == 1:
        value_column = keys[0]

    # determine separator (group_by)
    group_by = group_by or mapping_fill or mapping_color
    # initialize point kwargs if necessary
    point_kwargs = {} if point_kwargs is None else dict(point_kwargs)

    # determine index to unpivot
    index = list(
        dict.fromkeys(c for c in (group_by, mapping_fill, mapping_color) if c is not None)
    )
    if add_keys is not None:
        if isinstance(add_keys, str):
            add_keys = [add_keys]
        index.extend(add_keys)

    # DETERMINE: axis if not provided
    axis = _determine_axis(data=data, keys=keys) if axis is None else axis

    # BUILD: the DataFrame (variable_keys is still needed for tooltip resolution)
    variable_keys: list[str] = []
    metadata_columns: list[str] = []
    _collect_aes_columns(
        data,
        keys=[*keys, group_by, mapping_fill, mapping_color, *(add_keys or [])],
        mapping=mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=axis,
    )
    _collect_aes_columns(
        data,
        keys=[],
        mapping=point_mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=axis,
    )
    # Resolve tooltips before the unpivot so tooltip fields reach both the frame
    # and the unpivot index; otherwise the unpivot drops tooltip-only columns.
    # for a single key the `variable` column just repeats the key name, so drop it
    tooltip_defaults = [value_column] if len(keys) == 1 else [variable_column, value_column]
    tooltips = _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=tooltip_defaults,
        metadata_columns=metadata_columns,
        axis=axis,
    )
    # Keep tooltip-referenced metadata columns through the unpivot.
    for column in metadata_columns:
        if column not in index and column not in keys:
            index.append(column)
    for column in point_mapping.as_dict().values():
        if (
            isinstance(column, str)
            and column not in index
            and column not in keys
            and column not in {value_column, variable_column}
        ):
            index.append(column)
    if frame is None:
        observation_column_name = None if tooltips == "none" else observations_name
        variable_column_name = None if tooltips == "none" else variables_name
        frame = build_frame(
            data=data,
            variable_keys=variable_keys,
            axis=axis,
            observations_name=observation_column_name,
            variables_name=variable_column_name,
            metadata_columns=metadata_columns,
        )

    if group_by is not None:
        frame = frame.filter(pl.col(group_by).is_not_null())

    # HANDLE: groups filter (categorical-only) on the grouping column
    if groups is not None and group_by is not None:
        if isinstance(groups, str):
            groups = [groups]
        if frame[group_by].dtype == pl.Categorical:
            frame = frame.filter(pl.col(group_by).is_in(list(groups)))
        else:
            msg = f"group_by `{group_by}` is not categorical, `groups` filter ignored"
            _warn(msg)

    # HANDLE: drop filter (categorical-only) on the grouping column
    if drop is not None and group_by is not None:
        if isinstance(drop, str):
            drop = [drop]
        if frame[group_by].dtype == pl.Categorical:
            frame = frame.filter(~pl.col(group_by).is_in(list(drop)).fill_null(False))
        else:
            msg = f"group_by `{group_by}` is not categorical, `drop` filter ignored"
            _warn(msg)

    frame = frame.unpivot(
        on=keys, index=index, value_name=value_column, variable_name=variable_column
    )
    frame = _drop_nonfinite_rows(frame, [value_column])
    if threshold is not None:
        frame = frame.filter(pl.col(value_column) >= threshold)
    if group_by is None:
        group_by = variable_column
    elif len(keys) > 1 and group_by != variable_column:
        _warn(
            f"Multiple keys provided; `group_by={group_by!r}` cannot share the "
            f"x-axis with the variable column.\nFalling back to "
            f"`group_by={variable_column!r}`. For a per-key panel, use the plural variant ({geom}s)."
        )
        group_by = variable_column

    # VALIDATE: tooltips were resolved before the frame build / unpivot above.
    _validate_tooltips(tooltips, frame)

    # for a single key the `variable` column just repeats the key name, so drop it
    geom_tooltips = [
        column for column in frame.columns if not (len(keys) == 1 and column == variable_column)
    ]

    # BUILD: the plot
    dst = ggplot(data=frame) + _THEME_DIST

    # add the geom layer
    if geom == "violin":
        dst += geom_violin(
            mapping=aes(
                x=group_by,
                y=value_column,
                **mapping.as_dict(),
            ),
            fill=geom_fill,
            color=geom_color,
            tooltips=geom_tooltips,
            **geom_kwargs,
        )
    elif geom == "boxplot":
        dst += geom_boxplot(
            mapping=aes(
                x=group_by,
                y=value_column,
                **mapping.as_dict(),
            ),
            fill=geom_fill,
            color=geom_color,
            tooltips=geom_tooltips,
            **geom_kwargs,
        )
    elif geom == "histogram":
        # histogram puts the value on the x-axis; grouping rides `fill` (from
        # mapping), not the x-axis, so `group_by` only filters rows here.
        dst += geom_histogram(
            mapping=aes(
                x=value_column,
                **mapping.as_dict(),
            ),
            fill=geom_fill,
            color=geom_color,
            tooltips=geom_tooltips,
            **geom_kwargs,
        )

    # handle the points (jitter,point,sina)
    if show_points:
        if point_geom in ["jitter", "point", "sina"]:
            geom_functions = {
                "jitter": geom_jitter,
                "point": geom_point,
                "sina": geom_sina,
            }
            dodge_width = geom_kwargs.get("width", 0.95)
            positions = {
                "jitter": position_jitterdodge(dodge_width=dodge_width),
                "point": position_dodge(width=dodge_width),
                "sina": None,
            }
            geom_function = geom_functions.get(point_geom, geom_jitter)
            if "position" not in point_kwargs:
                position = positions.get(point_geom, position_jitterdodge())
            else:
                position = point_kwargs.pop("position")

            point_kwargs.update(color=point_color, alpha=point_alpha, size=point_size)
            for key in point_mapping.as_dict():
                point_kwargs.pop(key, None)
            dst += geom_function(
                data=frame,
                mapping=aes(x=group_by, y=value_column, **point_mapping.as_dict()),
                tooltips=tooltips,
                position=position,
                **point_kwargs,
            )
        else:
            msg = "point_geom must be one of ['jitter','point','sina']."
            raise ValueError(msg)

    # handle interactive
    if interactive:
        dst += ggtb(size_zoomin=-1)

    return dst
