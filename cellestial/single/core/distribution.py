from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal
from warnings import warn

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    geom_boxplot,
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
    _determine_axis,
    _resolve_tooltips,
    _select_variable_keys,
    _validate_tooltips,
)
from cellestial.util.errors import UnsupportedDataTypeError

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec


def _distribution(
    data: AnnData,
    key: str | Sequence[str],
    *,
    group_by: str | None = None,
    mapping: FeatureSpec | None = None,
    geom: Literal["violin", "boxplot"] = "violin",
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
        msg = f"Unsupported data type: `{type(data)}`"
        raise UnsupportedDataTypeError(msg)

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
    index = list(dict.fromkeys(c for c in (group_by, mapping_fill, mapping_color) if c is not None))
    if add_keys is not None:
        if isinstance(add_keys, str):
            add_keys = [add_keys]
        index.extend(add_keys)

    # DETERMINE: axis if not provided
    axis = _determine_axis(data=data, keys=keys) if axis is None else axis

    # BUILD: the DataFrame
    variable_keys = _select_variable_keys(data=data, keys=keys)
    variable_keys.extend(_select_variable_keys(data=data, keys=add_keys))
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=observations_name,
        variables_name=variables_name,
    )

    if group_by is not None:
        frame = frame.filter(pl.col(group_by).is_not_null())

    frame = frame.unpivot(
        on=keys, index=index, value_name=value_column, variable_name=variable_column
    )
    frame = frame.drop_nulls(subset=[value_column])
    if threshold is not None:
        frame = frame.filter(pl.col(value_column) >= threshold)
    if group_by is None:
        group_by = variable_column
    elif len(keys) > 1 and group_by != variable_column:
        warn(
            f"Multiple keys provided; `group_by={group_by!r}` cannot share the "
            f"x-axis with the variable column. Falling back to "
            f"`group_by={variable_column!r}`. For a per-key panel, use the plural variant.",
            stacklevel=3,
        )
        group_by = variable_column

    # HANDLE: tooltips
    tooltips = _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=[variable_column, value_column],
    )
    _validate_tooltips(tooltips, frame)

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
            tooltips=frame.columns,
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
            tooltips=frame.columns,
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


def violin(
    data: AnnData,
    key: str | Sequence[str],
    *,
    group_by: str | None = None,
    mapping: FeatureSpec | None = None,
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
    """
    Violin Plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str | Sequence[str]
        The key(s) to get the values (numerical).
        e.g., 'total_counts' or a gene name.
    group_by : str | None, default=None
        Column to group observations on the x-axis.
        If not provided, falls back to ``fill``, ``color``, or the variable column.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    axis : {0,1} | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the violin plot (categorical).
        Shortcut for mapping=aes(color=...)
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the violin plot (categorical).
        Shortcut for mapping=aes(fill=...)
        e,g., 'cell_type' or 'leiden'.
    threshold : float | None, default=None
        If provided, filters out rows where the value column is below the threshold.
    add_keys : Sequence[str] | str | None, default=None
        Additional keys to include in the dataframe.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    geom_fill : str | None, default=None
        Fill color for all violins in the violin plot.

            **Accepts:**

            - hex code e.g. '#f1f1f1'
            - color name (https://lets-plot.org/python/pages/named_colors.html).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    geom_color : str | None, default=None
        Border color for all violins in the violin plot.

            **Accepts:**

            - hex code e.g. '#f1f1f1'
            - color name (https://lets-plot.org/python/pages/named_colors.html).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_color : str, default='#1f1f1f'
        Color for the points in the violin plot.

            **Accepts:**

            - hex code e.g. '#f1f1f1'
            - color name (https://lets-plot.org/python/pages/named_colors.html).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_alpha : float, default=0.7
        Alpha (transparency) for the points in the violin plot.
    point_size : float, default=0.5
        Size for the points in the violin plot.
    point_geom : {'jitter','point','sina'}, default is 'jitter'
        Geom type of the points, default is geom_jitter.
    point_mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the points, the result of `aes()`.
    observations_name : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default='Variable'
        The name to give to variable index column in the dataframe.
    show_points : bool, default=True
        Whether to show points.
    interactive : bool, default=False
        Whether to make the plot interactive.
    variable_column : str, default='variable'
        The name of the variable column in the dataframe.
    value_column : str, default='value'
        The name of the value column in the dataframe.
    point_kwargs : dict[str, Any] | None, default=None
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html
    **geom_kwargs
        Additional parameters for the `geom_violin` layer.
        For more information on geom_violin parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_violin.html

    Returns
    -------
    PlotSpec
        Violin plot.

    Examples
    --------

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = sc.read_h5ad('data/pbmc3k_pped.h5ad')

        violin = (
            cl.violin(
                data,
                "CD14",
                fill="cell_type_lvl1",
                scale="width",
                point_size=2,
                threshold=0.1,
                trim=False,
            )
            + ggsize(800, 400)
            + scale_fill_brewer(palette="Set2")
            + guides(fill=guide_legend(ncol=2))
        )

        violin

    Remove the points.

    .. jupyter-execute::
        :emphasize-lines: 17

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = sc.read_h5ad('data/pbmc3k_pped.h5ad')

        violin = (
            cl.violin(
                data,
                "CD14",
                fill="cell_type_lvl1",
                scale="width",
                point_size=2,
                threshold=0.1,
                trim=False,
                show_points=False,
            )
            + ggsize(800, 400)
            + scale_fill_brewer(palette="Set2")
            + guides(fill=guide_legend(ncol=2))
        )

        violin

    Providing a list of keys.

    .. jupyter-execute::
        :emphasize-lines: 11

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = sc.read_h5ad('data/pbmc3k_pped.h5ad')

        violin = (
            cl.violin(
                data,
                ["pct_counts_in_top_200_genes", "n_genes_by_counts"],
                fill="cell_type_lvl1",
                trim=False,
                scale="width",
                point_size=0.5,
                point_alpha=0.4,
            )
            + scale_y_log2()
            + ggsize(800, 400)
        )
        violin
    """
    return _distribution(
        data=data,
        key=key,
        group_by=group_by,
        mapping=mapping,
        geom="violin",
        axis=axis,
        color=color,
        fill=fill,
        threshold=threshold,
        add_keys=add_keys,
        tooltips=tooltips,
        geom_fill=geom_fill,
        geom_color=geom_color,
        point_color=point_color,
        point_alpha=point_alpha,
        point_size=point_size,
        point_geom=point_geom,
        point_mapping=point_mapping,
        observations_name=observations_name,
        variables_name=variables_name,
        show_points=show_points,
        interactive=interactive,
        value_column=value_column,
        variable_column=variable_column,
        point_kwargs=point_kwargs,
        **geom_kwargs,
    )


def boxplot(
    data: AnnData,
    key: str | Sequence[str],
    *,
    group_by: str | None = None,
    mapping: FeatureSpec | None = None,
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
    """
    Boxplot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str | Sequence[str]
        The key(s) to get the values (numerical).
        e.g., 'total_counts' or a gene name.
    group_by : str | None, default=None
        Column to group observations on the x-axis.
        If not provided, falls back to ``fill``, ``color``, or the variable column.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    axis : {0,1} | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the boxplot (categorical).
        Shortcut for mapping=aes(color=...)
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the boxplot (categorical).
        Shortcut for mapping=aes(fill=...)
        e,g., 'cell_type' or 'leiden'.
    threshold : float | None, default=None
        If provided, filters out rows where the value column is below the threshold.
    add_keys : Sequence[str] | str | None, default=None
        Additional keys to include in the dataframe.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    geom_fill : str | None, default=None
        Fill color for all boxplots in the boxplot.

            **Accepts:**

            - hex code e.g. '#f1f1f1'
            - color name (https://lets-plot.org/python/pages/named_colors.html).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    geom_color : str | None, default=None
        Border color for all boxplots in the boxplot.

            **Accepts:**

            - hex code e.g. '#f1f1f1'
            - color name (https://lets-plot.org/python/pages/named_colors.html).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_color : str, default='#1f1f1f'
        Color for the points in the boxplot.

            **Accepts:**

            - hex code e.g. '#f1f1f1'
            - color name (https://lets-plot.org/python/pages/named_colors.html).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_alpha : float, default=0.7
        Alpha (transparency) for the points in the boxplot.
    point_size : float, default=0.5
        Size for the points in the boxplot.
    point_geom : {'jitter','point','sina'}, default is 'jitter'
        Geom type of the points, default is geom_jitter.
    point_mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the points, the result of `aes()`.
    observations_name : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default='Variable'
        The name to give to variable index column in the dataframe.
    show_points : bool, default=True
        Whether to show points.
    interactive : bool, default=False
        Whether to make the plot interactive.
    variable_column : str, default='variable'
        The name of the variable column in the dataframe.
    value_column : str, default='value'
        The name of the value column in the dataframe.
    point_kwargs : dict[str, Any] | None, default=None
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html
    **geom_kwargs
        Additional parameters for the `geom_boxplot` layer.
        For more information on geom_boxplot parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_boxplot.html

    Returns
    -------
    PlotSpec
        Boxplot.

    Examples
    --------

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = sc.read_h5ad('data/pbmc3k_pped.h5ad')

        boxplot = (
            cl.boxplot(
                data,
                "CD14",
                fill="cell_type_lvl1",
                point_size=2,
                threshold=0.1,
            )
            + ggsize(800, 400)
            + scale_fill_brewer(palette="Set2")
            + guides(fill=guide_legend(ncol=2))
        )

        boxplot

    Remove the points.

    .. jupyter-execute::
        :emphasize-lines: 15

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = sc.read_h5ad('data/pbmc3k_pped.h5ad')

        boxplot = (
            cl.boxplot(
                data,
                "CD14",
                fill="cell_type_lvl1",
                point_size=2,
                threshold=0.1,
                show_points=False,
            )
            + ggsize(800, 400)
            + scale_fill_brewer(palette="Set2")
            + guides(fill=guide_legend(ncol=2))
        )

        boxplot

    Providing a list of keys.

    .. jupyter-execute::
        :emphasize-lines: 11

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = sc.read_h5ad('data/pbmc3k_pped.h5ad')

        boxplot = (
            cl.boxplot(
                data,
                ["pct_counts_in_top_200_genes", "n_genes_by_counts"],
                fill="cell_type_lvl1",
                point_size=0.3,
                point_alpha=0.4,
            )
            + scale_y_log2()
            + ggsize(800, 400)
        )
        boxplot
    """
    return _distribution(
        data=data,
        key=key,
        group_by=group_by,
        mapping=mapping,
        geom="boxplot",
        axis=axis,
        color=color,
        fill=fill,
        threshold=threshold,
        add_keys=add_keys,
        tooltips=tooltips,
        geom_fill=geom_fill,
        geom_color=geom_color,
        point_color=point_color,
        point_alpha=point_alpha,
        point_size=point_size,
        point_geom=point_geom,
        point_mapping=point_mapping,
        observations_name=observations_name,
        variables_name=variables_name,
        show_points=show_points,
        interactive=interactive,
        value_column=value_column,
        variable_column=variable_column,
        point_kwargs=point_kwargs,
        **geom_kwargs,
    )
