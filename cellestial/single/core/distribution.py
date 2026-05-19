from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from anndata import AnnData
from lets_plot.plot.core import FeatureSpec, PlotSpec

from cellestial.single.core.utilities import _distribution

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import PlotSpec


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
        If not provided, falls back to `fill`, `color`, or the variable column.
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
    geom_color : str | None, default=None
        Border color for all violins in the violin plot.
    point_color : str, default='#1f1f1f'
        Color for the points in the violin plot.
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

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    ValueError
        If `point_geom` is not one of the supported point geoms.

    Examples
    --------

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = cl.datasets.pbmc3k(cache_directory="data")

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

        data = cl.datasets.pbmc3k(cache_directory="data")

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

        data = cl.datasets.pbmc3k(cache_directory="data")

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
        If not provided, falls back to `fill`, `color`, or the variable column.
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
    geom_color : str | None, default=None
        Border color for all boxplots in the boxplot.
    point_color : str, default='#1f1f1f'
        Color for the points in the boxplot.
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

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    ValueError
        If `point_geom` is not one of the supported point geoms.

    Examples
    --------

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = cl.datasets.pbmc3k(cache_directory="data")

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

        data = cl.datasets.pbmc3k(cache_directory="data")

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

        data = cl.datasets.pbmc3k(cache_directory="data")

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
