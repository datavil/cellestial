from __future__ import annotations

from collections.abc import Iterable
from math import ceil
from typing import TYPE_CHECKING, Any

from anndata import AnnData
from lets_plot import (
    aes,
    geom_boxplot,
    geom_jitter,
    geom_violin,
    gggrid,
    ggplot,
    ggtb,
    guide_legend,
    guides,
    layer_tooltips,
)
from lets_plot.plot.core import FeatureSpec, LayerSpec, PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

from cellestial.frames import _axis_data, _construct_cell_frame, _construct_var_frame
from cellestial.themes import _THEME_BOXPLOT, _THEME_VIOLIN
from cellestial.util import _build_tooltips, _decide_tooltips

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec


def violin(
    data: AnnData,
    key: str | Iterable[str],
    *,
    color: str | None = None,
    fill: str | None = None,
    violin_fill: str = "#FF00FF",
    violin_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    trim: bool = False,
    barcode_name: str = "Barcode",
    var_name: str = "Gene",
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | str | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | str | None = None,
    tooltips_title: str | None = None,
    interactive: bool = False,
    point_kwargs: dict[str, Any] | None = None,
    **violin_kwargs: dict[str, Any],
) -> PlotSpec:
    """
    Violin Plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str
        The key to get the values (numerical).
        e.g., 'total_counts' or a gene name.
    color : str | None, default=None
        Color aesthetic to split the violin plot (categorical).
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the violin plot (categorical).
        e,g., 'cell_type' or 'leiden'.
    violin_fill : str, default="#FF00FF"
        Fill color for all violins in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    violin_color : str, default="#2f2f2f"
        Border color for all violins in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_color : str, default="#1f1f1f"
        Color for the points in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_alpha : float, default=0.7
        Alpha (transparency) for the points in the violin plot.
    point_size : float, default=0.5
        Size for the points in the violin plot.
    trim : bool, default=False
        Whether to trim the violin plot.
    barcode_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    var_name : str, default="Gene"
        The name to give to variable index column in the dataframe.
    show_tooltips : bool, default=True
        Whether to show tooltips.
    show_points : bool, default=True
        Whether to show points.
    add_tooltips : list[str] | tuple[str] | Iterable[str] | str | None, default=None
        Additional tooltips to show.
    custom_tooltips : list[str] | tuple[str] | Iterable[str] | str | None, default=None
        Custom tooltips to show.
    tooltips_title : str | None, default=None
        Title for the tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    point_kwargs : dict[str, Any] | None, default=None
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html
    **violin_kwargs : dict[str, Any]
        Additional parameters for the `geom_violin` layer.
        For more information on geom_violin parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_violin.html

    Returns
    -------
    PlotSpec
        Violin plot.
    """
    # Handling Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # handle violin_kwargs
    if violin_kwargs:
        if "tooltips" in violin_kwargs:
            msg = "violin tooltips are non-customizable by `violin_kwargs`"
            raise KeyError(msg)

    # handle point_kwargs
    if point_kwargs is None:
        point_kwargs = {}
    else:
        if "tooltips" in point_kwargs:
            msg = "use tooltips args within the function instead of adding `'tooltips' : 'value'` to `point_kwargs`\n"
            raise KeyError(msg)

    if isinstance(key,str):
        # handle tooltips
        axis = _axis_data(data=data, key=key)
        identifier = barcode_name if axis == 0 else var_name
        base_tooltips = [identifier, key]
        if color is not None:
            base_tooltips.append(color)
        if fill is not None:
            base_tooltips.append(fill)

        tooltips = _decide_tooltips(
            base_tooltips=base_tooltips,
            add_tooltips=add_tooltips,
            custom_tooltips=custom_tooltips,
            show_tooltips=show_tooltips,
        )

        tooltips_object = _build_tooltips(
            tooltips=tooltips,
            cluster_name=None,
            key=key,
            title=tooltips_title,
            clustering=False,
        )

        # construct the frame
        all_keys = []
        if key is not None:
            all_keys.append(key)
        if tooltips != "none":
            for tooltip in tooltips:
                if tooltip not in all_keys and tooltip != identifier:
                    all_keys.append(tooltip)

        if axis == 0:  # for obs and var_names
            frame = _construct_cell_frame(
                data=data,
                keys=all_keys,
                xy=None,
                barcode_name=barcode_name,
            )
        elif axis == 1:  # for var
            frame = _construct_var_frame(
                data=data,
                keys=all_keys,
                var_name=var_name,
            )
        # handle fill and color
        violin_fill = None if fill is not None else violin_fill
        violin_color = None if color is not None else violin_color
        # handle violin tooltips
        violin_tooltips = [key]
        violin_tooltips.append(color) if color is not None else None
        violin_tooltips.append(fill) if fill is not None else None
        # generate the plot
        vln = (
            ggplot(data=frame)
            + geom_violin(
                data=frame,
                mapping=aes(x=fill, y=key, color=color, fill=fill),
                fill=violin_fill,
                color=violin_color,
                trim=trim,
                tooltips=layer_tooltips(violin_tooltips),
                **violin_kwargs,
            )
            + _THEME_VIOLIN
        )
        # handle the point (jitter)
        if show_points:
            vln += geom_jitter(
                data=frame,
                mapping=aes(x=fill, y=key),
                color=point_color,
                alpha=point_alpha,
                size=point_size,
                tooltips=tooltips_object,
                **point_kwargs,
            )



        # handle interactive
        if interactive:
            vln += ggtb()

    elif isinstance(key,Iterable):
        pass

    # wrap the legend
    if fill is not None:
        n_distinct = frame.select(fill).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            vln = vln + guides(fill=guide_legend(ncol=ncol))
    if color is not None:
        n_distinct = frame.select(color).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            vln = vln + guides(color=guide_legend(ncol=ncol))

    return vln

def boxplot(
    data: AnnData,
    key: str,
    *,
    color: str | None = None,
    fill: str | None = None,
    boxplot_fill: str = "#FF00FF",
    boxplot_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    barcode_name: str = "Barcode",
    var_name: str = "Gene",
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | str | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | str | None = None,
    tooltips_title: str | None = None,
    interactive: bool = False,
    point_kwargs: dict[str, Any] | None = None,
    **boxplot_kwargs: dict[str, Any],
) -> PlotSpec:
    """
    Boxplot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str
        The key to get the values (numerical).
        e.g., 'total_counts' or a gene name.
    color : str | None, default=None
        Color aesthetic to split the boxplot (categorical).
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the boxplot (categorical).
        e,g., 'cell_type' or 'leiden'.
    boxplot_fill : str, default="#FF00FF"
        Fill color for all boxplots in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    boxplot_color : str, default="#2f2f2f"
        Border color for all boxplots in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_color : str, default="#1f1f1f"
        Color for the points in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_alpha : float, default=0.7
        Alpha (transparency) for the points in the boxplot.
    point_size : float, default=0.5
        Size for the points in the boxplot.
    barcode_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    var_name : str, default="Gene"
        The name to give to variable index column in the dataframe.
    show_tooltips : bool, default=True
        Whether to show tooltips.
    show_points : bool, default=True
        Whether to show points.
    add_tooltips : list[str] | tuple[str] | Iterable[str] | str | None, default=None
        Additional tooltips to show.
    custom_tooltips : list[str] | tuple[str] | Iterable[str] | str | None, default=None
        Custom tooltips to show.
    tooltips_title : str | None, default=None
        Title for the tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    point_kwargs : dict[str, Any] | None, default=None
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html
    **boxplot_kwargs : dict[str, Any]
        Additional parameters for the `geom_boxplot` layer.
        For more information on geom_boxplot parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_boxplot.html

    Returns
    -------
    PlotSpec
        Boxplot.
    """
    # Handling Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # handle violin_kwargs
    if boxplot_kwargs:
        if "tooltips" in boxplot_kwargs:
            msg = "violin tooltips are non-customizable by `boxplot_kwargs`"
            raise KeyError(msg)
    # handle point_kwargs
    if point_kwargs is None:
        point_kwargs = {}
    else:
        if "tooltips" in point_kwargs:
            msg = "use tooltips args within the function instead of adding `'tooltips' : 'value'` to `point_kwargs`\n"
            raise KeyError(msg)
    # handle tooltips
    axis = _axis_data(data=data, key=key)
    identifier = barcode_name if axis == 0 else var_name
    base_tooltips = [identifier, key]
    if color is not None:
        base_tooltips.append(color)
    if fill is not None:
        base_tooltips.append(fill)

    tooltips = _decide_tooltips(
        base_tooltips=base_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        show_tooltips=show_tooltips,
    )

    tooltips_object = _build_tooltips(
        tooltips=tooltips,
        cluster_name=None,
        key=key,
        title=tooltips_title,
        clustering=False,
    )

    # construct the frame
    all_keys = []
    if key is not None:
        all_keys.append(key)
    if tooltips != "none":
        for tooltip in tooltips:
            if tooltip not in all_keys and tooltip != identifier:
                all_keys.append(tooltip)

    if axis == 0:  # for obs and var_names
        frame = _construct_cell_frame(
            data=data,
            keys=all_keys,
            xy=None,
            barcode_name=barcode_name,
        )
    elif axis == 1:  # for var
        frame = _construct_var_frame(
            data=data,
            keys=all_keys,
            var_name=var_name,
        )
    # handle fill and color
    boxplot_fill = None if fill is not None else boxplot_fill
    boxplot_color = None if color is not None else boxplot_color
    # handle box tooltips
    boxplot_tooltips = [key]
    boxplot_tooltips.append(color) if color is not None else None
    boxplot_tooltips.append(fill) if fill is not None else None

    # handle boxplot_kwargs
    if boxplot_kwargs is None:
        boxplot_kwargs = {}

    # generate the plot
    bxplt = (
        ggplot(data=frame)
        + geom_boxplot(
            data=frame,
            mapping=aes(x=fill, y=key, color=color, fill=fill),
            fill=boxplot_fill,
            color=boxplot_color,
            tooltips=layer_tooltips(boxplot_tooltips),
            **boxplot_kwargs,
        )
        + _THEME_BOXPLOT
    )
    # handle the point (jitter)
    if show_points:
        bxplt += geom_jitter(
            data=frame,
            mapping=aes(x=fill, y=key),
            color=point_color,
            alpha=point_alpha,
            size=point_size,
            tooltips=tooltips_object,
            **point_kwargs,
        )

    # wrap the legend
    if fill is not None:
        n_distinct = frame.select(fill).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            bxplt = bxplt + guides(fill=guide_legend(ncol=ncol))
    if color is not None:
        n_distinct = frame.select(color).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            bxplt = bxplt + guides(color=guide_legend(ncol=ncol))

    # handle interactive
    if interactive:
        bxplt += ggtb()

    return bxplt


def violins(
    data,
    keys: list[str] | tuple[str] | Iterable[str],
    *,
    color: str | None = None,
    fill: str | None = None,
    violin_fill: str = "#FF00FF",
    violin_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    trim: bool = False,
    barcode_name: str = "Barcode",
    var_name: str = "Gene",
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | str | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | str | None = None,
    tooltips_title: str | None = None,
    interactive: bool = False,
    layers: list | tuple | Iterable | FeatureSpec | LayerSpec | None = None,
    multi_panel: bool = True,
    variable_name: str = "variable",
    value_name: str = "value",
    # grid args
    ncol: int | None = None,
    sharex: str | None = None,
    sharey: str | None = None,
    widths: list | None = None,
    heights: list | None = None,
    hspace: float | None = None,
    vspace: float | None = None,
    fit: bool | None = None,
    align: bool | None = None,
    # other kwargs
    point_kwargs: dict[str, Any] | None = None,
    **violin_kwargs: dict[str, Any],
) -> SupPlotsSpec | PlotSpec:
    """
    Violin Plots.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    keys : list[str] | tuple[str] | Iterable[str]
        The keys to get the values (numerical).
        e.g., ['total_counts', 'pct_counts_in_top_50_genes'] or a list of gene names.
    color : str | None, default=None
        Color aesthetic to split the violin plot (categorical).
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the violin plot (categorical).
        e,g., 'cell_type' or 'leiden'.
    violin_fill : str, default="#FF00FF"
        Fill color for all violins in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    violin_color : str, default="#2f2f2f"
        Border color for all violins in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_color : str, default="#1f1f1f"
        Color for the points in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_alpha : float, default=0.7
        Alpha (transparency) for the points in the violin plot.
    point_size : float, default=0.5
        Size for the points in the violin plot.
    trim : bool, default=False
        Whether to trim the violin plot.
    barcode_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    var_name : str, default="Gene"
        The name to give to variable index column in the dataframe.
    show_tooltips : bool, default=True
        Whether to show tooltips.
    show_points : bool, default=True
        Whether to show points.
    add_tooltips : list[str] | tuple[str] | Iterable[str] | str | None, default=None
        Additional tooltips to show.
    custom_tooltips : list[str] | tuple[str] | Iterable[str] | str | None, default=None
        Custom tooltips to show.
    tooltips_title : str | None, default=None
        Title for the tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    layers : list | tuple | Iterable | FeatureSpec | LayerSpec | None, default=None
        Additional layers to add to the plot.
    multi_panel : bool, default=True
        Whether to plot the violin plots in a grid.
        If True, the violin plots will be plotted in a grid and returns a `SupPlotsSpec` object.
        If False, the violin plots will be merged into a single panel and retruns a `PlotSpec` object.
    variable_name : str, default="variable"
        The name of the variable column in the dataframe.
    value_name : str, default="value"
        The name of the value column in the dataframe.
    ncol : int, default=None
        Number of columns in grid. If not specified, shows plots horizontally, in one row.
    sharex, sharey : bool, default=None
        Controls sharing of axis limits between subplots in the grid.
        `all`/True - share limits between all subplots.
        `none`/False - do not share limits between subplots.
        `row` - share limits between subplots in the same row.
        `col` - share limits between subplots in the same column.
    widths : list[float], default=None
        Relative width of each column of grid, left to right.
    heights : list[float], default=None
        Relative height of each row of grid, top-down.
    hspace : float | None = None
        Cell horizontal spacing in px.
    vspace : float | None = None
        Cell vertical spacing in px.
    fit : bool, default=True
        Whether to stretch each plot to match the aspect ratio of its cell (fit=True),
        or to preserve the original aspect ratio of plots (fit=False).
    align : bool, default=False
        If True, align inner areas (i.e. “geom” bounds) of plots.
        However, cells containing other (sub)grids are not participating
        in the plot “inner areas” layouting.
    point_kwargs : dict[str, Any] | None, default=None
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html
    **violin_kwargs : dict[str, Any]
        Additional parameters for the `geom_violin` layer.
        For more information on geom_violin parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_violin.html
    """
    if multi_panel:  # standard grid plotting
        plots = []
        for key in keys:
            vln = violin(
                data=data,
                key=key,
                color=color,
                fill=fill,
                violin_fill=violin_fill,
                violin_color=violin_color,
                point_color=point_color,
                point_alpha=point_alpha,
                point_size=point_size,
                trim=trim,
                barcode_name=barcode_name,
                var_name=var_name,
                show_tooltips=show_tooltips,
                show_points=show_points,
                add_tooltips=add_tooltips,
                custom_tooltips=custom_tooltips,
                tooltips_title=tooltips_title,
                interactive=interactive,
                point_kwargs=point_kwargs,
                **violin_kwargs,
            )
            # handle the layers
            if layers is not None:
                if not isinstance(layers, Iterable):
                    layers = [layers]
                for layer in list(layers):
                    vln += layer

            plots.append(vln)

        vlns = gggrid(
            plots,
            ncol=ncol,
            sharex=sharex,
            sharey=sharey,
            widths=widths,
            heights=heights,
            hspace=hspace,
            vspace=vspace,
            fit=fit,
            align=align,
        )

    else:  # unpivot the data so that it can be plotted in a single (combined) panel
        # Handling Data types
        if not isinstance(data, AnnData):
            msg = "data must be an `AnnData` object"
            raise TypeError(msg)

        # handle violin_kwargs
        if violin_kwargs:
            if "tooltips" in violin_kwargs:
                msg = "violin tooltips are non-customizable by `violin_kwargs`"
                raise KeyError(msg)

        # handle point_kwargs
        if point_kwargs is None:
            point_kwargs = {}
        else:
            if "tooltips" in point_kwargs:
                msg = "use tooltips args within the function instead of adding `'tooltips' : 'value'` to `point_kwargs`\n"
                raise KeyError(msg)

        # handle tooltips
        axis = _axis_data(data=data, key=keys[0])
        identifier = barcode_name if axis == 0 else var_name
        base_tooltips = [identifier]
        if color is not None:
            base_tooltips.append(color)
        if fill is not None:
            base_tooltips.append(fill)

        tooltips = _decide_tooltips(
            base_tooltips=base_tooltips,
            add_tooltips=add_tooltips,
            custom_tooltips=custom_tooltips,
            show_tooltips=show_tooltips,
        )

        tooltips_object = _build_tooltips(
            tooltips=tooltips,
            cluster_name=None,
            title=tooltips_title,
            clustering=False,
        )

        # construct the frame
        all_keys = list(keys)
        if tooltips != "none":
            for tooltip in tooltips:
                if tooltip not in all_keys and tooltip != identifier:
                    all_keys.append(tooltip)

        if axis == 0:  # for obs and var_names
            frame = _construct_cell_frame(
                data=data,
                keys=all_keys,
                xy=None,
                barcode_name=barcode_name,
            )
        elif axis == 1:  # for var
            frame = _construct_var_frame(
                data=data,
                keys=all_keys,
                var_name=var_name,
            )

        print(frame.columns)
        # unpivot the data
        if tooltips != "none":
            frame = frame.unpivot(
                index=tooltips, variable_name=variable_name, value_name=value_name
            )
        else:
            frame = frame.unpivot(
                index=barcode_name, variable_name=variable_name, value_name=value_name
            )
        # handle fill and color
        violin_fill = None if fill is not None else violin_fill
        violin_color = None if color is not None else violin_color
        # handle violin tooltips
        violin_tooltips = [variable_name]
        violin_tooltips.append(color) if color is not None else None
        violin_tooltips.append(fill) if fill is not None else None
        # generate the plot
        vlns = (
            ggplot(data=frame)
            + geom_violin(
                data=frame,
                mapping=aes(x=variable_name, y=value_name, color=color, fill=fill),
                fill=violin_fill,
                color=violin_color,
                trim=trim,
                tooltips=layer_tooltips(violin_tooltips),
                **violin_kwargs,
            )
            + _THEME_VIOLIN
        )
        # handle the point (jitter)
        if show_points:
            vlns += geom_jitter(
                data=frame,
                mapping=aes(x=variable_name, y=value_name),
                color=point_color,
                alpha=point_alpha,
                size=point_size,
                tooltips=tooltips_object,
                **point_kwargs,
            )

        # wrap the legend
        if fill is not None:
            n_distinct = frame.select(variable_name).unique().height
            if n_distinct > 10:
                ncol = ceil(n_distinct / 10)
                vln = vln + guides(fill=guide_legend(ncol=ncol))
        if color is not None:
            n_distinct = frame.select(variable_name).unique().height
            if n_distinct > 10:
                ncol = ceil(n_distinct / 10)
                vln = vln + guides(color=guide_legend(ncol=ncol))

        # handle the layers
        if layers is not None:
            if not isinstance(layers, Iterable):
                layers = [layers]
            for layer in list(layers):
                vlns += layer

    # handle interactive
    if interactive:
        vlns += ggtb()

    return vlns


def boxplots(
    data,
    keys: list[str] | tuple[str] | Iterable[str],
    *,
    color: str | None = None,
    fill: str | None = None,
    boxplot_fill: str = "#FF00FF",
    boxplot_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    barcode_name: str = "Barcode",
    var_name: str = "Gene",
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | str | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | str | None = None,
    tooltips_title: str | None = None,
    interactive: bool = False,
    layers: list | tuple | Iterable | FeatureSpec | LayerSpec | None = None,
    multi_panel: bool = True,
    variable_name: str = "variable",
    value_name: str = "value",
    # grid args
    ncol: int | None = None,
    sharex: str | None = None,
    sharey: str | None = None,
    widths: list | None = None,
    heights: list | None = None,
    hspace: float | None = None,
    vspace: float | None = None,
    fit: bool | None = None,
    align: bool | None = None,
    # other kwargs
    point_kwargs: dict[str, Any] | None = None,
    **boxplot_kwargs: dict[str, Any],
) -> SupPlotsSpec | PlotSpec:
    """
    Boxplots.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    keys : list[str] | tuple[str] | Iterable[str]
        The keys to get the values (numerical).
        e.g., ['total_counts', 'pct_counts_in_top_50_genes'] or a list of gene names.
    color : str | None, default=None
        Color aesthetic to split the boxplot (categorical).
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the boxplot (categorical).
        e,g., 'cell_type' or 'leiden'.
    boxplot_fill : str, default="#FF00FF"
        Fill color for all boxplots in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    boxplot_color : str, default="#2f2f2f"
        Border color for all boxplots in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_color : str, default="#1f1f1f"
        Color for the points in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_alpha : float, default=0.7
        Alpha (transparency) for the points in the boxplot.
    point_size : float, default=0.5
        Size for the points in the boxplot.
    barcode_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    var_name : str, default="Gene"
        The name to give to variable index column in the dataframe.
    show_tooltips : bool, default=True
        Whether to show tooltips.
    show_points : bool, default=True
        Whether to show points.
    add_tooltips : list[str] | tuple[str] | Iterable[str] | str | None, default=None
        Additional tooltips to show.
    custom_tooltips : list[str] | tuple[str] | Iterable[str] | str | None, default=None
        Custom tooltips to show.
    tooltips_title : str | None, default=None
        Title for the tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    layers : list | tuple | Iterable | FeatureSpec | LayerSpec | None, default=None
        Additional layers to add to the plot.
    multi_panel : bool, default=True
        Whether to plot the boxplots in a grid.
        If True, the boxplots will be plotted in a grid and returns a `SupPlotsSpec` object.
        If False, the boxplots will be merged into a single panel and retruns a `PlotSpec` object.
    variable_name : str, default="variable"
        The name of the variable column in the dataframe.
    value_name : str, default="value"
        The name of the value column in the dataframe.
    ncol : int, default=None
        Number of columns in grid. If not specified, shows plots horizontally, in one row.
    sharex, sharey : bool, default=None
        Controls sharing of axis limits between subplots in the grid.
        `all`/True - share limits between all subplots.
        `none`/False - do not share limits between subplots.
        `row` - share limits between subplots in the same row.
        `col` - share limits between subplots in the same column.
    widths : list[float], default=None
        Relative width of each column of grid, left to right.
    heights : list[float], default=None
        Relative height of each row of grid, top-down.
    hspace : float | None = None
        Cell horizontal spacing in px.
    vspace : float | None = None
        Cell vertical spacing in px.
    fit : bool, default=True
        Whether to stretch each plot to match the aspect ratio of its cell (fit=True),
        or to preserve the original aspect ratio of plots (fit=False).
    align : bool, default=False
        If True, align inner areas (i.e. “geom” bounds) of plots.
        However, cells containing other (sub)grids are not participating
        in the plot “inner areas” layouting.
    point_kwargs : dict[str, Any] | None, default=None
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html
    **boxplot_kwargs : dict[str, Any]
        Additional parameters for the `geom_boxplot` layer.
        For more information on geom_boxplot parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_boxplot.html

    Returns
    -------
    SupPlotsSpec | PlotSpec
        Boxplots.
    """
    if multi_panel:  # standard grid plotting
        plots = []
        for key in keys:
            bxplt = boxplot(
                data=data,
                key=key,
                color=color,
                fill=fill,
                boxplot_fill=boxplot_fill,
                boxplot_color=boxplot_color,
                point_color=point_color,
                point_alpha=point_alpha,
                point_size=point_size,
                barcode_name=barcode_name,
                var_name=var_name,
                show_tooltips=show_tooltips,
                show_points=show_points,
                add_tooltips=add_tooltips,
                custom_tooltips=custom_tooltips,
                tooltips_title=tooltips_title,
                interactive=interactive,
                point_kwargs=point_kwargs,
                **boxplot_kwargs,
            )
            # handle the layers
            if layers is not None:
                if not isinstance(layers, Iterable):
                    layers = [layers]
                for layer in list(layers):
                    bxplt += layer

            plots.append(bxplt)

        bxplts = gggrid(
            plots,
            ncol=ncol,
            sharex=sharex,
            sharey=sharey,
            widths=widths,
            heights=heights,
            hspace=hspace,
            vspace=vspace,
            fit=fit,
            align=align,
        )

    else:  # unpivot the data so that it can be plotted in a single (combined) panel
        # Handling Data types
        if not isinstance(data, AnnData):
            msg = "data must be an `AnnData` object"
            raise TypeError(msg)

        # handle boxplot_kwargs
        if boxplot_kwargs:
            if "tooltips" in boxplot_kwargs:
                msg = "boxplot tooltips are non-customizable by `boxplot_kwargs`"
                raise KeyError(msg)

        # handle point_kwargs
        if point_kwargs is None:
            point_kwargs = {}
        else:
            if "tooltips" in point_kwargs:
                msg = "use tooltips args within the function instead of adding `'tooltips' : 'value'` to `point_kwargs`\n"
                raise KeyError(msg)

        # handle tooltips
        axis = _axis_data(data=data, key=keys[0])
        identifier = barcode_name if axis == 0 else var_name
        base_tooltips = [identifier]
        if color is not None:
            base_tooltips.append(color)
        if fill is not None:
            base_tooltips.append(fill)

        tooltips = _decide_tooltips(
            base_tooltips=base_tooltips,
            add_tooltips=add_tooltips,
            custom_tooltips=custom_tooltips,
            show_tooltips=show_tooltips,
        )

        tooltips_object = _build_tooltips(
            tooltips=tooltips,
            cluster_name=None,
            title=tooltips_title,
            clustering=False,
        )

        # construct the frame
        all_keys = list(keys)
        if tooltips != "none":
            for tooltip in tooltips:
                if tooltip not in all_keys and tooltip != identifier:
                    all_keys.append(tooltip)

        if axis == 0:  # for obs and var_names
            frame = _construct_cell_frame(
                data=data,
                keys=all_keys,
                xy=None,
                barcode_name=barcode_name,
            )
        elif axis == 1:  # for var
            frame = _construct_var_frame(
                data=data,
                keys=all_keys,
                var_name=var_name,
            )
        # unpivot the data
        if tooltips != "none":
            frame = frame.unpivot(
                index=tooltips, variable_name=variable_name, value_name=value_name
            )
        else:
            frame = frame.unpivot(
                index=barcode_name, variable_name=variable_name, value_name=value_name
            )
        # handle fill and color
        boxplot_fill = None if fill is not None else boxplot_fill
        boxplot_color = None if color is not None else boxplot_color
        # handle boxplot tooltips
        boxplot_tooltips = [variable_name]
        boxplot_tooltips.append(color) if color is not None else None
        boxplot_tooltips.append(fill) if fill is not None else None
        # generate the plot
        bxplts = (
            ggplot(data=frame)
            + geom_boxplot(
                data=frame,
                mapping=aes(x=variable_name, y=value_name, color=color, fill=fill),
                fill=boxplot_fill,
                color=boxplot_color,
                tooltips=layer_tooltips(boxplot_tooltips),
                **boxplot_kwargs,
            )
            + _THEME_BOXPLOT
        )
        # handle the point (jitter)
        if show_points:
            bxplts += geom_jitter(
                data=frame,
                mapping=aes(x=variable_name, y=value_name),
                color=point_color,
                alpha=point_alpha,
                size=point_size,
                tooltips=tooltips_object,
                **point_kwargs,
            )

        # wrap the legend
        if fill is not None:
            n_distinct = frame.select(variable_name).unique().height
            if n_distinct > 10:
                ncol = ceil(n_distinct / 10)
                bxplts = bxplts + guides(fill=guide_legend(ncol=ncol))
        if color is not None:
            n_distinct = frame.select(variable_name).unique().height
            if n_distinct > 10:
                ncol = ceil(n_distinct / 10)
                bxplts = bxplts + guides(color=guide_legend(ncol=ncol))

        # handle the layers
        if layers is not None:
            if not isinstance(layers, Iterable):
                layers = [layers]
            for layer in list(layers):
                bxplts += layer

    # handle interactive
    if interactive:
        bxplts += ggtb()

    return bxplts
