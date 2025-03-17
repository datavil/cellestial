from __future__ import annotations

from collections.abc import Iterable
from math import ceil
from typing import TYPE_CHECKING, Any

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    geom_boxplot,
    geom_jitter,
    geom_violin,
    gggrid,
    ggplot,
    ggsize,
    ggtb,
    guide_legend,
    guides,
    layer_tooltips,
)
from lets_plot.plot.core import FeatureSpec, LayerSpec, PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

from cellestial.frames import _construct_cell_frame
from cellestial.themes import _THEME_BOXPLOT, _THEME_VIOLIN
from cellestial.util import _build_tooltips, _decide_tooltips

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec

# TODO: add barcode_name to the arguments of the functions
# TODO: add the expansion of the frame
# TODO: pop already used arguments from dicts
# TODO: make layers accept a single layer


def violin(
    data: AnnData,
    key: str,
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
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    tooltips_title: str | None = None,
    interactive: bool = False,
    point_kwargs: dict[str, Any] | None = None,
    **violin_kwargs: dict[str, Any],
) -> PlotSpec:
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
    base_tooltips = [barcode_name, key]
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
    if tooltips == "none":
        if key is not None:
            all_keys.append(key)
    else:
        for tooltip in tooltips:
            if tooltip != barcode_name:
                all_keys.append(tooltip)

    frame = _construct_cell_frame(
        data=data,
        keys=all_keys,
        dimensions=None,
        xy=None,
        use_key=None,
        barcode_name=barcode_name,
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

    # handle interactive
    if interactive:
        vln += ggtb()

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
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    tooltips_title: str | None = None,
    interactive: bool = False,
    point_kwargs: dict[str, Any] | None = None,
    **boxplot_kwargs: dict[str, Any],
) -> PlotSpec:
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
    base_tooltips = [barcode_name, key]
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
    if tooltips == "none":
        if key is not None:
            all_keys.append(key)
    else:
        for tooltip in tooltips:
            if tooltip != barcode_name:
                all_keys.append(tooltip)

    frame = _construct_cell_frame(
        data=data,
        keys=all_keys,
        dimensions=None,
        xy=None,
        use_key=None,
        barcode_name=barcode_name,
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
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
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
        base_tooltips = [barcode_name]
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
            key=None,
            title=tooltips_title,
            clustering=False,
        )

        # construct the frame
        all_keys = list(keys)
        if tooltips == "none":
            if key is not None:
                all_keys.append(key)
        else:
            for tooltip in tooltips:
                if tooltip != barcode_name:
                    all_keys.append(tooltip)

        frame = _construct_cell_frame(
            data=data,
            keys=all_keys,
            dimensions=None,
            xy=None,
            use_key=None,
            barcode_name=barcode_name,
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
    trim: bool = False,
    barcode_name: str = "Barcode",
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
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
        base_tooltips = [barcode_name]
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
            key=None,
            title=tooltips_title,
            clustering=False,
        )

        # construct the frame
        all_keys = list(keys)
        if tooltips == "none":
            if key is not None:
                all_keys.append(key)
        else:
            for tooltip in tooltips:
                if tooltip != barcode_name:
                    all_keys.append(tooltip)

        frame = _construct_cell_frame(
            data=data,
            keys=all_keys,
            dimensions=None,
            xy=None,
            use_key=None,
            barcode_name=barcode_name,
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
                trim=trim,
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
            for layer in list(layers):
                bxplts += layer

    # handle interactive
    if interactive:
        bxplts += ggtb()

    return bxplts
