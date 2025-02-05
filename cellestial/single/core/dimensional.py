from __future__ import annotations

from collections.abc import Iterable
from math import ceil
from typing import TYPE_CHECKING, Any, Literal

# Core scverse libraries
import polars as pl
from anndata import AnnData

# Data retrieval
from lets_plot import (
    aes,
    geom_point,
    ggplot,
    ggtb,
    guide_legend,
    guides,
    labs,
    layer_tooltips,
    scale_color_brewer,
)
from lets_plot.plot.core import PlotSpec

from cellestial.themes import _THEME_DIMENSION
from cellestial.util import _add_arrow_axis, _color_gradient, _decide_tooltips

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec


def dimensional(
    data: AnnData,
    key: Literal["leiden", "louvain"] | str = "leiden",
    *,
    dimensions: Literal["umap", "pca", "tsne"] = "umap",
    size: float = 0.8,
    interactive: bool = False,
    cluster_name: str = "Cluster",
    barcode_name: str = "CellID",
    color_low: str = "#e6e6e6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    mid_point: Literal["mean", "median", "mid"] | float = "median",
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    show_tooltips: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    **point_kwargs: dict[str, Any],
) -> PlotSpec:
    """
    Dimensional reduction plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : Literal['leiden', 'louvain'] | str, default='leiden'
        The key (cell feature) to color the points by.
        e.g., 'leiden' or 'louvain' to color by clusters or gene name for expression.
    dimensions : Literal['umap', 'pca', 'tsne'], default='umap'
        The dimensional reduction method to use.
        e.g., 'umap' or 'pca' or 'tsne'.
    size : float, default=0.8
        The size of the points.
    interactive : bool, default=False
        Whether to make the plot interactive.
    cluster_name : str, default='Cluster'
        The name to overwrite the clustering key in the dataframe and the plot.
    barcode_name : str, default='CellID'
        The name to give the barcode (or index) column in the dataframe.
    color_low : str, default='#e6e6e6'
        The color to use for the low end of the color gradient.
        Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
        Applies to continuous (non-categorical) data.
    color_mid : str, default=None
        The color to use for the mid part of the color gradient.
        If provided, color scale will have 3 colors.
        Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA (e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)').
        Applies to continuous (non-categorical) data.
    color_high : str, default='#377EB8'
        The color to use for the high end of the color gradient.
        Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA (e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)').
        Applies to continuous (non-categorical) data.
    mid_point : Literal["mean", "median", "mid"] | float, default="median"
        The midpoint (in data value) of the color gradient.
        Can be 'mean', 'median' and 'mid' or a number (float or int).
        If 'mean', the midpoint is the mean of the data.
        If 'median', the midpoint is the median of the data.
        If 'mid', the midpoint is the mean of 'min' and 'max' of the data.
    axis_type : Literal["axis", "arrow"] | None
        Whether to use regular axis or arrows as the axis.
    arrow_length : float, default=0.25
        Length of the arrow head (px).
    arrow_size : float, default=1
        Size of the arrow.
    arrow_color : str, default='#3f3f3f'
        Color of the arrow.
        Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA (e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)').
    arrow_angle : float, default=10
        Angle of the arrow head in degrees.
    show_tooltips : bool, default=True
        Whether to show tooltips.
    add_tooltips : list[str] | tuple[str] | Iterable[str] | None, default=None
        Additional tooltips, will be appended to the base_tooltips.
    custom_tooltips : list[str] | tuple[str] | Iterable[str] | None, default=None
        Custom tooltips, will overwrite the base_tooltips.
    **point_kwargs : dict[str, Any]
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensional reduction plot.

    Examples
    --------
    .. jupyter-execute::
        :linenos:
        :emphasize-lines: 10

        import scanpy as sc
        import cellestial as cl

        data = sc.read("data/pbmc3k_pped.h5ad")
        plot = cl.dimensional(data, key="leiden", dimensions="umap", size=0.6)
        plot

    |

    .. jupyter-execute::
        :linenos:
        :emphasize-lines: 10

        import scanpy as sc
        import cellestial as cl

        data = sc.read("data/pbmc3k_pped.h5ad")
        plot = cl.dimensional(data, key="leiden", dimensions="pca", size=0.8)
        plot

    """
    # Handling Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # only take the first two dimensions (pca comes with more dimensions)
    frame = pl.from_numpy(
        data.obsm[f"X_{dimensions}"][:, :2], schema=[f"{dimensions}1", f"{dimensions}2"]
    ).with_columns(pl.Series(barcode_name, data.obs_names))

    # handle point_kwargs
    if point_kwargs is None:
        point_kwargs = {}
    else:
        if "size" in point_kwargs:
            size = point_kwargs.get("size")
            msg = "use `size = value` instead of adding `'size' : 'value'` to `point_kwargs`\n"
            msg += f"size args will overwritten by the value `{size}` in the point_kwargs, "
            raise Warning(msg)
        if "tooltips" in point_kwargs:
            msg = "use tooltips args within the function instead of adding `'tooltips' : 'value'` to `point_kwargs`\n"
            raise KeyError(msg)

    # handle tooltips
    if key.startswith(("leiden", "louvain")):
        base_tooltips = [barcode_name, cluster_name]
    else:
        base_tooltips = [barcode_name, key]

    tooltips = _decide_tooltips(
        base_tooltips=base_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        show_tooltips=show_tooltips,
    )

    # get the coordinates of the cells in the dimension reduced space
    # -------------------------- IF IT IS A CELL ANNOTATION --------------------------
    if key in data.obs.columns:
        if key.startswith(("leiden", "louvain")):  # if it is a clustering
            # update the key column name if it is a cluster
            frame = frame.with_columns(pl.Series(cluster_name, data.obs[key]))
            color_key = cluster_name
        else:
            frame = frame.with_columns(pl.Series(data.obs[key]))
            color_key = key
        # cluster scatter
        scttr = (
            ggplot(data=frame)
            + geom_point(
                aes(x=f"{dimensions}1", y=f"{dimensions}2", color=color_key),
                size=size,
                tooltips=layer_tooltips(tooltips),
                **point_kwargs,
            )
            + labs(
                x=f"{dimensions}1".upper(), y=f"{dimensions}2".upper()
            )  # UMAP1 and UMAP2 rather than umap1 and umap2 etc.,
        ) + _THEME_DIMENSION
        # wrap the legend
        if frame.schema[color_key] == pl.Categorical:
            scttr += scale_color_brewer(palette="Set2")
            n_distinct = frame.select(color_key).unique().height
            if n_distinct > 10:
                ncol = ceil(n_distinct / 10)
                scttr = scttr + guides(color=guide_legend(ncol=ncol))
        else:
            scttr += _color_gradient(
                frame[color_key],
                color_low=color_low,
                color_mid=color_mid,
                color_high=color_high,
                mid_point=mid_point,
            )

    # -------------------------- IF IT IS A GENE --------------------------
    elif key in data.var_names:  # if it is a gene
        # adata.X is a sparse matrix , axis0 is cells, axis1 is genes
        # find the index of the gene
        index = data.var_names.get_indexer([key])  # get the index of the gene
        frame = frame.with_columns(
            pl.Series(key, data.X[:, index].flatten().astype("float32")),
        )
        scttr = (
            ggplot(data=frame)
            + geom_point(
                aes(x=f"{dimensions}1", y=f"{dimensions}2", color=key),
                size=size,
                tooltips=layer_tooltips(tooltips),
                **point_kwargs,
            )
            + _color_gradient(
                frame[key],
                color_low=color_low,
                color_mid=color_mid,
                color_high=color_high,
                mid_point=mid_point,
            )
            + labs(
                x=f"{dimensions}1".upper(), y=f"{dimensions}2".upper()
            )  # UMAP1 and UMAP2 rather than umap1 and umap2 etc.,
        ) + _THEME_DIMENSION
    # -------------------------- NOT A GENE OR CLUSTER --------------------------
    else:
        msg = f"'{key}' is not present in `observation (.obs) names` nor `gene (.var) names`"
        raise ValueError(msg)

    # special case for labels
    if dimensions == "tsne":
        scttr += labs(x="tSNE1", y="tSNE2")

    # handle arrow axis
    scttr += _add_arrow_axis(
        frame=frame,
        axis_type=axis_type,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        arrow_length=arrow_length,
        dimensions=dimensions,
    )

    # handle interactive
    if interactive:
        scttr += ggtb()

    return scttr

''' old implementation of expression function
def expression(
    data: AnnData,
    gene: str,
    *,
    dimensions: Literal["umap", "pca", "tsne"] = "umap",
    size: float = 0.8,
    interactive: bool = False,
    cluster_name: str = "Cluster",
    cluster_type: Literal["leiden", "louvain"] | None = None,
    barcode_name: str = "CellID",
    color_low: str = "#e6e6e6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    mid_point: Literal["mean", "median", "mid"] | float = "median",
    axis_type: Literal["axis", "arrow"] | None = "arrow",
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    show_tooltips: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    **point_kwargs: dict[str, Any],
) -> PlotSpec:
    # Handling Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # only take the first two dimensions (pca comes with more dimensions)
    frame = pl.from_numpy(
        data.obsm[f"X_{dimensions}"][:, :2], schema=[f"{dimensions}1", f"{dimensions}2"]
    ).with_columns(pl.Series(barcode_name, data.obs_names))

    if cluster_type is not None:
        if cluster_type.startswith(("leiden", "louvain")):
            frame = frame.with_columns(pl.Series(cluster_name, data.obs[cluster_type]))
        else:
            msg = f"'{cluster_type}' is not a valid cluster type"
            raise ValueError(msg)
    # handle point_kwargs
    if point_kwargs is None:
        point_kwargs = {}
    else:
        if "size" in point_kwargs:
            size = point_kwargs.get("size")
            msg = "use `size = value` instead of adding `'size' : 'value'` to `point_kwargs`\n"
            msg += f"size args will overwritten by the value `{size}` in the point_kwargs, "
            raise Warning(msg)
        if "tooltips" in point_kwargs:
            msg = "use tooltips args within the function instead of adding `'tooltips' : 'value'` to `point_kwargs`\n"
            raise KeyError(msg)
    # handle tooltips
    base_tooltips = [barcode_name, gene]
    tooltips = _decide_tooltips(
        base_tooltips=base_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        show_tooltips=show_tooltips,
    )
    # get the coordinates of the cells in the dimension reduced space
    # -------------------------- IF IT IS A GENE --------------------------
    if gene in data.var_names:  # if it is a gene
        # adata.X is a sparse matrix, axis0 is cells, axis1 is genes
        # find the index of the gene
        index = data.var_names.get_indexer([gene])  # get the index of the gene
        frame = frame.with_columns(
            pl.Series(gene, data.X[:, index].flatten().astype("float32")),
        )
        if cluster_type is not None:
            tooltips = [barcode_name, gene, cluster_name]
        else:
            tooltips = [barcode_name, gene]
        scttr = (
            ggplot(data=frame)
            + geom_point(
                aes(x=f"{dimensions}1", y=f"{dimensions}2", color=gene),
                size=size,
                tooltips=layer_tooltips(tooltips),
                **point_kwargs,
            )
            + _color_gradient(
                frame[gene],
                color_low=color_low,
                color_mid=color_mid,
                color_high=color_high,
                mid_point=mid_point,
            )
            + labs(
                x=f"{dimensions}1".upper(), y=f"{dimensions}2".upper()
            )  # UMAP1 and UMAP2 rather than umap1 and umap2 etc.,
        ) + _THEME_DIMENSION

    # -------------------------- NOT A GENE OR CLUSTER --------------------------
    else:
        msg = f"'{gene}' is not present in `gene names`"
        raise ValueError(msg)
    # special case for labels
    if dimensions == "tsne":
        scttr += labs(x="tSNE1", y="tSNE2")

    # handle arrow axis
    scttr += _add_arrow_axis(
        frame=frame,
        axis_type=axis_type,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        arrow_length=arrow_length,
        dimensions=dimensions,
    )

    # handle interactive
    if interactive:
        scttr += ggtb()

    return scttr
'''

def test_dimension():
    import os
    from pathlib import Path

    import scanpy as sc

    os.chdir(Path(__file__).parent.parent.parent.parent)  # to project root
    data = sc.read("data/pbmc3k_pped.h5ad")

    for ax in [None, "arrow", "axis"]:
        plot = dimensional(data, axis_type=ax)
        plot.to_html(f"plots/test_dim_umap_{ax}.html")
        plot.to_svg(f"plots/test_dim_umap_{ax}.svg")

    return


def test_expression():
    import os
    from pathlib import Path

    import scanpy as sc

    os.chdir(Path(__file__).parent.parent.parent.parent)  # to project root
    data = sc.read("data/pbmc3k_pped.h5ad")
    plot = expression(data, gene="MT-ND2", cluster_type="leiden").to_html(
        "plots/test_expression.html"
    )
    plot.to_html("plots/test_expression.svg")
    plot.to_svg("plots/test_expression.svg")


if __name__ == "__main__":
    test_dimension()
    test_expression()
