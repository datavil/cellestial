from __future__ import annotations

from collections.abc import Iterable
from math import ceil
from typing import TYPE_CHECKING, Any, Literal

# Core scverse libraries
import polars as pl

# Data retrieval
import scanpy as sc
from lets_plot import (
    LetsPlot,
    aes,
    geom_point,
    ggplot,
    ggtb,
    guide_legend,
    guides,
    labs,
    layer_tooltips,
    scale_color_brewer,
    scale_color_continuous,
)
from lets_plot.plot.core import PlotSpec
from scanpy import AnnData

from cellestial.themes import _THEME_DIMENSION
from cellestial.util import _add_arrow_axis

LetsPlot.setup_html()

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
    color_high: str = "#377eb8",
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
    # Handling Data tpyes
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
    if key in ["leiden", "louvain"]:
        base_tooltips = [barcode_name, cluster_name]
    else:
        base_tooltips = [barcode_name, key]
    if not show_tooltips:
        tooltips = "none"  # for letsplot, this removes the tooltips
    else:
        if isinstance(custom_tooltips, Iterable):
            tooltips = list(custom_tooltips)
        elif isinstance(add_tooltips, Iterable):
            tooltips = base_tooltips + list(add_tooltips)
        else:
            tooltips = base_tooltips


    # get the coordinates of the cells in the dimension reduced space
    # -------------------------- IF IT IS A CLUSTER --------------------------
    if key in ["leiden", "louvain"]:  # if it is a clustering
        # update the key column name if it is a cluster
        frame = frame.with_columns(
            pl.Series(barcode_name, data.obs_names), pl.Series(cluster_name, data.obs[key])
        )
        # cluster scatter
        scttr = (
            (
                ggplot(data=frame)
                + geom_point(
                    aes(x=f"{dimensions}1", y=f"{dimensions}2", color=cluster_name),
                    size=size,
                    tooltips=layer_tooltips(tooltips),
                    **point_kwargs,
                )
                + labs(
                    x=f"{dimensions}1".upper(), y=f"{dimensions}2".upper()
                )  # UMAP1 and UMAP2 rather than umap1 and umap2 etc.,
            )
            + _THEME_DIMENSION
            + scale_color_brewer(palette="Set2")
        )
        # wrap the legend
        n_distinct = frame.select(cluster_name).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            scttr = scttr + guides(color=guide_legend(ncol=ncol))

    # -------------------------- IF IT IS A GENE --------------------------
    elif key in data.var_names:  # if it is a gene
        # adata.X is a sparse matrix , axis0 is cells, axis1 is genes
        # find the index of the gene
        index = data.var_names.get_indexer(
            data.var_names[data.var_names.str.startswith(key)]
        )  # get the index of the gene
        frame = frame.with_columns(
            pl.Series(barcode_name, data.obs_names),
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
            + scale_color_continuous(low=color_low, high=color_high)
            + labs(
                x=f"{dimensions}1".upper(), y=f"{dimensions}2".upper()
            )  # UMAP1 and UMAP2 rather than umap1 and umap2 etc.,
        ) + _THEME_DIMENSION
    # -------------------------- NOT A GENE OR CLUSTER --------------------------
    else:
        msg = f"'{key}' is not present in `cluster names` nor `gene names`"
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
    color_high: str = "#377eb8",
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
    # Handling Data tpyes
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # only take the first two dimensions (pca comes with more dimensions)
    frame = pl.from_numpy(
        data.obsm[f"X_{dimensions}"][:, :2], schema=[f"{dimensions}1", f"{dimensions}2"]
    ).with_columns(pl.Series(barcode_name, data.obs_names))

    if cluster_type is not None:
        if cluster_type in ["leiden", "louvain"]:
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
    if not show_tooltips:
        tooltips = "none"  # for letsplot, this removes the tooltips
    else:
        if isinstance(custom_tooltips, Iterable):
            tooltips = list(custom_tooltips)
        elif isinstance(add_tooltips, Iterable):
            tooltips = base_tooltips + list(add_tooltips)
        else:
            tooltips = base_tooltips
    # get the coordinates of the cells in the dimension reduced space
    # -------------------------- IF IT IS A GENE --------------------------
    if gene in data.var_names:  # if it is a gene
        # adata.X is a sparse matrix, axis0 is cells, axis1 is genes
        # find the index of the gene
        index = data.var_names.get_indexer(
            data.var_names[data.var_names.str.startswith(gene)]
        )  # get the index of the gene
        frame = frame.with_columns(
            pl.Series(barcode_name, data.obs_names),
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
            + scale_color_continuous(low=color_low, high=color_high)
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


def test_dimension():
    import os
    from pathlib import Path

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
