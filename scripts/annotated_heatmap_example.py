"""
Annotated heatmap example assembled with `gggrid`.

Reproduces the common publication layout: a main expression heatmap with
column annotations (cell type, time point) above it and a row annotation
(gene module) to its right. All data is synthetic.

Run:
    poetry run python scripts/annotated_heatmap_example.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
from lets_plot import (
    LetsPlot,
    aes,
    geom_segment,
    geom_tile,
    gggrid,
    ggplot,
    ggsize,
    scale_fill_gradient,
    scale_fill_gradientn,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_void,
)

LetsPlot.setup_html()

rng = np.random.default_rng(0)

# --- synthetic experiment design -----------------------------------------
cell_types = ["Progenitor", "Neuron", "Astrocyte", "Microglia", "Oligo"]
cell_type_colors = {
    "Progenitor": "#7e1e9c",
    "Neuron": "#e6432f",
    "Astrocyte": "#f4c20d",
    "Microglia": "#f08c1d",
    "Oligo": "#2a5fb0",
}
cells_per_type = 16
n_cells = len(cell_types) * cells_per_type

# Each gene module is up-regulated in one cell type -> block structure.
modules_per_type = 2
genes_per_module = 9
n_modules = len(cell_types) * modules_per_type
n_genes = n_modules * genes_per_module

# Row annotation: an independent two-state lineage tag per gene.
lineage_colors = {"Early": "#4aa564", "Late": "#e8743b", "None": "#e9e9e9"}


def _column_design() -> pl.DataFrame:
    """Build the per-cell (column) annotation table."""
    cell_index = np.arange(n_cells)
    type_of_cell = np.repeat(cell_types, cells_per_type)
    # Time point ramps within each cell-type block (grayscale strip).
    time_point = np.tile(np.linspace(0, 1, cells_per_type), len(cell_types))
    return pl.DataFrame(
        {
            "cell": cell_index,
            "cell_type": type_of_cell,
            "time_point": time_point,
        }
    )


def _row_design() -> pl.DataFrame:
    """Build the per-gene (row) annotation table."""
    gene_index = np.arange(n_genes)
    module_index = np.repeat(np.arange(n_modules), genes_per_module)
    high_in = np.repeat(np.repeat(cell_types, modules_per_type), genes_per_module)
    lineage = rng.choice(list(lineage_colors), size=n_genes, p=[0.45, 0.45, 0.10])
    return pl.DataFrame(
        {
            "gene": gene_index,
            "module": module_index,
            "high_in": high_in,
            "lineage": lineage,
        }
    )


def _expression(columns: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
    """Synthesize the long-form expression matrix with block structure."""
    high_in = rows["high_in"].to_numpy()
    cell_type = columns["cell_type"].to_numpy()
    # base noise + a bump where the gene's target cell type matches the column.
    base = rng.normal(0.15, 0.08, size=(n_genes, n_cells))
    match = high_in[:, None] == cell_type[None, :]
    base += match * rng.normal(0.85, 0.12, size=(n_genes, n_cells))
    base = base.clip(0, None)
    frame = pl.DataFrame(base, schema=[str(cell) for cell in range(n_cells)]).with_columns(
        gene=pl.Series(rows["gene"])
    )
    return frame.unpivot(index="gene", variable_name="cell", value_name="value").with_columns(
        pl.col("cell").cast(pl.Int64)
    )


columns = _column_design().with_columns(strip=pl.lit(0))
rows = _row_design().with_columns(strip=pl.lit(0))
expression = _expression(columns, rows)

# --- block separator lines -----------------------------------------------
# Vertical lines at cell-type boundaries, horizontal lines at module boundaries.
column_boundaries = np.arange(cells_per_type, n_cells, cells_per_type) - 0.5
row_boundaries = np.arange(genes_per_module, n_genes, genes_per_module) - 0.5
vlines = pl.DataFrame({"x": column_boundaries})
hlines = pl.DataFrame({"y": row_boundaries})

# Tiles fill panels edge to edge so the strips line up with the heatmap.
_FLUSH = (
    scale_x_continuous(expand=[0, 0])
    + scale_y_continuous(expand=[0, 0])
    + theme_void()
    + theme(legend_position="none")
)


def _heatmap() -> ggplot:
    return (
        ggplot(expression, aes("cell", "gene"))
        + geom_tile(aes(fill="value"))
        + scale_fill_gradientn(
            colors=["#1f3b73", "#2c6e9c", "#3f9aa8", "#73c9b0"],
            guide="none",
        )
        + geom_segment(
            aes(x="x", xend="x"),
            y=-0.5,
            yend=n_genes - 0.5,
            data=vlines,
            color="black",
            size=0.5,
        )
        + geom_segment(
            aes(y="y", yend="y"),
            x=-0.5,
            xend=n_cells - 0.5,
            data=hlines,
            color="black",
            size=0.5,
        )
        + _FLUSH
    )


def _cell_type_strip() -> ggplot:
    return (
        ggplot(columns, aes("cell", "strip"))
        + geom_tile(aes(fill="cell_type"))
        + scale_fill_manual(values=cell_type_colors)
        + _FLUSH
    )


def _time_point_strip() -> ggplot:
    return (
        ggplot(columns, aes("cell", "strip"))
        + geom_tile(aes(fill="time_point"))
        + scale_fill_gradient(low="#ffffff", high="#1a1a1a")
        + _FLUSH
    )


def _row_strip() -> ggplot:
    return (
        ggplot(rows, aes("strip", "gene"))
        + geom_tile(aes(fill="lineage"))
        + scale_fill_manual(values=lineage_colors)
        + _FLUSH
    )


# --- compose -------------------------------------------------------------
# 3 rows x 2 columns; the heatmap and its annotations align on shared axes.
# `None` leaves the cell next to each top strip blank.
grid = gggrid(
    [
        _cell_type_strip(),
        None,
        _time_point_strip(),
        None,
        _heatmap(),
        _row_strip(),
    ],
    ncol=2,
    widths=[1.0, 0.05],
    heights=[0.04, 0.04, 1.0],
    hspace=2,
    vspace=2,
) + ggsize(420, 760)


if __name__ == "__main__":
    grid.to_png("figures/annotated_heatmap_example.png", scale=2)
    print("wrote figures/annotated_heatmap_example.png")
