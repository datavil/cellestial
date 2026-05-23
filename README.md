# Cellestial

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/DataVil/Cellestial)
[![PyPI](https://img.shields.io/pypi/v/cellestial?color=377eb8)](https://pypi.org/project/cellestial/)
[![codecov](https://codecov.io/gh/datavil/cellestial/branch/master/graph/badge.svg)](https://codecov.io/gh/datavil/cellestial)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-ff0000)](https://opensource.org/licenses/Apache-2.0)
[![Powered by Polars](https://img.shields.io/badge/Powered%20by-Polars-377eb8?logo=polars&logoColor=white)](https://www.pola.rs/)
[![Graphics: Lets-Plot](https://img.shields.io/badge/Graphics-Lets--Plot-FF00CC?logo=jetbrains&logoColor=white)](https://lets-plot.org/)

<img align="left" src="https://raw.githubusercontent.com/datavil/cellestial/master/assets/cellestial.anim.svg" alt="Cellestial Logo" width="280">

The __Grammar of Graphics__ for single-cell omics.

__Cellestial__ is an interactive and a highly customizable _Single-Cell_ & _Spatial_ omics data visualization library. Built on top [Lets-Plot](https://lets-plot.org/), it offers a ggplot-like layered and modular approach offering __high customizability__ and __publication-ready figures__.

Cellestial is highly integrated with scverse's [AnnData](https://github.com/scverse/anndata) with room for integration with any upcoming single-cell omics data type in the Python single-cell omics ecosystem.

Cellestial leverages the performance of [Polars](https://pola.rs/) ensuring speed and scalability.


## Installation

```bash
pip install cellestial
```

```bash
uv add cellestial
```

```bash
poetry add cellestial
```

## Quickstart

Cellestial accepts a standard `AnnData` directly. The same function plots a categorical column or a gene, and everything composes through `+` like ggplot.

```python
from lets_plot import *
import cellestial as cl

data = cl.datasets.pbmc3k(cache_directory="data")

cl.umap(
    data,
    key="cell_type_lvl1",
    axis_type="arrow",
    legend_ondata=True,
) + scale_color_hue()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/datavil/cellestial/master/assets/first-umap.svg" alt="UMAP coloured by cell type" width="600">
</p>

## Examples

### Spatial omics

Overlay categorical labels or gene expression on tissue coordinates.

<details>
<summary><b>Show code</b></summary>

```python
import squidpy as sq

lymph_node = cl.datasets.human_lymph_node(cache_directory="data")
hne = sq.datasets.visium_hne_adata()

gggrid(
    [
        cl.spatial(lymph_node, key="clusters"),
        cl.spatial(hne, key="leiden"),
    ],
    ncol=2,
) + ggsize(1000, 400)
```

</details>

<p align="center">
  <img src="https://raw.githubusercontent.com/datavil/cellestial/master/assets/spatial-categorical.svg" alt="Spatial categorical overlay" width="800">
</p>

### Dimensionality reduction with layers

Cellestial ships single-cell-specific layers (`cluster_outlines`, `stream`, `arrow_axis`, `ondata_legend`) that compose with `+` like any geom.

<details>
<summary><b>Show code</b></summary>

```python
velocity_data = cl.datasets.pancreas(cache_directory="data")

outlined = cl.umap(
    data,
    key="cell_type_lvl1",
    axis_type="arrow",
    size=1.5,
    legend_ondata=True,
) + scale_color_hue() + cl.cluster_outlines(groups=["Lymphocytes", "B Cells"])

streamed = cl.umap(
    velocity_data,
    key="clusters_coarse",
    axis_type="arrow",
    size=4,
    alpha=0.4,
    legend_ondata=True,
    ondata_color="black",
) + cl.stream()

gggrid([outlined, streamed])
```

</details>

<p align="center">
  <img src="https://raw.githubusercontent.com/datavil/cellestial/master/assets/dimensionality-layers.svg" alt="UMAP with cluster outlines and velocity stream" width="800">
</p>

### Marker genes

Heatmaps, dotplots, matrixplots and stacked violins share the same call shape and ship with built-in dendrograms and group bars.

<details>
<summary><b>Show code</b></summary>

```python
markers = [
    "PSAP", "LYZ", "CST3",          # Monocytes
    "CD79A", "CD79B",                # B cells
    "IL7R", "CD3D", "CD3E", "CD4",   # T cells (CD4+)
    "CD8A", "CD8B",                  # T cells (CD8+)
    "NKG7", "GNLY", "KLRD1",         # NK cells
    "HLA-DRA", "FCER1A",             # Dendritic cells
]

cl.heatmap(
    data,
    group_by="cell_type_lvl1",
    keys=markers,
    geom="raster",
    group_lines_size=0.5,
    group_lines_color="white",
    group_bars=True,
    group_bars_labels=True,
    dendrogram=True,
    dendrogram_size=1,
) + scale_fill_viridis()
```

</details>

<p align="center">
  <img src="https://raw.githubusercontent.com/datavil/cellestial/master/assets/heatmap.svg" alt="Marker gene heatmap" width="700">
</p>

### Statistical comparisons

`cl.bracket` runs pairwise tests on a `cl.boxplot` or `cl.violin` and draws annotated significance brackets.

<details>
<summary><b>Show code</b></summary>

```python
cl.boxplot(
    data,
    key="CD3D",
    fill="cell_type_lvl1",
    threshold=0.1,
) + scale_fill_hue() + cl.bracket(
    y_padding=0.2,
    label="pvalue",
    prefix="p",
    prefix_style="<",
    comparisons=[
        ("Lymphocytes", "Monocytes"),
        ("Monocytes", "Erythroid"),
        ("Monocytes", "B Cells"),
    ],
)
```

</details>

<p align="center">
  <img src="https://raw.githubusercontent.com/datavil/cellestial/master/assets/boxplot-bracket.svg" alt="Boxplot with significance brackets" width="600">
</p>

### Ridge plots

Quick exploratory views of expression distributions across groups.

<details>
<summary><b>Show code</b></summary>

```python
cl.ridge(
    data,
    key="B2M",
    alpha=0.6,
    group_by="cell_type_lvl1",
) + scale_fill_hue()
```

</details>

<p align="center">
  <img src="https://raw.githubusercontent.com/datavil/cellestial/master/assets/ridge.svg" alt="Ridge plot of B2M expression" width="600">
</p>

## Migrating from Scanpy

Cellestial mirrors most of `scanpy.pl` with a few naming shifts:

- `color=` becomes `key=`.
- Multi-panel calls use a plural function (`cl.umaps`, `cl.violins`, ...).
- `groupby` becomes `group_by`, `var_names` becomes `keys`.
- Saving is a separate `cl.save(plot, "umap.png")` call.

See the [migration guide](https://cellestial.datavil.org/migrating_from_scanpy.html) for a side-by-side mapping.

## Documentation

- [Quickstart](https://cellestial.datavil.org/quickstart.html) — guided tour of the API.
- [Features](https://cellestial.datavil.org/features.html) — tooltips, layers, grids, colors, faceting.
- [API reference](https://cellestial.datavil.org/API.html) — every function.
- [Migrating from Scanpy](https://cellestial.datavil.org/migrating_from_scanpy.html).

## License

Apache 2.0. See [LICENSE](LICENSE).
