# Cellestial
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/DataVil/Cellestial) 
[![PyPI](https://img.shields.io/pypi/v/cellestial?color=blue)](https://pypi.org/project/cellestial/)

<img align="right" src="assets/cellestial_logo.png" alt="Cellestial Logo" width="300">

The __Grammar of Graphics__ for single-cell omics.


### Installation

```bash
pip install cellestial
```

### Introduction


__Cellestial__ is an interactive and a highly customizable _Single-Cell_ & _Spatial_ omics data visualization library. Built on top [Lets-Plot](https://lets-plot.org/), it offers a ggplot-like layered and modular approach offering __high customizability__ and __publication-ready figures__. 

Cellestial is highly integrated with scverse's [AnnData](https://github.com/scverse/anndata) with room for integration with any upcoming single-cell omics data type in the Python single-cell omics ecosystem.

Cellestial leverages the performance of [Polars](https://pola.rs/) ensuring speed and scalability.

_Cellestial is built-on top the following core design philosophies:_
1. Modularity over abstraction
1. Predictability over flexibility
1. Explicity over implicity
1. Simplicity & Expressiveness


with _reproduciblity_, _intiutiveness_ & _ease-of-use_, and, of course, _beautiful publication-ready plots/charts/figures_ in mind.

[__More on Design Philosophy__](./philosophy.md)


To see the example Figures visit [Cellestial Webpage](https://datavil.github.io/cellestial-examples/).

### Example Plots Grid

<img src="./assets/overall.png" alt="multipanel" width="750">

### Usage

```python
import cellestial as cl
```

#### Interactive tooltips of individual data points
```python
umap = cl.umap(data, size=1, axis_type="arrow")
umap
```
<img src="./assets/tooltips.png" alt="tooltips" width="500">

and tooltips can be extended with other features..

#### Zooming and Paning Options
```python
umap = cl.umap(data, size=1, axis_type="arrow", interactive=True)
```

<img src="./assets/interactive.gif" width="500" />


#### Plots are exteremly customizable

```python
umap + scale_color_hue() + ggsize(500,400)
```
<img src="./assets/customized.png" alt="Customized" width="400">


#### Multi plots are distinct functions

Instead of singular function names (`umap`), multi-grid plots requires the plural (`umaps`),providing predictability which guarentees the reproducibility.

Which are valid for all `dimensional` subsets (`expression`,`pca`,`umap`, `tsne`).

```python
cl.umaps(
    data,
    keys=["leiden", "HBD", "NEAT1", "IGKC"],
    ncol=2,
    size=1,
    color_high="red",
) + ggsize(900, 600)
```
<img src="./assets/multi_umap.png" alt="multi" width="700">