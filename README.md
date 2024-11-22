<p align="center">
    <img src="https://github.com/datavil/cellestial/blob/master/assets/cellestial_logo.png?raw=true" alt="Cellestial Logo" width="250">
</p>

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/DataVil/Cellestial) [![PyPI](https://img.shields.io/pypi/v/cellestial?color=blue)](https://pypi.org/project/cellestial/)

# Cellestial

An Interactive and Highly Customizable __Single-Cell__ & __Spatial__ Plotting Tool over a ggplot-like API.

Name Encuplates: Space (of Spatial), Scatters of Stars, and of course Cells.

## Installation

```bash
pip install cellestial
```

## Usage

```python
import cellestial as cl
```

Interactive tooltips of individual data points
```python
umap = cl.umap(data, size=1, axis_type="arrow")
umap
```
<img src="./assets/tooltips.png" alt="tooltips" width="500">

or with Zooming and Paning Options
```python
umap = cl.umap(data, size=1, axis_type="arrow", interactive=True)
```

<img src="./assets/interactive.gif" width="500" />


### Plots are exteremly customizable

```python
umap + scale_color_hue() + ggsize(500,400)
```
<img src="./assets/customized.png" alt="Customized" width="400">


### Multi plots are distinct functions

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