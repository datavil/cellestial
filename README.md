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
umap = cl.umap(data, size=1, axis_type="arrow")+ggsize(700,500)
umap
```
<img src="./assets/tooltips.png" alt="Customized" width="700">


### Plots are exteremly customizable

```python
umap + scale_color_hue() + ggsize(500,400)
```
<img src="./assets/customized.png" alt="Customized" width="500">


### Multi plots are different function

Instead of singular function names (`umap`), multi-grid plots requires the plural (`umaps`),providing predictability which guarentees the reproducibility.

Which are valid for all `dimensional` subsets (`expression`,`pca`,`umap`, `tsne`).



