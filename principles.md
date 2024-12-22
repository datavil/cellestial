# Principles

## Naming

### Plotting Functions

This is already a plotting library, Function names should not include "plot" suffix/prefix

e.g., instead of `plot_scatter` use `scatter`,
e,g., instead of `plot_violin` use `violin`

## Explicit over Implicit

Function names, variable names including the constants should have short but explicit and descriptive names.

e.g., instead of `vlnplot` use `violin`


### Multi vs Single Plotting Names

in the case of multi-plots, functions should change to plural form. e.g., `scatter` to `scatters`

## Generalized vs Specific Plots

For example `umap` plot is a subset of `dimension` plot. 


## Abstractions

Abstractions should be done **modarately**. they come with cost in the form of **reduction** in *customizability* and *flexibility*.

Plots already return customizable plot objects, So layers can be added later and existing layers can be customized by the user.

Rather than taking everything as an argument, accept only non-customizable parts of the plot.
--- Expections can be made for user very common layers (e.g., the geom size, color, etc.)

e.g., do not take plot size as it can later be changed by the user with `+(ggsize)`.

1. It limits options.
2. Users can always change/add the layer themselves.

### Multi-Plot Abstractions

For multi-plots though, layers (e.g `scale_color/fill_*`) can be taken as `layers: list | iterable`. 

Since, gggrid is not customizable in terms of color.

multi-plots should take keyword arguments for the ggrid as the standard keyword arguments.

`kwargs` should be reserved for geom keyword arguments.
e.g., `**violin_kwargs`


## General Syntax

Python conventions should be followed.

### Styling and Formatting

Use __Ruff__ for formatting and linting and sorting imports.

### No need for classes

This project already uses `lets_plot` objects, there is no need for additional classes.


