.. _philosophy:

Design Philosophy
==================

These design philosophies are mainly borrowed from ggplot, Zen of Python, and Rust.

Modularity over abstraction
---------------------------

Abstraction is nice but sacrifices customizability. If the user can provide ``+ ggsize(800,600)`` and change the figure size at any point, there is no need to accept for plotting functions *width* or *height* as parameters. The same applies to the many other possible layers and aesthetics of the plot. This approach is particularly useful with notebooks and *exploratory data analysis*, EDA. On top of eliminating unnecessary complexity, the approach eliminates the risk of conflicting arguments.

Predictability over flexibility
-------------------------------

Flexibility provides convenience but sacrifices predictability. If providing a single key produces a single plot and providing a sequence of keys produces a grid of plots the return type would be different (i.e, ``PlotSpec`` vs ``SupPlotsSpec``) for the same function. The follow-up workflow would have to be totally different for these two cases.

Cellestial ensures reproducibility by strict return types. Naming convention allows such strict return types. Instead of ``cl.umap()`` accepting a sequence of keys (``Sequence[str]``) and returning a grid of plots (``SupPlotsSpec``), it only accepts a single key (``str``) it is guaranteed to return a single plot (``PlotSpec``).

Plural versions of such functions are available, if the user needs a grid of plots with provided keys. For instance, ``cl.umaps()`` requires a sequence of strings as keys and is guaranteed to return grid of plots (``SupPlotsSpec``).

In case the user wants to merge multiple violins/boxplot geoms on a single plot, the user can simply call the ``cl.violin()`` or ``cl.boxplot()`` with sequence of keys which strictly return ``PlotSpec``. However, if the user wants the grid with the same keys, the sequence can be provided to the plural versions of the same plot types i.e ``cl.violins()`` or ``cl.boxplots()`` which are guaranteed to return ``SupPlotsSpec``.

Explicitness over implicitness
------------------------------

Explicit function and parameter names allow the users to be fully aware and sure of what they are doing. While it is more convenient or easier to write ``vln()`` or ``inc_dims =`` instead of ``violin()`` or ``include_dimensions =``, the former is less intuitive and requires users to remember how the programmer chose to shorten the words.

Simplicity & Expressiveness
---------------------------

Cellestial is a data visualization library, therefore the plotting functions names do not have to include keywords such as *plot* in the names nor they have to be in different namespaces. Cellestial API is simple enough for most users yet expressive enough for users to know what exactly they are using.