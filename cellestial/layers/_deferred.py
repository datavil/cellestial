from __future__ import annotations

from typing import TYPE_CHECKING

from lets_plot.plot.core import FeatureSpec, PlotSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from lets_plot.plot.core import FeatureSpecArray, LayerSpec


class DeferredLayer(PlotSpec):
    """
    A layer whose construction depends on an existing plot's data and aesthetics.

    Parameters
    ----------
    builder : Callable[[PlotSpec], LayerSpec | FeatureSpec | FeatureSpecArray]
        Callable that accepts the receiving plot and returns the concrete layer
        (or a feature array) to add to it.

    Notes
    -----
    Built by cellestial layer functions that need to introspect the plot they are
    added to. Rather than taking the plot explicitly (`plot + cl.layer(plot)`),
    the real layer is assembled at `+` time via :meth:`__radd__`, so callers
    write `plot + cl.layer()`.

    Why subclass `PlotSpec`? Python's reflected operator priority rule says
    that when the right operand's class is a subclass of the left operand's
    class, `b.__radd__(a)` is tried *before* `a.__add__(b)`. Lets-Plot's
    `PlotSpec.__add__` raises `TypeError` for unknown types instead of
    returning `NotImplemented`, so the usual `__radd__` fallback would
    never fire, subclassing restores that fallback via the priority rule.
    The base `__init__` is deliberately bypassed because `DeferredLayer`
    does not carry real plot data; only :meth:`__radd__` is ever used.
    """

    __slots__ = ("_builder",)

    def __init__(
        self,
        builder: Callable[[PlotSpec], LayerSpec | FeatureSpec | FeatureSpecArray],
    ) -> None:
        # bypass `PlotSpec.__init__`, this wrapper does not carry plot data.
        FeatureSpec.__init__(self, kind="deferred", name=None)
        self._builder = builder

    def __radd__(self, plot: PlotSpec) -> PlotSpec:
        return plot + self._builder(plot)
