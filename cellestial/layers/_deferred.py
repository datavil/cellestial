from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from ggrust.plot.core import FeatureSpec, FeatureSpecArray, LayerSpec, PlotSpec


class DeferredLayer:
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
    write `plot + cl.layer()`. ggrust's `PlotSpec.__add__` returns
    `NotImplemented` for operands it does not know, which hands them to this
    `__radd__`.
    """

    __slots__ = ("_builder",)

    def __init__(
        self,
        builder: Callable[[PlotSpec], LayerSpec | FeatureSpec | FeatureSpecArray],
    ) -> None:
        self._builder = builder

    def __radd__(self, plot: PlotSpec) -> PlotSpec:
        return plot + self._builder(plot)
