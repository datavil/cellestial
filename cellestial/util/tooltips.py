from __future__ import annotations

from collections.abc import Sequence
from lets_plot import layer_tooltips
from cellestial.util.utilities import _select_variable_keys



def _decide_tooltips(
    base_tooltips: Sequence[str] | str | None,
    add_tooltips: Sequence[str] | str | None,
    custom_tooltips: Sequence[str] | str | None,
    *,
    show_tooltips: bool,
) -> list[str]:
    """
    Decide on the tooltips.

    Parameters
    ----------
    base_tooltips : list[str] | str
        Base tooltips, default ones by the function.
    add_tooltips : list[str] | str
        Additional tooltips, will be appended to the base_tooltips.
    custom_tooltips : list[str] | str
        Custom tooltips, will overwrite the base_tooltips.
    show_tooltips : bool
        Whether to show tooltips at all.
        Set tooltip to the Literal 'none' if False.

    Returns
    -------
    list[str]
        Tooltips.
    """
    # PART 1: CONVERT str TO list
    if isinstance(base_tooltips, str):
        base_tooltips = [base_tooltips]
    if isinstance(add_tooltips, str):
        add_tooltips = [add_tooltips]
    if isinstance(custom_tooltips, str):
        custom_tooltips = [custom_tooltips]

    # PART 2: HANDLE TOOLTIP LOGIC
    if not show_tooltips:
        tooltips = "none"  # for letsplot, this removes the tooltips
    else:
        if isinstance(custom_tooltips, Sequence):
            tooltips = list(custom_tooltips)
        elif isinstance(add_tooltips, Sequence):
            tooltips = list(base_tooltips) + list(add_tooltips)
        else:
            tooltips = list(base_tooltips)

    return tooltips

def _build_tooltips(tooltips:layer_tooltips|list[str]) ->layer_tooltips:
    pass
    #TODO
