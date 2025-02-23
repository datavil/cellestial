from __future__ import annotations

from typing import Iterable

import polars as pl
from anndata import AnnData

from cellestial.util.errors import ConflictingKeysError, KeyNotFoundError


def _expand_frame(data: AnnData, frame: pl.DataFrame, to_add: list[str]) -> pl.DataFrame:
    """
    frame already has dimensions, cellID and key (or cluster).

    expand the frame with:
    - given tooltips
       - it can be in obs
           - check if the key is in obs
       - it can be a gene expression level
           - check if the key is in var_names
    add the columns to frame
    return the frame
    """
    for key in to_add:
        if key not in frame.columns:
            if key in data.obs.columns:
                frame = frame.with_columns(pl.Series(key, data.obs[key]))
            elif key in data.var_names:
                index = data.var_names.get_indexer([key])  # get the index of the gene
                frame = frame.with_columns(
                    pl.Series(key, data.X[:, index].flatten().astype("float32")),
                )
            else:
                msg = f"key '{key}' to expand is not present in `observation (.obs) names` nor `gene (.var) names`"
                raise ValueError(msg)
    return frame


def _check_key_conflicts(data: AnnData, keys: Iterable[str]) -> None:
    """Check if there are any keys conflicts in the data."""
    keys_from = []
    if isinstance(data, AnnData):
        for key in keys:
            if key in data.obs.columns and "obs" not in keys_from:
                keys_from.append("obs")
            elif key in data.var_names and "var_names" not in keys_from:
                keys_from.append("var_names")
            elif key in data.var.columns and "var" not in keys_from:
                keys_from.append("var")
            else:
                msg = f"key '{key}' not found in the data"
                raise KeyNotFoundError(msg)
        # conflicting scenarios
        if "var" in keys_from:
            if "var_names" in keys_from or "obs" in keys_from:
                msg = "keys from var and var_names or obs cannot be used together"
                raise ConflictingKeysError(msg)

    return


def _construct_frame(data: AnnData, keys: Iterable[str], dimension: str, xy = (1,2)) -> pl.DataFrame:
    """
    Construct a polars DataFrame from data.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to construct the DataFrame from.
    keys : Iterable[str]
        The keys to include in the DataFrame.

    Returns
    -------
    pl.DataFrame
        The constructed DataFrame.
    """
    if not isinstance(data, (AnnData)):
        msg = "data must be an AnnData object"
        raise TypeError(msg)
    # check if there are any keys conflicts
    _check_key_conflicts(data, keys)  # raises an error if there are conflicts
    # initialize the frame
    frame = pl.DataFrame()
    # there could be other types in the future
    if isinstance(data, AnnData):
        for key in keys:
            if key in data.obs.columns:
                frame = frame.with_columns(pl.Series(key, data.obs[key]))
            elif key in data.var_names:
                index = data.var_names.get_indexer([key])  # get the index of the gene
                frame = frame.with_columns(
                    pl.Series(key, data.X[:, index].flatten().astype("float32")),
                )
            elif key in data.var.columns:
                frame = frame.with_columns(pl.Series(key, data.var[key]))
            else:
                msg = f"key '{key}' not found in the data"
                raise KeyNotFoundError(msg)

    return frame
