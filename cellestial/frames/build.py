from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import polars as pl
from anndata import AnnData

from cellestial.util.errors import ConflictingKeysError, KeyNotFoundError


def _add_variable_columns(frame: pl.DataFrame, keys: str | Sequence[str]) -> pl.DataFrame:
    """Add variable keys to the DataFrame."""
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if key in frame.columns:
            continue
        elif key in data.var_names:
            # get the index of the gene
            index = data.var_names.get_indexer([key])
            # add the variable to the frame
            frame = frame.with_columns(
                pl.Series(key, data.X[:, index].flatten().astype("float32")),
            )
        else:
            msg = f"Key `{key}` not found in data."
            raise ValueError(msg)

    return frame

def anndata_observations_frame(
    data: AnnData,
    /,
    keys: str | Sequence[str] | None = None,
    *,
    observations_name="barcode",
    include_dimensions: bool = False,
) -> pl.DataFrame:
    """
    Build an Observations DataFrame from an AnnData object.

    Parameters
    ----------
    data : AnnData
        The AnnData object containing the observations.
    keys : str or Sequence[str] or None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    observations_name : str, optional
        The name of the observations column, by default "barcode".
    include_dimensions : bool, optional
        Whether to include dimensions from `obsm` in the DataFrame, by default False.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the observations, with optional variable keys and dimensions.
    """
    # Check if data is an AnnData object
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)
    # PART 1: INITIALIZE
    frame = pl.DataFrame()
    # PART 3: ADD obs_names
    frame = frame.with_columns(pl.Series("obs_names", data.obs_names))
    # PART 2: ADD AnnData.obs
    for key in data.obs.columns:
        frame = frame.with_columns(pl.Series(key, data.obs[key]))
    # PART 4: ADD dimensions if needed
    if include_dimensions:
        for X in data.obsm:
            col_count = data.obsm[X].shape[1]  # Number of dimensions (columns)
            for col in range(col_count):
                frame = frame.with_columns(pl.Series(f"{X}_{col+1}", data.obsm[X][:, col]))

    # PART 5: ADD keys if provided
    if keys is not None:
        frame = _add_variable_columns(frame, keys)

    return frame


def anndata_variables_frame(
    data: AnnData,
    *,
    variables_name: str = "variable",
    include_dimensions: bool = False,
) -> pl.DataFrame:
    """
    Build a Variables DataFrame from an AnnData object.

    Parameters
    ----------
    data : AnnData
        The AnnData object containing the variables.
    variables_name : str
        Name for the variables index column, default is 'variable'
    include_dimensions : bool
        Whether to include dimensionality reductions fields.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the variables.
    """
    # PART 1: INITIALIZE
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)
    frame = pl.DataFrame()

    # PART 2: ADD var_names
    frame = frame.with_columns(pl.Series("variable", data.var_names))

    # PART 3: ADD AnnData.var
    for key in data.var.columns:
        frame = frame.with_columns(pl.Series(key, data.var[key]))

    # PART 4: ADD dimensions if needed
    if include_dimensions:
        for X in data.varm:
            col_count = data.varm[X].shape[1] # Number of dimensions (columns)
            for col in range(col_count):
                frame = frame.with_columns(pl.Series(f"{X}_{col+1}", data.varm[X][:, col]))

    return frame