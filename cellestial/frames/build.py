from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
from anndata import AnnData
from scipy.sparse import issparse

from cellestial.util.errors import KeyNotFoundError

if TYPE_CHECKING:
    from collections.abc import Sequence


def _add_anndata_variable_columns(
    data: AnnData, frame: pl.DataFrame, keys: str | Sequence[str]
) -> pl.DataFrame:
    """Add variable keys to the DataFrame."""
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if key in frame.columns:
            continue
        elif key in data.var_names:
            # get the index of the variable
            index = data.var_names.get_loc(key)
            # handle sparse matrix
            if issparse(data.X):  # sparse matrix
                column = data.X[:, index].toarray()
            else:  # numpy array
                column = data.X[:, index]

            # add the variable to the frame
            frame = frame.with_columns(
                pl.Series(key, column.flatten().astype("float32")),
            )
        else:
            msg = f"Key `{key}` not found in data."
            raise KeyNotFoundError(msg)

    return frame


def anndata_observations_frame(
    data: AnnData,
    /,
    variable_keys: str | Sequence[str] | None = None,
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
    variable_keys : str | Sequence[str] | None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    observations_name : str, optional
        The name of the observations column, default is "barcode".
    include_dimensions : bool, optional
        Whether to include dimensions from `obsm` in the DataFrame, default is False.

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
    # PART 2: ADD obs_names
    frame = frame.with_columns(pl.Series(observations_name, data.obs_names))
    # PART 3: ADD AnnData.obs
    for key in data.obs.columns:
        # handle categorical integer data
        if data.obs[key].dtype == "category":
            # Check if the categories are numeric (integer 'i','u' or float 'f' kinds)
            category_kind = data.obs[key].cat.categories.dtype.kind
            # Only convert if the category dtype is numeric ('i', 'u', 'f')
            if category_kind in "iuf":
                # Convert to string (str) and then back to categorical
                data.obs[key] = data.obs[key].astype(str).astype("category")
        # add the columns
        frame = frame.with_columns(pl.Series(key, data.obs[key]))
    # PART 4: ADD dimensions if needed
    if include_dimensions:
        for X in data.obsm:
            col_count = data.obsm[X].shape[1]  # Number of dimensions (columns)
            for col in range(col_count):
                frame = frame.with_columns(pl.Series(f"{X.upper()}{col+1}", data.obsm[X][:, col]))

    # PART 5: ADD keys if provided
    if variable_keys is not None:
        frame = _add_anndata_variable_columns(data=data, frame=frame, keys=variable_keys)

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
    frame = frame.with_columns(pl.Series(variables_name, data.var_names))

    # PART 3: ADD AnnData.var
    for key in data.var.columns:
        # handle categorical integer data
        if data.var[key].dtype == "category":
            # Check if the categories are numeric (integer 'i','u' or float 'f' kinds)
            category_kind = data.var[key].cat.categories.dtype.kind
            # Only convert if the category dtype is numeric ('i', 'u', 'f')
            if category_kind in "iuf":
                # Convert to string (str) and then back to categorical
                data.var[key] = data.var[key].astype(str).astype("category")
        # add the columns
        frame = frame.with_columns(pl.Series(key, data.var[key]))

    # PART 4: ADD dimensions if needed
    if include_dimensions:
        for X in data.varm:
            col_count = data.varm[X].shape[1]  # Number of dimensions (columns)
            for col in range(col_count):
                frame = frame.with_columns(pl.Series(f"{X.upper()}{col+1}", data.varm[X][:, col]))

    return frame


def build_frame(
    data: AnnData,
    *,
    variable_keys: str | Sequence[str] | None = None,
    axis: Literal[0, 1] | None = None,
    observations_name: str = "barcode",
    variables_name: str = "variable",
    include_dimensions: bool = False,
) -> pl.DataFrame:
    """
    Build a DataFrame from an AnnData object.

    Parameters
    ----------
    data : AnnData
        The AnnData object containing the variables.
    variable_keys : str | Sequence[str] | None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    axis : Literal[0,1] | None
        The axis to build the frame for. 0 for observations, 1 for variables.
    observations_name : str
        The name of the observations column, default is "barcode".
    variables_name : str
        Name for the variables index column, default is 'variable'
    include_dimensions : bool
        Whether to include dimensionality reductions fields.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the variables.
    """
    if isinstance(data, AnnData):
        # infer the axis if not provided
        if axis is None and variable_keys is not None:
            axis = 0

        if axis == 0:
            frame = anndata_observations_frame(
                data,
                variable_keys=variable_keys,
                observations_name=observations_name,
                include_dimensions=include_dimensions,
            )
        elif axis == 1:
            frame = anndata_variables_frame(
                data,
                variables_name=variables_name,
                include_dimensions=include_dimensions,
            )
        elif axis is None:
            msg = "`axis` parameter must be specified, 0 for observations, 1 for variables."
            raise ValueError(msg)
    else:
        msg = f"Unsupported data type: `{type(data)}`"
        raise TypeError(msg)

    return frame
