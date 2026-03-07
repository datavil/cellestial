from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import polars as pl
from anndata import AnnData

from cellestial.util.errors import KeyNotFoundError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars import DataFrame


def _anndata_variable_columns(
    data: AnnData, column_names: list[str], keys: str | Sequence[str]
) -> list[pl.Series]:
    """Return a list of variable columns as Polars `Series`."""
    columns = []
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if key in column_names:  # check for repeats
            continue
        elif key in data.var_names:
            values = data.obs_vector(key)
            # add the variable column to the list of columns
            columns.append(pl.Series(key, values).cast(pl.Float32))
            column_names.append(key)
        else:
            msg = f"Key `{key}` not found in data."
            raise KeyNotFoundError(msg)

    return columns


def anndata_observations_frame(
    data: AnnData,
    /,
    variable_keys: str | Sequence[str] | None = None,
    *,
    observations_name="barcode",
    include_dimensions: bool | int = False,
) -> DataFrame:
    """
    Build an Observations DataFrame from an AnnData object.

    Parameters
    ----------
    data : AnnData
        The AnnData object containing the observations.
    variable_keys : str | Sequence[str] | None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    observations_name : str, optional
        The name of the observations column, default is 'barcode'.
    include_dimensions : bool | int
        Whether to include dimensions from `obsm` in the DataFrame, default is False.
        Providing an integer will limit the number of dimensions to given number.

    Returns
    -------
    DataFrame
        A DataFrame containing the observations, with optional variable keys and dimensions.
    """
    # Check if data is an AnnData object
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)
    if not isinstance(data.obs, pd.DataFrame):  # in case of Dataset2D
        part = data.obs.to_memory()
    else:
        part = data.obs
    partm = data.obsm
    # PART 1: INITIALIZE
    columns = [pl.Series(observations_name, data.obs_names)]
    # PART 2: ADD AnnData.obs
    for key in part.columns:
        # handle categorical integer data
        if part.dtypes[key] == "category" and part[key].cat.categories.dtype.kind in "iuf":
            # Check if the categories are numeric (integer 'i','u' or float 'f' kinds)
            # Only convert if the category dtype is numeric ('i', 'u', 'f')
            # Convert to string (str) and then back to categorical
            columns.append(pl.Series(part[key].astype(str)).cast(pl.Categorical))
        else:
            columns.append(pl.Series(part[key]))

    # PART 3: ADD dimensions if needed
    if include_dimensions:
        for X in partm:
            total_cols = partm[X].shape[1]  # Number of dimensions (columns)
            if isinstance(include_dimensions, int) and not isinstance(include_dimensions, bool):
                if include_dimensions >= 0:
                    col_count = min(include_dimensions, total_cols)
                else:
                    msg = "Number of dimensions cannot be a negative number."
                    raise ValueError(msg)
            elif isinstance(include_dimensions, bool):
                col_count = total_cols
            else:
                msg = "Argument for `include_dimensions` MUST be either a `bool` or an `int` type."
                msg += f" You provided type {type(include_dimensions)}"
                raise TypeError(msg)

            for col in range(col_count):
                columns.append(pl.Series(f"{X.upper()}{col + 1}", partm[X][:, col]))

    # PART 4: ADD keys if provided
    if variable_keys is not None:
        column_names = [column.name for column in columns]
        columns.extend(
            _anndata_variable_columns(data=data, column_names=column_names, keys=variable_keys)
        )

    return pl.DataFrame(columns)


def anndata_variables_frame(
    data: AnnData,
    *,
    variables_name: str = "variable",
    include_dimensions: bool | int = False,
) -> DataFrame:
    """
    Build a Variables DataFrame from an AnnData object.

    Parameters
    ----------
    data : AnnData
        The AnnData object containing the variables.
    variables_name : str
        Name for the variables index column, default is 'variable'
    include_dimensions : bool | int
        Whether to include dimensions from `varm` in the DataFrame, default is False.
        Providing an integer will limit the number of dimensions to given number.

    Returns
    -------
    DataFrame
        A DataFrame containing the variables.
    """
    # PART 1: INITIALIZE
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)
    if not isinstance(data.var, pd.DataFrame):  # in case of Dataset2D
        part = data.var.to_memory()
    else:
        part = data.var
    partm = data.varm
    # PART1: initalize columns
    columns = [pl.Series(variables_name, data.var_names)]
    # PART 3: ADD AnnData.var
    for key in part.columns:
        # handle categorical integer data
        if part.dtypes[key] == "category" and part[key].cat.categories.dtype.kind in "iuf":
            # Check if the categories are numeric (integer 'i','u' or float 'f' kinds)
            # Only convert if the category dtype is numeric ('i', 'u', 'f')
            # Convert to string (str) and then back to categorical
            columns.append(pl.Series(part[key].astype(str)).cast(pl.Categorical))
        else:
            columns.append(pl.Series(part[key]))

    # PART 4: ADD dimensions if needed
    if include_dimensions:
        for X in partm:
            total_cols = partm[X].shape[1]  # Number of dimensions (columns)
            if isinstance(include_dimensions, int) and not isinstance(include_dimensions, bool):
                if include_dimensions >= 0:
                    col_count = min(include_dimensions, total_cols)
                else:
                    msg = "Number of dimensions cannot be a negative number."
                    raise ValueError(msg)
            elif isinstance(include_dimensions, bool):
                col_count = total_cols
            else:
                msg = "Argument for `include_dimensions` MUST be either a `bool` or an `int` type."
                msg += f" You provided type {type(include_dimensions)}"
                raise TypeError(msg)

            for col in range(col_count):
                columns.append(pl.Series(f"{X.upper()}{col + 1}", partm[X][:, col]))

    return pl.DataFrame(columns)


def build_frame(
    data: AnnData,
    *,
    variable_keys: str | Sequence[str] | None = None,
    axis: Literal[0, 1] | None = None,
    observations_name: str = "barcode",
    variables_name: str = "variable",
    include_dimensions: bool | int = False,
) -> DataFrame:
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
        The name of the observations column, default is 'barcode'.
    variables_name : str
        Name for the variables index column, default is 'variable'
    include_dimensions : bool | int
        Whether to include dimensions in the DataFrame, default is False.
        Providing an integer will limit the number of dimensions to given number.


    Returns
    -------
    DataFrame
        A polars DataFrame containing the variables.
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
