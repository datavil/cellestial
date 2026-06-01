from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from scipy.sparse import issparse

from cellestial.util.errors import UnsupportedDataTypeError, VariableNotFoundError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars import DataFrame
    from spatialdata import SpatialData


def anndata_variable_columns(
    data: AnnData, column_names: list[str], keys: str | Sequence[str]
) -> list[pl.Series]:
    """
    Return a list of variable columns as Polars `Series`.

    Raises
    ------
    VariableNotFoundError
        If any key is not present in `data.var_names`.
    """
    columns = []
    if isinstance(keys, str):
        keys = [keys]
    # remove keys that are already in column_names to avoid repeats
    keys = [key for key in keys if key not in column_names]
    # check that all keys are in data.var_names
    available_names = set(data.var_names)
    missing = [key for key in keys if key not in available_names]
    if missing:
        msg = f"Keys not found in data.var_names: {missing}"
        raise VariableNotFoundError(msg)

    # subset X
    X = data[:, keys].X

    if issparse(X):
        X = X.toarray()  # ty:ignore[unresolved-attribute]
    else:
        X = np.asarray(X)
    columns = [pl.Series(key, X[:, i]) for i, key in enumerate(keys)]

    return columns


def anndata_observations_frame(
    data: AnnData,
    /,
    variable_keys: str | Sequence[str] | None = None,
    *,
    observations_name: str | None = "barcode",
    include_dimensions: bool | int = False,
    metadata_columns: Sequence[str] | None = None,
    dimension_keys: Sequence[str] | None = None,
) -> DataFrame:
    """
    Build an Observations DataFrame from an AnnData object.

    Parameters
    ----------
    data : AnnData
        The AnnData object containing the observations.
    variable_keys : str | Sequence[str] | None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    observations_name : str | None, optional
        The name of the observation identifier column, default is 'barcode'.
        Pass `None` to omit the identifier column entirely.
    include_dimensions : bool | int
        Whether to include dimensions from embeddings in the DataFrame, default is False.
        Providing an integer will limit the number of dimensions to given number.
    metadata_columns : Sequence[str] | None
        Restrict which observation metadata columns are materialised. `None`
        includes all metadata columns (default). An empty sequence skips them.
    dimension_keys : Sequence[str] | None
        Restrict which embeddings are materialised when `include_dimensions`
        is truthy. Case-insensitive. `None` includes all embeddings (default).
        An empty sequence skips embeddings entirely.

    Returns
    -------
    DataFrame
        A DataFrame containing the observations, with optional variable keys and dimensions.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not an AnnData object.
    VariableNotFoundError
        If any `variable_keys` entry is not present in the variable names.
    KeyError
        If any `metadata_columns` entry is not a known metadata column, or any
        `dimension_keys` entry is not a known embedding.
    ValueError
        If `include_dimensions` is a negative integer.
    """
    # Check if data is an AnnData object
    if not isinstance(data, AnnData):
        msg = f"Unsupported data type: `{type(data)}`"
        raise UnsupportedDataTypeError(msg)
    if not isinstance(data.obs, pd.DataFrame):  # in case of Dataset2D
        part = data.obs.to_memory()
    else:
        part = data.obs
    partm = data.obsm
    # PART 1: INITIALIZE
    if observations_name is None:
        columns = []
    else:
        columns = [pl.Series(observations_name, data.obs_names)]
    # PART 2: ADD AnnData.obs
    if metadata_columns is None:
        selected_columns = list(part.columns)
    else:
        missing = [name for name in metadata_columns if name not in part.columns]
        if missing:
            msg = f"metadata_columns not found in observations: {missing}"
            raise KeyError(msg)
        selected_columns = list(metadata_columns)
    for key in selected_columns:
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
        selected_embeddings = _select_embedding_keys(partm, dimension_keys)
        for X in selected_embeddings:
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
    # Empty list short-circuits: data[:, []].X still triggers a full sparse slice.
    if variable_keys:
        column_names = [column.name for column in columns]
        columns.extend(
            anndata_variable_columns(data=data, column_names=column_names, keys=variable_keys)
        )

    return pl.DataFrame(columns)


def anndata_variables_frame(
    data: AnnData,
    *,
    variables_name: str | None = "variable",
    include_dimensions: bool | int = False,
    metadata_columns: Sequence[str] | None = None,
    dimension_keys: Sequence[str] | None = None,
) -> DataFrame:
    """
    Build a Variables DataFrame from an AnnData object.

    Parameters
    ----------
    data : AnnData
        The AnnData object containing the variables.
    variables_name : str | None
        Name for the variable identifier column, default is 'variable'.
        Pass `None` to omit the identifier column entirely.
    include_dimensions : bool | int
        Whether to include variable-axis embeddings in the DataFrame, default is False.
        Providing an integer will limit the number of dimensions to given number.
    metadata_columns : Sequence[str] | None
        Restrict which variable metadata columns are materialised. `None`
        includes all metadata columns (default). An empty sequence skips them.
    dimension_keys : Sequence[str] | None
        Restrict which variable-axis embeddings are materialised when
        `include_dimensions` is truthy. Case-insensitive. `None` includes all
        embeddings (default). An empty sequence skips embeddings entirely.

    Returns
    -------
    DataFrame
        A DataFrame containing the variables.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not an AnnData object.
    KeyError
        If any `metadata_columns` entry is not a known metadata column, or any
        `dimension_keys` entry is not a known embedding.
    ValueError
        If `include_dimensions` is a negative integer.
    """
    # PART 1: INITIALIZE
    if not isinstance(data, AnnData):
        msg = f"Unsupported data type: `{type(data)}`"
        raise UnsupportedDataTypeError(msg)
    if not isinstance(data.var, pd.DataFrame):  # in case of Dataset2D
        part = data.var.to_memory()
    else:
        part = data.var
    partm = data.varm
    # PART1: initalize columns
    if variables_name is None:
        columns = []
    else:
        columns = [pl.Series(variables_name, data.var_names)]
    # PART 3: ADD AnnData.var
    if metadata_columns is None:
        selected_columns = list(part.columns)
    else:
        missing = [name for name in metadata_columns if name not in part.columns]
        if missing:
            msg = f"metadata_columns not found in variables: {missing}"
            raise KeyError(msg)
        selected_columns = list(metadata_columns)
    for key in selected_columns:
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
        selected_embeddings = _select_embedding_keys(partm, dimension_keys)
        for X in selected_embeddings:
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


def _select_embedding_keys(embeddings, dimension_keys: Sequence[str] | None) -> list[str]:
    """Return the subset of embedding keys to materialise, matched case-insensitively."""
    available = list(embeddings.keys())
    if dimension_keys is None:
        return available
    available_upper = {name.upper(): name for name in available}
    selected = []
    missing = []
    for key in dimension_keys:
        actual = available_upper.get(key.upper())
        if actual is None:
            missing.append(key)
        else:
            selected.append(actual)
    if missing:
        msg = f"dimension_keys not found in embeddings: {missing}"
        raise KeyError(msg)
    return selected


def build_frame(
    data: AnnData | SpatialData,
    *,
    variable_keys: str | Sequence[str] | None = None,
    axis: Literal[0, 1] | None = None,
    observations_name: str | None = "barcode",
    variables_name: str | None = "variable",
    include_dimensions: bool | int = False,
    metadata_columns: Sequence[str] | None = None,
    dimension_keys: Sequence[str] | None = None,
) -> DataFrame:
    """
    Build a DataFrame from a single-cell object.

    Parameters
    ----------
    data : AnnData | SpatialData
        The AnnData or SpatialData object containing the variables.
    variable_keys : str | Sequence[str] | None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    axis : {0,1} | None
        The axis to build the frame for. 0 for observations, 1 for variables.
    observations_name : str | None
        The name of the observation identifier column, default is 'barcode'.
        Pass `None` to omit the identifier column entirely.
    variables_name : str | None
        Name for the variable identifier column, default is 'variable'.
        Pass `None` to omit the identifier column entirely.
    include_dimensions : bool | int
        Whether to include dimensions in the DataFrame, default is False.
        Providing an integer will limit the number of dimensions to given number.
    metadata_columns : Sequence[str] | None
        Restrict which metadata columns from the selected axis are materialised.
        `None` includes all metadata columns (default). An empty sequence skips them.
    dimension_keys : Sequence[str] | None
        Restrict which embeddings are materialised when `include_dimensions`
        is truthy. Case-insensitive. `None` includes all embeddings (default).

    Returns
    -------
    DataFrame
        A polars DataFrame containing the variables.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported data object.
    VariableNotFoundError
        If any `variable_keys` entry is not present in variable names.
    KeyError
        If any `metadata_columns` or `dimension_keys` entry is unknown.
    ValueError
        If `axis` cannot be inferred, or the data object does not resolve to
        exactly one annotation table.

    Examples
    --------
    Providing axis, 0 for observations axis and 1 for variables axis.

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        data = cl.datasets.pbmc3k(cache_directory="data")
        frame = cl.build_frame(data, axis=0, include_dimensions=2)
        frame.head()

    Providing variable_keys allows function to infer the axis as 0.
    It also adds the given variable_keys to the DataFrame.

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        data = cl.datasets.pbmc3k(cache_directory="data")
        frame = cl.build_frame(data, variable_keys=["CD14", "HBA1"], include_dimensions=2)
        frame.head()

    """
    from spatialdata import SpatialData

    if isinstance(data, SpatialData):
        tables = list(data.tables.keys())
        if len(tables) == 1:
            # retrieve an anndata
            data = data.tables[tables[0]]
        elif len(tables) == 0:
            msg = "No annotation tables found in the SpatialData object."
            raise ValueError(msg)
        else:
            msg = (
                f"Multiple annotation tables found: {tables}. "
                "Pass a SpatialData with a single table, or extract the table "
                "yourself before calling `build_frame`."
            )
            raise ValueError(msg)

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
                metadata_columns=metadata_columns,
                dimension_keys=dimension_keys,
            )
        elif axis == 1:
            frame = anndata_variables_frame(
                data,
                variables_name=variables_name,
                include_dimensions=include_dimensions,
                metadata_columns=metadata_columns,
                dimension_keys=dimension_keys,
            )
        elif axis is None:
            msg = "`axis` parameter must be specified, 0 for observations, 1 for variables."
            raise ValueError(msg)
    else:
        msg = f"Unsupported data type: `{type(data)}`"
        raise UnsupportedDataTypeError(msg)

    return frame
