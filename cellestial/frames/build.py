from __future__ import annotations

__lazy_modules__ = ["anndata", "mudata", "scipy.sparse"]

from typing import TYPE_CHECKING, Literal

import polars as pl
from anndata import AnnData
from mudata import MuData
from scipy.sparse import issparse

from cellestial.frames._container import _container
from cellestial.util.errors import (
    DuplicateKeysError,
    InvalidArgumentError,
    InvalidArgumentTypeError,
    KeyNotFoundError,
    _did_you_mean,
    _unsupported_data_type,
)
from cellestial.util.utilities import _is_spatialdata

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from polars import DataFrame
    from spatialdata import SpatialData


def _categorical_series(values: pd.Series) -> pl.Series:
    """
    Convert a categorical metadata column, keeping its category order.

    Notes
    -----
    Categories become a `pl.Enum` in their source order, so legends and group
    axes list them as the data defines them rather than by first appearance
    (polars `Categorical` keeps no order of its own). Numeric categories are
    renamed to strings first, so missing values stay null (`astype(str)` would
    turn them into a "nan" category). Categoricals that polars reads as another
    type (e.g. boolean or datetime categories) keep that type.
    """
    if values.cat.categories.dtype.kind in "iuf":
        values = values.cat.rename_categories(str)
    series = pl.Series(values)
    if series.dtype == pl.Categorical:
        levels = list(dict.fromkeys(str(category) for category in values.cat.categories))
        series = series.cast(pl.String).cast(pl.Enum(levels))
    return series


def _reject_id_column_collision(
    columns: list[pl.Series], selected_columns: list[str], parameter: str, axis_name: str
) -> None:
    """
    Raise `InvalidArgumentError` when the id column shares its name with a metadata column.

    Notes
    -----
    Both would become columns of the same name, which a frame cannot hold.
    """
    if columns and columns[0].name in selected_columns:
        name = columns[0].name
        msg = (
            f"`{parameter}={name!r}` collides with the {axis_name} metadata column `{name}`. "
            f"Pass another name, e.g. `{parameter}='{name}_id'`, or `{parameter}=None` to "
            f"omit the id column. Where the function has no `{parameter}`, rename the "
            "metadata column."
        )
        raise InvalidArgumentError(msg)


def _reject_duplicate_keys(
    keys: Sequence[str], parameter: str, *, case_sensitive: bool = True
) -> None:
    """
    Raise `DuplicateKeysError` when `keys` names the same key more than once.

    Notes
    -----
    With `case_sensitive=False`, case variants (`X_umap`, `x_UMAP`) count as one
    key, matching how embeddings are looked up.
    """
    seen: dict[str, str] = {}
    repeated: list[str] = []
    for key in keys:
        normalized = key if case_sensitive else key.upper()
        if normalized in seen and key not in repeated:
            repeated.append(key)
        seen.setdefault(normalized, key)
    if repeated:
        msg = (
            f"`{parameter}` names {repeated!r} more than once. "
            f"Pass each key once: {list(seen.values())!r}."
        )
        raise DuplicateKeysError(msg)


def anndata_variable_columns(
    data: AnnData | MuData, column_names: list[str], keys: str | Sequence[str]
) -> list[pl.Series]:
    """
    Return a list of variable columns as Polars `Series`.

    Raises
    ------
    VariableNotFoundError
        If any key is not present in the variable names.
    AmbiguousVariableError
        If a bare key is present in more than one modality.
    DuplicateKeysError
        If a key is repeated.
    """
    if isinstance(keys, str):
        keys = [keys]
    _reject_duplicate_keys(keys, "variable_keys")
    # remove keys that are already in column_names, to avoid repeats
    keys = [key for key in keys if key not in column_names]

    return _container(data).fetch_variable_columns(keys)


def anndata_observations_frame(
    data: AnnData | MuData,
    /,
    variable_keys: str | Sequence[str] | None = None,
    *,
    observation_id_column: str | None = "Barcode",
    include_dimensions: bool | int = False,
    metadata_columns: str | Sequence[str] | None = None,
    dimension_keys: str | Sequence[str] | None = None,
) -> DataFrame:
    """
    Build an Observations DataFrame from an AnnData object.

    Parameters
    ----------
    data : AnnData | MuData
        The data object containing the observations.
    variable_keys : str | Sequence[str] | None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    observation_id_column : str | None, optional
        The name of the observation identifier column, default is 'Barcode'.
        Pass `None` to omit the identifier column entirely.
    include_dimensions : bool | int
        Whether to include dimensions from embeddings in the DataFrame, default is False.
        Providing an integer will limit the number of dimensions to given number.
    metadata_columns : str | Sequence[str] | None
        Restrict which observation metadata columns are materialised. `None`
        includes all metadata columns (default). An empty sequence skips them.
        A single string selects one column.
    dimension_keys : str | Sequence[str] | None
        Restrict which embeddings are materialised when `include_dimensions`
        is truthy. Case-insensitive. `None` includes all embeddings (default).
        A single string selects one embedding.
        An empty sequence skips embeddings entirely.

    Returns
    -------
    DataFrame
        A DataFrame containing the observations, with optional variable keys and dimensions.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported data object.
    VariableNotFoundError
        If any `variable_keys` entry is not present in the variable names.
    AmbiguousVariableError
        If a bare `variable_keys` entry is present in more than one modality.
    KeyNotFoundError
        If any `metadata_columns` entry is not a known metadata column, or any
        `dimension_keys` entry is not a known embedding.
    InvalidArgumentError
        If `include_dimensions` is a negative integer, or the id column shares
        its name with a selected metadata column.
    InvalidArgumentTypeError
        If `include_dimensions` is neither a bool nor an int.
    """
    container = _container(data)
    part = container.observation_metadata()
    partm = container.observation_embeddings()
    # PART 1: INITIALIZE
    if observation_id_column is None:
        columns = []
    else:
        columns = [pl.Series(observation_id_column, container.observation_names())]
    # PART 2: ADD AnnData.obs
    if isinstance(metadata_columns, str):
        metadata_columns = [metadata_columns]
    if metadata_columns is None:
        selected_columns = list(part.columns)
    else:
        missing = [name for name in metadata_columns if name not in part.columns]
        if missing:
            msg = f"`metadata_columns` not found in observations: {missing}."
            hint = _did_you_mean(missing, part.columns)
            if hint:
                msg += f" {hint}"
            raise KeyNotFoundError(msg)
        _reject_duplicate_keys(metadata_columns, "metadata_columns")
        selected_columns = list(metadata_columns)
    _reject_id_column_collision(columns, selected_columns, "observation_id_column", "observation")
    for key in selected_columns:
        if part.dtypes[key] == "category":
            columns.append(_categorical_series(part[key]))
        else:
            columns.append(pl.Series(part[key]))

    # PART 3: ADD dimensions if needed
    if include_dimensions:
        selected_embeddings = _select_embedding_keys(partm, dimension_keys)
        for X in selected_embeddings:
            total_columns = partm[X].shape[1]  # Number of dimensions (columns)
            if isinstance(include_dimensions, int) and not isinstance(include_dimensions, bool):
                if include_dimensions >= 0:
                    column_count = min(include_dimensions, total_columns)
                else:
                    msg = (
                        "`include_dimensions` cannot be a negative number, "
                        f"but received {include_dimensions}."
                    )
                    raise InvalidArgumentError(msg)
            elif isinstance(include_dimensions, bool):
                column_count = total_columns
            else:
                msg = "Argument for `include_dimensions` MUST be either a `bool` or an `int` type."
                msg += f" You provided type {type(include_dimensions)}"
                raise InvalidArgumentTypeError(msg)

            embedding = partm[X]
            if issparse(embedding):
                embedding = embedding[:, :column_count].toarray()  # ty:ignore[unresolved-attribute]
            for column_index in range(column_count):
                columns.append(
                    pl.Series(f"{X.upper()}{column_index + 1}", embedding[:, column_index])
                )

    # PART 4: ADD keys if provided
    # Empty list short-circuits: data[:, []].X still triggers a full sparse slice.
    if variable_keys:
        column_names = [column.name for column in columns]
        columns.extend(
            anndata_variable_columns(data=data, column_names=column_names, keys=variable_keys)
        )

    return pl.DataFrame(columns)


def anndata_variables_frame(
    data: AnnData | MuData,
    *,
    variable_id_column: str | None = "Variable",
    include_dimensions: bool | int = False,
    metadata_columns: str | Sequence[str] | None = None,
    dimension_keys: str | Sequence[str] | None = None,
) -> DataFrame:
    """
    Build a Variables DataFrame from an AnnData object.

    Parameters
    ----------
    data : AnnData | MuData
        The data object containing the variables.
    variable_id_column : str | None
        Name for the variable identifier column, default is 'Variable'.
        Pass `None` to omit the identifier column entirely.
    include_dimensions : bool | int
        Whether to include variable-axis embeddings in the DataFrame, default is False.
        Providing an integer will limit the number of dimensions to given number.
    metadata_columns : str | Sequence[str] | None
        Restrict which variable metadata columns are materialised. `None`
        includes all metadata columns (default). An empty sequence skips them.
        A single string selects one column.
    dimension_keys : str | Sequence[str] | None
        Restrict which variable-axis embeddings are materialised when
        `include_dimensions` is truthy. Case-insensitive. `None` includes all
        embeddings (default). An empty sequence skips embeddings entirely.
        A single string selects one embedding.

    Returns
    -------
    DataFrame
        A DataFrame containing the variables.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported data object.
    KeyNotFoundError
        If any `metadata_columns` entry is not a known metadata column, or any
        `dimension_keys` entry is not a known embedding.
    InvalidArgumentError
        If `include_dimensions` is a negative integer, or the id column shares
        its name with a selected metadata column.
    InvalidArgumentTypeError
        If `include_dimensions` is neither a bool nor an int.
    """
    # PART 1: INITIALIZE
    container = _container(data)
    part = container.variable_metadata()
    partm = container.variable_embeddings()
    # PART 2: initalize columns
    if variable_id_column is None:
        columns = []
    else:
        columns = [pl.Series(variable_id_column, container.variable_names())]
    # PART 3: ADD AnnData.var
    if isinstance(metadata_columns, str):
        metadata_columns = [metadata_columns]
    if metadata_columns is None:
        selected_columns = list(part.columns)
    else:
        missing = [name for name in metadata_columns if name not in part.columns]
        if missing:
            msg = f"`metadata_columns` not found in variables: {missing}."
            hint = _did_you_mean(missing, part.columns)
            if hint:
                msg += f" {hint}"
            raise KeyNotFoundError(msg)
        _reject_duplicate_keys(metadata_columns, "metadata_columns")
        selected_columns = list(metadata_columns)
    _reject_id_column_collision(columns, selected_columns, "variable_id_column", "variable")
    for key in selected_columns:
        if part.dtypes[key] == "category":
            columns.append(_categorical_series(part[key]))
        else:
            columns.append(pl.Series(part[key]))

    # PART 4: ADD dimensions if needed
    if include_dimensions:
        selected_embeddings = _select_embedding_keys(partm, dimension_keys)
        for X in selected_embeddings:
            total_columns = partm[X].shape[1]  # Number of dimensions (columns)
            if isinstance(include_dimensions, int) and not isinstance(include_dimensions, bool):
                if include_dimensions >= 0:
                    column_count = min(include_dimensions, total_columns)
                else:
                    msg = (
                        "`include_dimensions` cannot be a negative number, "
                        f"but received {include_dimensions}."
                    )
                    raise InvalidArgumentError(msg)
            elif isinstance(include_dimensions, bool):
                column_count = total_columns
            else:
                msg = "Argument for `include_dimensions` MUST be either a `bool` or an `int` type."
                msg += f" You provided type {type(include_dimensions)}"
                raise InvalidArgumentTypeError(msg)

            embedding = partm[X]
            if issparse(embedding):
                embedding = embedding[:, :column_count].toarray()  # ty:ignore[unresolved-attribute]
            for column_index in range(column_count):
                columns.append(
                    pl.Series(f"{X.upper()}{column_index + 1}", embedding[:, column_index])
                )

    return pl.DataFrame(columns)


def _select_embedding_keys(embeddings, dimension_keys: str | Sequence[str] | None) -> list[str]:
    """Return the subset of embedding keys to materialise, matched case-insensitively."""
    available = list(embeddings.keys())
    if isinstance(dimension_keys, str):
        dimension_keys = [dimension_keys]
    if dimension_keys is None:
        return available
    _reject_duplicate_keys(dimension_keys, "dimension_keys", case_sensitive=False)
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
        msg = f"`dimension_keys` not found in embeddings: {missing}. Available: {available}."
        hint = _did_you_mean(missing, available)
        if hint:
            msg += f" {hint}"
        raise KeyNotFoundError(msg)
    return selected


def build_frame(
    data: AnnData | SpatialData | MuData,
    *,
    variable_keys: str | Sequence[str] | None = None,
    axis: Literal[0, 1] | None = None,
    observation_id_column: str | None = "Barcode",
    variable_id_column: str | None = "Variable",
    include_dimensions: bool | int = False,
    metadata_columns: str | Sequence[str] | None = None,
    dimension_keys: str | Sequence[str] | None = None,
) -> DataFrame:
    """
    Build a DataFrame from a single-cell object.

    Parameters
    ----------
    data : AnnData | SpatialData | MuData
        The data object containing the variables.
    variable_keys : str | Sequence[str] | None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    axis : {0,1} | None
        The axis to build the frame for. 0 for observations, 1 for variables.
    observation_id_column : str | None
        The name of the observation identifier column, default is 'Barcode'.
        Pass `None` to omit the identifier column entirely.
    variable_id_column : str | None
        Name for the variable identifier column, default is 'Variable'.
        Pass `None` to omit the identifier column entirely.
    include_dimensions : bool | int
        Whether to include dimensions in the DataFrame, default is False.
        Providing an integer will limit the number of dimensions to given number.
    metadata_columns : str | Sequence[str] | None
        Restrict which metadata columns from the selected axis are materialised.
        `None` includes all metadata columns (default). An empty sequence skips them.
        A single string selects one column.
    dimension_keys : str | Sequence[str] | None
        Restrict which embeddings are materialised when `include_dimensions`
        is truthy. Case-insensitive. `None` includes all embeddings (default).
        A single string selects one embedding.

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
    AmbiguousVariableError
        If a bare `variable_keys` entry is present in more than one modality.
    KeyNotFoundError
        If any `metadata_columns` or `dimension_keys` entry is unknown.
    InvalidArgumentError
        If `axis` is missing or is neither 0 nor 1, `include_dimensions` is
        negative, an id column shares its name with a selected metadata column,
        the data object does not resolve to exactly one annotation
        table, or a multimodal object does not share observations across its
        modalities.
    InvalidArgumentTypeError
        If `include_dimensions` is neither a bool nor an int.

    Notes
    -----
    For a multimodal object, metadata and embeddings are read at the container
    level. Modality-level columns are already available there, prefixed as
    `modality:column`. Variables resolve to the modality owning them: a bare
    name works when it is unique across modalities, and `modality:name`
    disambiguates otherwise.

    Examples
    --------
    Providing axis, 0 for observations axis and 1 for variables axis.

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        data = cl.datasets.pbmc3k()
        frame = cl.build_frame(data, axis=0, include_dimensions=2)
        frame.head()

    Providing variable_keys allows function to infer the axis as 0.
    It also adds the given variable_keys to the DataFrame.

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        data = cl.datasets.pbmc3k()
        frame = cl.build_frame(data, variable_keys=["CD14", "HBA1"], include_dimensions=2)
        frame.head()

    """
    if _is_spatialdata(data):
        tables = list(data.tables.keys())
        if len(tables) == 1:
            # retrieve an anndata
            data = data.tables[tables[0]]
        elif len(tables) == 0:
            msg = "No annotation tables found in the SpatialData object."
            raise InvalidArgumentError(msg)
        else:
            msg = (
                f"Multiple annotation tables found: {tables}. "
                "Pass a SpatialData with a single table, or extract the table "
                "yourself before calling `build_frame`."
            )
            raise InvalidArgumentError(msg)

    if isinstance(data, MuData) and data.axis != 0:
        # Only shared observations keep the observations-frame contract; the
        # other layouts concatenate observations, so identifiers can repeat.
        msg = (
            "Only multimodal objects whose modalities share observations are supported "
            f"(`axis=0`), but this object has `axis={data.axis}`. "
            "Pass a single modality instead, e.g. `data['rna']`."
        )
        raise InvalidArgumentError(msg)

    if isinstance(data, (AnnData, MuData)):
        # infer the axis if not provided
        if axis is None and variable_keys is not None:
            axis = 0

        if axis == 0:
            frame = anndata_observations_frame(
                data,
                variable_keys=variable_keys,
                observation_id_column=observation_id_column,
                include_dimensions=include_dimensions,
                metadata_columns=metadata_columns,
                dimension_keys=dimension_keys,
            )
        elif axis == 1:
            frame = anndata_variables_frame(
                data,
                variable_id_column=variable_id_column,
                include_dimensions=include_dimensions,
                metadata_columns=metadata_columns,
                dimension_keys=dimension_keys,
            )
        elif axis is None:
            msg = "`axis` parameter must be specified, 0 for observations, 1 for variables."
            raise InvalidArgumentError(msg)
        else:
            msg = f"`axis` must be 0 for observations or 1 for variables, but received {axis!r}."
            raise InvalidArgumentError(msg)
    else:
        raise _unsupported_data_type(data, AnnData, "SpatialData", MuData)

    return frame
