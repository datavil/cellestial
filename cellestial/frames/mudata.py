from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from mudata import MuData
from scipy.sparse import issparse

from cellestial._mudata import _resolve_mudata_variable, _validate_mudata_axis
from cellestial.frames._axis import _axis_frame_columns
from cellestial.util.errors import _unsupported_data_type

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars import DataFrame


def mudata_variable_columns(
    data: MuData,
    column_names: list[str],
    keys: str | Sequence[str],
) -> list[pl.Series]:
    """Return modality-qualified variables aligned to the global observations."""
    if not isinstance(data, MuData):
        raise _unsupported_data_type(data, MuData)
    _validate_mudata_axis(data)

    if isinstance(keys, str):
        keys = [keys]
    keys = [key for key in keys if key not in column_names]

    columns = []
    for key in keys:
        modality, _variable, variable_index = _resolve_mudata_variable(data, key)
        modality_data = data.mod[modality]
        values = modality_data[:, [variable_index]].X
        if issparse(values):
            values = values.toarray()  # ty:ignore[unresolved-attribute]
        local_values = np.asarray(values).reshape(-1)

        observation_map = np.asarray(data.obsmap[modality]).reshape(-1)
        global_indices = pl.Series(observation_map.astype(np.int64) - 1)
        global_indices = global_indices.set(global_indices < 0, None)
        columns.append(pl.Series(key, local_values).gather(global_indices))

    return columns


def mudata_observations_frame(
    data: MuData,
    /,
    variable_keys: str | Sequence[str] | None = None,
    *,
    observations_name: str | None = "Barcode",
    include_dimensions: bool | int = False,
    metadata_columns: Sequence[str] | None = None,
    dimension_keys: Sequence[str] | None = None,
) -> DataFrame:
    """Build an observations frame from container-level MuData annotations."""
    if not isinstance(data, MuData):
        raise _unsupported_data_type(data, MuData)
    _validate_mudata_axis(data)

    columns = _axis_frame_columns(
        identifiers=data.obs_names,
        identifier_name=observations_name,
        metadata=data.obs,
        metadata_columns=metadata_columns,
        metadata_axis="observations",
        embeddings=data.obsm,
        include_dimensions=include_dimensions,
        dimension_keys=dimension_keys,
    )
    if variable_keys:
        column_names = [column.name for column in columns]
        columns.extend(
            mudata_variable_columns(
                data=data,
                column_names=column_names,
                keys=variable_keys,
            )
        )

    return pl.DataFrame(columns)


def mudata_variables_frame(
    data: MuData,
    *,
    variables_name: str | None = "Variable",
    include_dimensions: bool | int = False,
    metadata_columns: Sequence[str] | None = None,
    dimension_keys: Sequence[str] | None = None,
) -> DataFrame:
    """Build a variables frame from container-level MuData annotations."""
    if not isinstance(data, MuData):
        raise _unsupported_data_type(data, MuData)
    _validate_mudata_axis(data)

    qualified_names = _qualified_variable_names(data)
    columns = _axis_frame_columns(
        identifiers=qualified_names,
        identifier_name=variables_name,
        metadata=data.var,
        metadata_columns=metadata_columns,
        metadata_axis="variables",
        embeddings=data.varm,
        include_dimensions=include_dimensions,
        dimension_keys=dimension_keys,
    )
    return pl.DataFrame(columns)


def _qualified_variable_names(data: MuData) -> list[str]:
    """Return global variable identifiers as `modality:variable`."""
    modality_maps = {
        modality: np.asarray(data.varmap[modality]).reshape(-1) for modality in data.mod
    }
    qualified_names = []
    for global_index in range(data.n_vars):
        owners = [
            (modality, int(mapping[global_index]))
            for modality, mapping in modality_maps.items()
            if mapping[global_index] > 0
        ]
        if len(owners) != 1:
            msg = (
                "Each variable in a shared-observation MuData must belong to "
                f"exactly one modality; global variable index {global_index} "
                f"belongs to {len(owners)}."
            )
            raise ValueError(msg)

        modality, local_index = owners[0]
        variable = data.mod[modality].var_names[local_index - 1]
        qualified_names.append(f"{modality}:{variable}")

    if len(qualified_names) != len(set(qualified_names)):
        msg = "MuData contains duplicate variable names within a modality."
        raise ValueError(msg)

    return qualified_names
