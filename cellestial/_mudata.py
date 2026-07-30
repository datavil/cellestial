from __future__ import annotations

import numpy as np
from mudata import MuData

from cellestial.util.errors import VariableNotFoundError

_MUDATA_VARIABLE_SEPARATOR = ":"


def _validate_mudata_axis(data: MuData) -> None:
    """Require the shared-observation MuData layout supported by Cellestial."""
    if data.axis != 0:
        msg = (
            "Cellestial supports MuData only when `data.axis == 0` "
            "(shared observations). Select a modality, for example "
            '`data["rna"]`, for other MuData layouts.'
        )
        raise ValueError(msg)


def _resolve_mudata_variable(data: MuData, key: str) -> tuple[str, str, int]:
    """Resolve a `modality:variable` key to its modality and column index."""
    _validate_mudata_axis(data)

    modality, separator, variable = key.partition(_MUDATA_VARIABLE_SEPARATOR)
    if not separator or not modality or not variable:
        msg = f"MuData variable key `{key}` must use `modality:variable`, for example `rna:NKG7`."
        raise VariableNotFoundError(msg)

    if modality not in data.mod:
        available = list(data.mod.keys())
        msg = f"Modality `{modality}` not found. Available modalities: {available}."
        raise VariableNotFoundError(msg)

    modality_data = data.mod[modality]
    matches = np.flatnonzero(np.asarray(modality_data.var_names == variable))
    if len(matches) == 0:
        msg = f"Variable `{variable}` not found in modality `{modality}`."
        raise VariableNotFoundError(msg)
    if len(matches) > 1:
        msg = f"Variable `{variable}` is not unique in modality `{modality}`."
        raise ValueError(msg)

    return modality, variable, int(matches[0])
