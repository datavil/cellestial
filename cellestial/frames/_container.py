from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from mudata import MuData
from scipy.sparse import issparse

from cellestial.util.errors import (
    AmbiguousVariableError,
    KeyNotFoundError,
    VariableNotFoundError,
    _unsupported_data_type,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def _as_array(embedding):
    """
    Return an embedding as something positionally indexable by column.

    An embedding store accepts a `DataFrame` as well as an array, and callers
    index embeddings as `value[:, column]`, which a `DataFrame` reads as a label
    lookup and rejects.
    """
    if isinstance(embedding, pd.DataFrame):
        return embedding.to_numpy()
    return embedding


class _Container:
    """
    Backend-agnostic view over a single-cell data object.

    Answers the questions cellestial asks of a data object (metadata tables,
    identifiers, embeddings, variable columns) so that no other module needs to
    know which backend it received.

    Notes
    -----
    Internal. Never appears in a public signature, is never constructed by
    users, and is not exported from the package. Build one with `_container`.

    This base implementation is written against the attributes MuData and
    AnnData share, so it is correct for both. `_MuDataContainer` overrides only
    the methods that genuinely diverge, which makes that override list a
    complete inventory of the difference between the two backends.

    `__init__` only stores a reference, so constructing a container is free and
    callers can keep taking the raw data object in their signatures.
    """

    __slots__ = ("_data",)

    # Annotation only, no assignment, so it coexists with `__slots__`.
    _data: AnnData | MuData

    def __init__(self, data: AnnData | MuData) -> None:
        self._data = data

    # --- shared: correct for AnnData and MuData alike ------------------------

    def observation_metadata(self) -> pd.DataFrame:
        """Return the observation metadata table."""
        part = self._data.obs
        if not isinstance(part, pd.DataFrame):  # in case of Dataset2D
            part = part.to_memory()
        return part

    def variable_metadata(self) -> pd.DataFrame:
        """Return the variable metadata table."""
        part = self._data.var
        if not isinstance(part, pd.DataFrame):  # in case of Dataset2D
            part = part.to_memory()
        return part

    def observation_columns(self) -> pd.Index:
        """
        Return the observation metadata column names.

        Notes
        -----
        Separate from `observation_metadata` so that key-classification callers,
        which only ask whether a name exists, never pull a backed table into
        memory just to read its column names.
        """
        return self._data.obs.columns

    def variable_columns(self) -> pd.Index:
        """Return the variable metadata column names. See `observation_columns`."""
        return self._data.var.columns

    def observation_names(self) -> pd.Index:
        """Return the observation identifiers."""
        return self._data.obs_names

    def variable_names(self) -> pd.Index:
        """Return the variable identifiers."""
        return self._data.var_names

    def n_observations(self) -> int:
        """Return the number of observations."""
        return self._data.n_obs

    # --- overridden by `_MuDataContainer` ------------------------------------

    def modality_names(self) -> list[str]:
        """Return the modality names, empty when the object has no modalities."""
        return []

    def select_modality(self, modality: str | None) -> AnnData:
        """Return the data object supplying stored analysis results."""
        if modality is not None:
            msg = (
                f"`modality={modality!r}` was given but this data object has no modalities. "
                "Drop the argument."
            )
            raise KeyNotFoundError(msg)
        # `_container` only builds this class for an AnnData; a multimodal
        # object always gets `_MuDataContainer`.
        return cast("AnnData", self._data)

    def modality_column(self, modality: str | None, column: str) -> str:
        """Return `column` as the selected modality names it."""
        return column

    def container_column(self, modality: str | None, column: str) -> str:
        """Return the container's name for a column of the selected modality."""
        return column

    def observation_embeddings(self) -> dict[str, NDArray]:
        """Return the observation-axis embeddings."""
        return {name: _as_array(value) for name, value in self._data.obsm.items()}

    def variable_embeddings(self) -> dict[str, NDArray]:
        """Return the variable-axis embeddings."""
        return {name: _as_array(value) for name, value in self._data.varm.items()}

    def owns_variable(self, key: str) -> bool:
        """Report whether `key` names a variable. Never raises."""
        return key in self._data.var_names

    def resolve_variable(self, key: str) -> tuple[str | None, str]:
        """
        Resolve `key` to the modality owning it and its name within that modality.

        Raises
        ------
        VariableNotFoundError
            If `key` does not name a variable.
        """
        if key not in self._data.var_names:
            msg = f"Keys not found in variable names: [{key!r}]"
            raise VariableNotFoundError(msg)
        return None, key

    @staticmethod
    def _require_unique_variables(names: pd.Index, keys: Sequence[str], where: str) -> None:
        """
        Reject keys that name more than one variable.

        A name matching several columns has no single set of values, and
        slicing on it silently yields the wrong ones rather than failing.
        """
        if names.is_unique:  # cached by pandas, so the common path is free
            return
        counts = names.value_counts()
        repeated = [key for key in dict.fromkeys(keys) if counts.get(key, 0) > 1]
        if repeated:
            msg = (
                f"{repeated} match more than one variable in {where}. "
                "Variable names must be unique; de-duplicate them before plotting."
            )
            raise AmbiguousVariableError(msg)

    def fetch_variable_columns(self, keys: Sequence[str]) -> list[pl.Series]:
        """Return one column of values per key, aligned to the observations."""
        missing = [key for key in keys if key not in self._data.var_names]
        if missing:
            msg = f"Keys not found in variable names: {missing}"
            raise VariableNotFoundError(msg)
        self._require_unique_variables(self._data.var_names, keys, "the data")

        matrix = self._data[:, keys].X
        if issparse(matrix):
            matrix = matrix.toarray()  # ty:ignore[unresolved-attribute]
        else:
            matrix = np.asarray(matrix)

        return [pl.Series(key, matrix[:, index]) for index, key in enumerate(keys)]


class _MuDataContainer(_Container):
    """
    Container for a multimodal object, routing variables to their modality.

    Notes
    -----
    Internal, see `_Container`. Metadata, identifiers and embeddings are read at
    the container level only: modality-level observation and variable columns
    are already present there, prefixed as `modality:column`.
    """

    __slots__ = ()

    # Narrows the base declaration so `.mod` / `.obsmap` resolve.
    _data: MuData

    def modality_names(self) -> list[str]:
        return list(self._data.mod)

    def select_modality(self, modality: str | None) -> AnnData:
        available = self.modality_names()
        if modality is None:
            if len(available) != 1:
                msg = f"`modality` is required for this data object. Available: {available}."
                raise KeyNotFoundError(msg)
            modality = available[0]
        elif modality not in self._data.mod:
            msg = f"Unknown modality `{modality}`. Available: {available}."
            raise KeyNotFoundError(msg)
        part = self._data.mod[modality]
        if not isinstance(part, AnnData):
            # A modality may itself be multimodal; nesting is out of scope.
            raise _unsupported_data_type(part, AnnData)
        return part

    def modality_column(self, modality: str | None, column: str) -> str:
        """
        Return `column` as the selected modality names it.

        A container carries modality columns prefixed as `modality:column`,
        while the modality itself holds them unprefixed. Callers that hand a
        container-level column name to a single modality translate it here.
        """
        prefix = f"{modality}:"
        return column.removeprefix(prefix) if column.startswith(prefix) else column

    def container_column(self, modality: str | None, column: str) -> str:
        """
        Return the container's name for a column of the selected modality.

        The inverse of `modality_column`. The qualified `modality:column` form
        wins whenever it exists, because that is the modality's own column: a
        container can carry an unrelated global column under the bare name. In
        `minipbcite.h5mu` both `leiden` (9 joint clusters) and `rna:leiden` (14
        RNA clusters) exist and disagree on 356 of 411 cells, so preferring the
        bare name would silently group results by the wrong clustering.
        """
        qualified = f"{modality}:{column}"
        columns = self._data.obs.columns
        if qualified in columns:
            return qualified
        if column in columns:
            return column
        msg = (
            f"Column `{column}` of modality `{modality}` has no counterpart on the "
            f"container (looked for `{qualified}` and `{column}`)."
        )
        raise KeyNotFoundError(msg)

    def observation_embeddings(self) -> dict[str, NDArray]:
        return self._without_modality_masks(self._data.obsm)

    def variable_embeddings(self) -> dict[str, NDArray]:
        return self._without_modality_masks(self._data.varm)

    def _without_modality_masks(self, embeddings) -> dict[str, NDArray]:
        """
        Drop the boolean membership masks stored alongside the real embeddings.

        A multimodal container keys one mask per modality in both `obsm` and
        `varm`. Left in, they would surface as junk single-column dimensions in
        every frame.
        """
        modalities = set(self._data.mod)
        return {
            name: _as_array(value) for name, value in embeddings.items() if name not in modalities
        }

    def owns_variable(self, key: str) -> bool:
        modality, separator, name = key.partition(":")
        if separator and modality in self._data.mod and name in self._data.mod[modality].var_names:
            return True
        # Fall back to the literal name: some datasets store variable names that
        # already carry the prefix (`rna:SAMD11`), and ATAC peak names contain
        # colons of their own (`chr1:1000-2000`).
        return any(key in part.var_names for part in self._data.mod.values())

    def resolve_variable(self, key: str) -> tuple[str, str]:
        """
        Resolve `key` to its owning modality.

        A qualified `modality:name` key selects the modality explicitly and is
        tried first. Otherwise the key is matched literally, and must then be
        owned by exactly one modality.

        Notes
        -----
        The qualified reading cannot be the only one: variable names may
        themselves contain colons, either because the dataset stores them
        already prefixed (`rna:SAMD11`) or because they are ATAC peaks
        (`chr1:1000-2000`). Both fall through to the literal match.

        Raises
        ------
        VariableNotFoundError
            If the key matches neither reading.
        AmbiguousVariableError
            If the literal key is owned by more than one modality.
        """
        modality, separator, name = key.partition(":")
        is_qualified = bool(separator) and modality in self._data.mod
        if is_qualified and name in self._data.mod[modality].var_names:
            return modality, name

        owners = [candidate for candidate, part in self._data.mod.items() if key in part.var_names]
        if not owners:
            if is_qualified:
                msg = f"`{name}` not found in the variable names of modality `{modality}`."
                raise VariableNotFoundError(msg)
            msg = (
                f"Keys not found in variable names: [{key!r}]. "
                f"Searched modalities: {self.modality_names()}."
            )
            raise VariableNotFoundError(msg)
        if len(owners) > 1:
            qualified = [f"{owner}:{key}" for owner in owners]
            msg = (
                f"`{key}` is present in more than one modality. "
                f"Qualify it with one of: {qualified}."
            )
            raise AmbiguousVariableError(msg)
        return owners[0], key

    def fetch_variable_columns(self, keys: Sequence[str]) -> list[pl.Series]:
        return [self._variable_column(key) for key in keys]

    def _variable_column(self, key: str) -> pl.Series:
        """
        Pull one variable from its modality, aligned to the container observations.

        Observations absent from the owning modality surface as NaN.
        """
        modality, name = self.resolve_variable(key)
        part = self._data.mod[modality]
        self._require_unique_variables(part.var_names, [name], f"modality `{modality}`")
        matrix = part[:, name].X
        if issparse(matrix):
            matrix = matrix.toarray()  # ty:ignore[unresolved-attribute]
        values = np.asarray(matrix).reshape(-1)

        # `obsmap` holds 1-based positions into the modality, 0 meaning the
        # observation is absent from it.
        positions = np.asarray(self._data.obsmap[modality]).reshape(-1)
        present = positions > 0

        # Promote from the source dtype rather than hardcoding float64: `X` is
        # float32 in most single-cell data, and hardcoding would double every
        # column. Integer counts still widen to float64, which float32 cannot
        # represent exactly beyond 2**24.
        dtype = np.promote_types(values.dtype, np.float32)
        aligned = np.full(self._data.n_obs, np.nan, dtype=dtype)
        aligned[present] = values[positions[present] - 1]

        return pl.Series(key, aligned)


def _container(data: object) -> _Container:
    """Return the container view for `data`."""
    if isinstance(data, MuData):
        return _MuDataContainer(data)
    if isinstance(data, AnnData):
        return _Container(data)
    raise _unsupported_data_type(data, AnnData, MuData)
