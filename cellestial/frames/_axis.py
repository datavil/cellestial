from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pandas as pd


def _axis_frame_columns(
    *,
    identifiers: Iterable[str],
    identifier_name: str | None,
    metadata: pd.DataFrame,
    metadata_columns: Sequence[str] | None,
    metadata_axis: Literal["observations", "variables"],
    embeddings,
    include_dimensions: bool | int,
    dimension_keys: Sequence[str] | None,
) -> list[pl.Series]:
    """Build identifier, metadata, and embedding columns for one annotated axis."""
    if identifier_name is None:
        columns = []
    else:
        columns = [pl.Series(identifier_name, identifiers)]

    if metadata_columns is None:
        selected_metadata = list(metadata.columns)
    else:
        missing = [name for name in metadata_columns if name not in metadata.columns]
        if missing:
            msg = f"metadata_columns not found in {metadata_axis}: {missing}"
            raise KeyError(msg)
        selected_metadata = list(metadata_columns)

    for key in selected_metadata:
        if metadata.dtypes[key] == "category" and metadata[key].cat.categories.dtype.kind in "iuf":
            columns.append(pl.Series(metadata[key].astype(str)).cast(pl.Categorical))
        else:
            columns.append(pl.Series(metadata[key]))

    if include_dimensions:
        selected_embeddings = _select_embedding_keys(embeddings, dimension_keys)
        for embedding_key in selected_embeddings:
            total_dimensions = embeddings[embedding_key].shape[1]
            if isinstance(include_dimensions, int) and not isinstance(include_dimensions, bool):
                if include_dimensions < 0:
                    msg = "Number of dimensions cannot be a negative number."
                    raise ValueError(msg)
                dimension_count = min(include_dimensions, total_dimensions)
            elif isinstance(include_dimensions, bool):
                dimension_count = total_dimensions
            else:
                msg = (
                    "Argument for `include_dimensions` MUST be either a `bool` "
                    f"or an `int` type. You provided type {type(include_dimensions)}"
                )
                raise TypeError(msg)

            for dimension_index in range(dimension_count):
                columns.append(
                    pl.Series(
                        f"{embedding_key.upper()}{dimension_index + 1}",
                        embeddings[embedding_key][:, dimension_index],
                    )
                )

    return columns


def _select_embedding_keys(embeddings, dimension_keys: Sequence[str] | None) -> list[str]:
    """Return embedding keys selected case-insensitively."""
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
