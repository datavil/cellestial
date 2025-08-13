from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import polars as pl
from anndata import AnnData

from cellestial.util.errors import ConflictingKeysError, KeyNotFoundError


def anndata_features_frame(
    data: AnnData,
) -> pl.DataFrame:
    return pl.DataFrame()


def anndata_observations_frame(
    data: AnnData,
    keys: str | Sequence[str] | None,
    include_dimensions: bool = False,
) -> pl.DataFrame:
    # PART 1: INITIALIZE
    frame = pl.DataFrame()
    # PART 2: ADD AnnData.obs
    for key in data.obs.columns:
        frame = frame.with_columns(pl.Series(key, data.obs[key]))
    # PART 3: ADD obs_names
    frame = frame.with_columns(pl.Series("obs_names", data.obs.index))
    if include_dimensions:
        for X in data.obsm:
            frame = frame.with_columns(pl.Series(data.obsm[X]))
    return frame
