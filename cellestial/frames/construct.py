from __future__ import annotations

from typing import Iterable

import polars as pl
from anndata import AnnData

from cellestial.util.errors import ConflictingKeysError, KeyNotFoundError


def _decide_dimensions_key(data: AnnData, dimensions: str) -> str:
    """Decide on which key to use for the dimensions."""
    if isinstance(data, AnnData):
        # get every key that contains the name of the dimensions
        keys_list = [key for key in data.obsm if dimensions in key.lower()]
        if len(keys_list) == 0:
            msg = f"dimensions '{dimensions}' not found in the data"
            raise KeyError(msg)
        elif len(keys_list) == 1:
            dimensions_key = keys_list[0]
        else:  # multiple keys found
            # shorter keys have precedence
            max_len = max(len(key) for key in keys_list)
            min_len = min(len(key) for key in keys_list)
            # 2d have precedence (over 3d etc.)
            has_2d = [key for key in keys_list if "2d" in key.lower()]
            # resolve the keys
            if max_len > min_len:
                dimensions_key = min(keys_list, key=len)
            elif len(has_2d) > 0:
                dimensions_key = has_2d[0]
            else:
                dimensions_key = keys_list[0]

    return dimensions_key


def _expand_frame(data: AnnData, frame: pl.DataFrame, to_add: list[str]) -> pl.DataFrame:
    """
    frame already has dimensions.

    expand the frame with:
    - given tooltips
       - it can be in obs
           - check if the key is in obs
       - it can be a gene expression level
           - check if the key is in var_names
    add the columns to frame
    return the frame
    """
    for key in to_add:
        if key not in frame.columns:
            if key in data.obs.columns:
                frame = frame.with_columns(pl.Series(key, data.obs[key]))
            elif key in data.var_names:
                index = data.var_names.get_indexer([key])  # get the index of the gene
                frame = frame.with_columns(
                    pl.Series(key, data.X[:, index].flatten().astype("float32")),
                )
            else:
                msg = f"key '{key}' to expand is not present in `observation (.obs) names` nor `gene (.var) names`"
                raise ValueError(msg)
    return frame


def _check_key_conflicts(data: AnnData, keys: Iterable[str]) -> None:
    """Check if there are any keys conflicts in the data."""
    keys_from = []
    if isinstance(data, AnnData):
        for key in keys:
            if key in data.obs.columns:
                keys_from.append("obs")
            if key in data.var_names:
                keys_from.append("var_names")
            if key in data.var.columns:
                keys_from.append("var")

        # conflicting scenarios
        if "var" in keys_from:
            if "var_names" in keys_from or "obs" in keys_from:
                print(keys_from)
                msg = "keys from var and var_names or obs cannot be used together"
                raise ConflictingKeysError(msg)

    return


def _add_dimensions(
    data: AnnData,
    frame: pl.DataFrame,
    use_key: str | None,
    dimensions: str,
    xy: tuple[int, int] = (1, 2),
) -> pl.DataFrame:
    """Add the dimensions to the frame."""
    if use_key is None:
        dimension_key = _decide_dimensions_key(data=data, dimensions=dimensions)
    else:
        dimension_key = use_key
    x = f"{dimensions}{xy[0]}"  # e.g. umap1
    y = f"{dimensions}{xy[1]}"  # e.g. umap2
    xy_index = (xy[0] - 1, xy[1] - 1)
    frame = frame.with_columns(pl.from_numpy(data.obsm[dimension_key][:, xy_index], schema=[x, y]))
    return frame


def _construct_cell_frame(
    *,
    data: AnnData,
    keys: Iterable[str],
    dimensions: str,
    xy: tuple[int, int] = (1, 2),
    use_key: str | None = None,
    barcode_name: str = "Barcode",
) -> pl.DataFrame:
    """
    Construct a polars DataFrame from data.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to construct the DataFrame from.
    keys : Iterable[str]
        The keys to include in the DataFrame.

    Returns
    -------
    pl.DataFrame
        The constructed DataFrame.
    """
    if not isinstance(data, (AnnData)):
        msg = "data must be an AnnData object"
        raise TypeError(msg)

    # initialize the frame
    frame = pl.DataFrame()
    # there could be other types in the future
    if isinstance(data, AnnData):
        # add the columns associated with the keys to the frame
        for key in keys:
            if key in data.obs.columns:
                # add the cell feature to the frame
                frame = frame.with_columns(pl.Series(key, data.obs[key]))
            elif key in data.var_names:
                # get the index of the gene
                # adata.X is a sparse matrix , axis0 is cells, axis1 is genes
                index = data.var_names.get_indexer([key])
                # add the gene expression level to the frame
                frame = frame.with_columns(
                    pl.Series(key, data.X[:, index].flatten().astype("float32")),
                )
            else:
                msg = f"key '{key}' not found in the data"
                raise KeyNotFoundError(msg)

        # add the cell index to the frame
        frame = frame.with_columns(pl.Series(barcode_name, data.obs_names))

        # add the dimensions to the frame
        frame = _add_dimensions(
            data=data,
            frame=frame,
            use_key=use_key,
            dimensions=dimensions,
        )

    return frame


def _construct_var_frame(
    *,
    data: AnnData,
    keys: Iterable[str],
    var_name: str = "Gene",
) -> pl.DataFrame:
    """
    Construct a polars DataFrame from data.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to construct the DataFrame from.
    keys : Iterable[str]
        The keys to include in the DataFrame.

    Returns
    -------
    pl.DataFrame
        The constructed DataFrame.
    """
    if not isinstance(data, (AnnData)):
        msg = "data must be an AnnData object"
        raise TypeError(msg)

    # initialize the frame
    frame = pl.DataFrame()
    # there could be other types in the future
    if isinstance(data, AnnData):
        # add the columns associated with the keys to the frame
        for key in keys:
            if key in data.var.columns:
                frame = frame.with_columns(pl.Series(key, data.var[key]))
            else:
                msg = f"key '{key}' not found in the data"
                raise KeyNotFoundError(msg)

        # add the variable index to the frame
        frame = frame.with_columns(pl.Series(var_name, data.var_names))

    return frame
