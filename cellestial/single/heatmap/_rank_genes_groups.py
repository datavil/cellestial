from __future__ import annotations

from typing import TYPE_CHECKING

from anndata import AnnData

from cellestial.util import _warn
from cellestial.util.errors import KeyNotFoundError, UnsupportedDataTypeError

if TYPE_CHECKING:
    from collections.abc import Sequence


_DEFAULT_RANK_GENES_KEY = "rank_genes_groups"


def _resolve_rank_genes_groups_key(rank_genes_groups: bool | str) -> str:
    """Translate the user-facing flag into the ``adata.uns`` key to read."""
    if rank_genes_groups is True:
        return _DEFAULT_RANK_GENES_KEY
    if isinstance(rank_genes_groups, str):
        return rank_genes_groups
    msg = (
        "`rank_genes_groups` must be True or a string naming the "
        "`adata.uns` key (e.g. 'rank_genes_groups_wilcoxon')."
    )
    raise TypeError(msg)


def _extract_rank_genes_groups(
    data: AnnData,
    *,
    rank_genes_groups: bool | str,
    n_genes: int,
    groups: Sequence[str] | None = None,
) -> tuple[dict[str, list[str]], str]:
    """
    Read top-N ranked genes per group from a precomputed rank result.

    Parameters
    ----------
    data : AnnData
        Source object that already contains a ``rank_genes_groups`` result.
    rank_genes_groups : bool | str
        ``True`` reads from the default ``rank_genes_groups`` key; a string
        reads from the matching custom key (e.g. ``rank_genes_groups_wilcoxon``).
    n_genes : int
        Number of top genes to pull per group.
    groups : Sequence[str] | None, default=None
        Subset of groups to include in mapping order. ``None`` keeps all
        groups in their stored order.

    Returns
    -------
    keys : dict[str, list[str]]
        Mapping from group label to top-N gene names, ready to feed into
        ``_resolve_key_groups``.
    group_by : str
        The categorical key used when ``rank_genes_groups`` was computed.
    """
    if n_genes < 1:
        msg = f"`n_genes` must be >= 1, got {n_genes}."
        raise ValueError(msg)

    uns_key = _resolve_rank_genes_groups_key(rank_genes_groups)

    if isinstance(data, AnnData):
        if uns_key not in data.uns:
            msg = (
                f"`adata.uns[{uns_key!r}]` not found. "
                "Run `scanpy.tl.rank_genes_groups` first "
                "(or pass the correct `key_added` value as `rank_genes_groups`)."
            )
            raise KeyNotFoundError(msg)

        record = data.uns[uns_key]
        if "names" not in record or "params" not in record:
            msg = (
                f"`adata.uns[{uns_key!r}]` does not look like a "
                "`rank_genes_groups` result (missing 'names' or 'params')."
            )
            raise KeyNotFoundError(msg)

        names = record["names"]
        available_groups = list(names.dtype.names)

        if groups is None:
            selected = available_groups
        else:
            selected = list(groups)
            unknown = [group for group in selected if group not in available_groups]
            if unknown:
                msg = (
                    f"Groups {unknown!r} not found in "
                    f"`adata.uns[{uns_key!r}]`. Available: {available_groups!r}."
                )
                raise KeyNotFoundError(msg)

        n_available = len(names)
        if n_genes > n_available:
            msg = (
                f"`n_genes={n_genes}` exceeds the {n_available} ranked genes "
                f"stored in `adata.uns[{uns_key!r}]`."
            )
            raise ValueError(msg)

        keys: dict[str, list[str]] = {}
        seen: dict[str, str] = {}
        dropped: list[tuple[str, str, str]] = []
        for group in selected:
            group_keys: list[str] = []
            for gene in names[group][:n_genes]:
                gene_name = str(gene)
                if gene_name in seen:
                    dropped.append((gene_name, seen[gene_name], group))
                    continue
                seen[gene_name] = group
                group_keys.append(gene_name)
            keys[group] = group_keys

        if dropped:
            examples = ", ".join(
                f"{gene!r} (kept in {first!r}, dropped from {later!r})"
                for gene, first, later in dropped[:3]
            )
            suffix = f" and {len(dropped) - 3} more" if len(dropped) > 3 else ""
            _warn(
                "Some genes ranked highly in multiple groups; "
                "keeping the first occurrence and dropping the rest "
                f"({examples}{suffix}). "
                "Increase `n_genes`, restrict `groups`, "
                "or pass an explicit `keys` mapping to control this."
            )

        params = record["params"]
        stored_group_by = params["groupby"] if "groupby" in params else None
        if stored_group_by is None:
            msg = (
                f"`adata.uns[{uns_key!r}]['params']` is missing 'groupby'; "
                "cannot infer `group_by`."
            )
            raise KeyNotFoundError(msg)

        return keys, str(stored_group_by)

    msg = f"Unsupported data type: `{type(data)}`"
    raise UnsupportedDataTypeError(msg)


def _resolve_rank_genes_groups_args(
    data: AnnData,
    *,
    rank_genes_groups: bool | str,
    n_genes: int,
    groups: Sequence[str] | None,
    keys: object,
    group_by: str | None,
) -> tuple[dict[str, list[str]], str]:
    """
    Validate caller inputs and return the resolved ``keys`` and ``group_by``.

    Raises if ``keys`` is also supplied, or if a caller-provided ``group_by``
    disagrees with the value stored in ``adata.uns[...]['params']['groupby']``.
    """
    if keys is not None:
        msg = (
            "`keys` must be omitted when `rank_genes_groups` is set; "
            "genes are derived from the stored ranking."
        )
        raise ValueError(msg)

    resolved_keys, stored_group_by = _extract_rank_genes_groups(
        data,
        rank_genes_groups=rank_genes_groups,
        n_genes=n_genes,
        groups=groups,
    )

    if group_by is not None and group_by != stored_group_by:
        msg = (
            f"`group_by={group_by!r}` does not match the value used when "
            f"`rank_genes_groups` was computed ({stored_group_by!r}). "
            "Omit `group_by` to inherit it, or recompute with the desired key."
        )
        raise ValueError(msg)

    return resolved_keys, stored_group_by
