from __future__ import annotations

from collections.abc import Sequence

from anndata import AnnData
from mudata import MuData

from cellestial.util.errors import KeyNotFoundError, _unsupported_data_type
from cellestial.util.utilities import _container


def _marker_names_per_group(
    data: AnnData,
    groups: Sequence[str] | None,
    *,
    key: str,
    n_genes: int,
) -> dict[str, list[str]]:
    """
    Read the top-`n_genes` gene names per group from a precomputed ranking.

    Shared core for `marker_genes` and `marker_genes_dict`: validates inputs
    and returns one full top-N list per selected group, in group order, with
    no cross-group de-duplication.
    """
    if groups is not None and (not isinstance(groups, Sequence) or isinstance(groups, str)):
        msg = "`groups` must be a Sequence of strings or None"
        raise TypeError(msg)

    if n_genes < 1:
        msg = f"`n_genes` must be >= 1, got {n_genes}."
        raise ValueError(msg)

    if isinstance(data, AnnData):
        if key not in data.uns:
            msg = (
                f"`adata.uns[{key!r}]` not found. "
                "Run `scanpy.tl.rank_genes_groups` first "
                "(or pass the correct `key_added` value as `key`)."
            )
            raise KeyNotFoundError(msg)

        record = data.uns[key]
        if "names" not in record:
            msg = (
                f"`adata.uns[{key!r}]` does not look like a "
                "`rank_genes_groups` result (missing 'names')."
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
                    f"`adata.uns[{key!r}]`. Available: {available_groups!r}."
                )
                raise KeyNotFoundError(msg)

        n_available = len(names)
        if n_genes > n_available:
            msg = (
                f"`n_genes={n_genes}` exceeds the {n_available} ranked genes "
                f"stored in `adata.uns[{key!r}]`."
            )
            raise ValueError(msg)

        markers = {group: [str(gene) for gene in names[group][:n_genes]] for group in selected}
    else:
        raise _unsupported_data_type(data, AnnData)

    return markers


def marker_genes(
    data: AnnData | MuData,
    groups: Sequence[str] | None = None,
    *,
    modality: str | None = None,
    key: str = "rank_genes_groups",
    n_genes: int = 5,
) -> list[str]:
    """
    Select the top-ranked marker gene names from a precomputed ranking.

    The returned list is meant to be passed straight into the `keys` argument
    of the dimensional and distribution plots (e.g. `umaps`, `violins`), so
    each marker becomes one panel.

    Parameters
    ----------
    data : AnnData
        The single-cell data object holding the precomputed differential
        expression ranking.
    groups : Sequence[str] | None, default=None
        Subset of groups to pull markers from, in order. `None` keeps all
        groups in their stored order.
    modality : str | None, default=None
        Which modality's stored analysis results to use. Required for a
        multimodal object holding more than one modality, and not accepted
        otherwise.
    key : str, default='rank_genes_groups'
        The key under which the precomputed ranking is stored on `data`.
    n_genes : int, default=5
        Number of top genes to pull per group.

    Returns
    -------
    list[str]
        Flattened, order-preserving list of marker gene names with duplicates
        removed (a gene ranked highly in several groups is kept once, at its
        first occurrence).

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    KeyNotFoundError
        If the ranking result or a requested group is missing.
    ValueError
        If `n_genes` is out of range.
    TypeError
        If `groups` is neither a Sequence of strings nor None.

    Notes
    -----
    Reads only the gene names, not their scores; use `markers` to plot the
    ranking itself. Use `marker_genes_dict` to keep the per-group grouping.

    Examples
    --------
    Color a UMAP grid by the top markers of a single group.

    .. jupyter-execute::

        import scanpy as sc

        import cellestial as cl

        data = sc.datasets.pbmc68k_reduced()

        cl.umaps(
            data,
            keys=cl.marker_genes(data, groups=["CD14+ Monocyte"], n_genes=4),
            size=2,
            ncol=2,
        )

    A heatmap of the top markers of all groups (3 genes per group) , with duplicates removed.

    .. jupyter-execute::

        cl.heatmap(data, keys=cl.marker_genes(data, n_genes=3), group_by="louvain")
    """
    per_group = _marker_names_per_group(
        _container(data).select_modality(modality), groups, key=key, n_genes=n_genes
    )

    # Flatten group by group, dropping any gene already seen so the list can
    # feed `keys` without producing duplicate panels.
    seen: set[str] = set()
    marker_names: list[str] = []
    for group_names in per_group.values():
        for gene_name in group_names:
            if gene_name in seen:
                continue
            seen.add(gene_name)
            marker_names.append(gene_name)

    return marker_names


def marker_genes_dict(
    data: AnnData | MuData,
    groups: Sequence[str] | None = None,
    *,
    modality: str | None = None,
    key: str = "rank_genes_groups",
    n_genes: int = 5,
) -> dict[str, list[str]]:
    """
    Select the top-ranked marker gene names per group from a precomputed ranking.

    Like `marker_genes`, but keeps the grouping: each group maps to its own
    full top-`n_genes` list.

    Parameters
    ----------
    data : AnnData
        The single-cell data object holding the precomputed differential
        expression ranking.
    groups : Sequence[str] | None, default=None
        Subset of groups to pull markers from, in order. `None` keeps all
        groups in their stored order.
    modality : str | None, default=None
        Which modality's stored analysis results to use. Required for a
        multimodal object holding more than one modality, and not accepted
        otherwise.
    key : str, default='rank_genes_groups'
        The key under which the precomputed ranking is stored on `data`.
    n_genes : int, default=5
        Number of top genes to pull per group.

    Returns
    -------
    dict[str, list[str]]
        Mapping from group label to its top-`n_genes` gene names, in group
        order. Each list is complete; a gene that ranks highly in several
        groups appears under each of them.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    KeyNotFoundError
        If the ranking result or a requested group is missing.
    ValueError
        If `n_genes` is out of range.
    TypeError
        If `groups` is neither a Sequence of strings nor None.

    Notes
    -----
    Reads only the gene names, not their scores; use `markers` to plot the
    ranking itself. Use `marker_genes` for a flat, de-duplicated list.

    Examples
    --------
    Inspect the top markers of each group.

    .. jupyter-execute::

        import scanpy as sc

        import cellestial as cl

        data = sc.datasets.pbmc68k_reduced()

        cl.marker_genes_dict(data, n_genes=3)
    """
    return _marker_names_per_group(
        _container(data).select_modality(modality), groups, key=key, n_genes=n_genes
    )
