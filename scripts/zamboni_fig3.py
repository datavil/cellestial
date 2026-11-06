"""
Reproduce Zamboni et al. 2025 (Nat Neurosci) Fig. 3b (RNA) and 3e (ATAC).

The paper's data (GEO GSE304349, GSE304196, GSE304399) is raw multiome
sequencing, so this builds a synthetic stand-in: 5 cell types x 5 time points,
features in modules up-regulated in one cell type with one of three temporal
dynamics. Cells are averaged into one pseudobulk per cell type x time point
sample and z-scored per feature.

Run:
    poetry run python scripts/zamboni_fig3.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from anndata import AnnData
from ggrust import (
    element_blank,
    element_text,
    geom_rect,
    ggsave,
    scale_fill_gradientn,
    theme,
)
from ggrust.plot.core import PlotSpec

import cellestial as cl

CELL_TYPES = ["Astrocyte", "Ependymal", "OPC", "Oligodendrocyte", "Microglia"]
CELL_TYPE_COLORS = {
    "Astrocyte": "#7b2a8c",
    "Ependymal": "#ef5a6a",
    "OPC": "#f8cc14",
    "Oligodendrocyte": "#f39a1e",
    "Microglia": "#2c5aa0",
}
TIME_POINTS = ["Uninjured", "1 dpi", "3 dpi", "7 dpi", "28 dpi"]
TIME_POINT_COLORS = dict(
    zip(TIME_POINTS, ["#f2f2f2", "#c7c7c7", "#969696", "#5e5e5e", "#1f1f1f"], strict=True)
)
# Temporal profile of each module's up-regulation over the time points.
DYNAMICS = {
    "Early": [0.0, 1.0, 0.6, 0.3, 0.1],
    "Sustained": [0.6, 0.8, 1.0, 1.0, 0.9],
    "Late": [0.0, 0.1, 0.4, 1.0, 0.7],
}
DYNAMICS_COLORS = {"Early": "#47a96b", "Sustained": "#e8ebe4", "Late": "#ef7d4a"}
# Genes labelled in Fig. 3b, by the module they sit in there.
PAPER_GENES = {
    "Astrocyte Early": ["Cldn10", "Gjb6", "Slc27a1"],
    "Astrocyte Sustained": ["Stat2", "Ccl2", "Yap1", "Vim", "Cd44", "Gfap"],
    "Astrocyte Late": ["Ncam2", "Stat3"],
    "Ependymal Early": ["Cfap52", "Dnali1", "Foxj1"],
    "Ependymal Sustained": ["Ccnd1", "Nes", "Cd24a", "Btg1"],
    "OPC Early": ["Olig1"],
    "OPC Sustained": ["Cspg4", "Mki67", "Stmn4"],
    "Oligodendrocyte Early": ["Mag"],
    "Microglia Early": ["Cd81", "Csf1r", "Il10ra"],
    "Microglia Sustained": ["Top2a", "Ezh2"],
    "Microglia Late": ["Tyrobp", "Cd274"],
}
# GO terms bracketed beside the Fig. 3b modules.
PAPER_TERMS = {
    "Astrocyte Early": "Cell adhesion\nLipid metabolic process",
    "Astrocyte Sustained": "Cell morphogenesis\nCellular response to stress",
    "Astrocyte Late": "Axon guidance\nCell differentiation",
    "Ependymal Early": "Cilium assembly",
    "Ependymal Sustained": "Ribosome biogenesis\nCell proliferation",
    "OPC Early": "Nervous system development\nIon transport",
    "OPC Sustained": "Cell cycle\nMyelination",
    "Microglia Early": "Immune system process\nInterleukin production",
    "Microglia Sustained": "Cell cycle",
    "Microglia Late": "Integrin-mediated signaling\nMicroglial cell activation",
}
CELLS_PER_SAMPLE = 40
OUTPUT = Path(__file__).resolve().parent.parent / "figures"


def synthetic_modality(
    module_sizes: dict[str, dict[str, int]], *, prefix: str, noise: float, seed: int
) -> AnnData:
    """
    Simulate one modality as pseudobulks x features with module block structure.

    Parameters
    ----------
    module_sizes : dict[str, dict[str, int]]
        Features per module, keyed by enriched cell type, then by dynamics.
    prefix : str
        Feature name prefix.
    noise : float
        Standard deviation of the per-cell Gaussian noise.
    seed : int
        Random seed.

    Returns
    -------
    AnnData
        One pseudobulk (mean over `CELLS_PER_SAMPLE` cells) per sample, ordered
        by cell type, then time point; features ordered by module. Values are
        z-scored per feature.
    """
    rng = np.random.default_rng(seed)
    observations = pl.DataFrame(
        [
            {"Cell type": cell_type, "Time point": time_point}
            for cell_type in CELL_TYPES
            for time_point in TIME_POINTS
            for _ in range(CELLS_PER_SAMPLE)
        ]
    )
    cell_type_index = observations["Cell type"].replace_strict(
        {name: index for index, name in enumerate(CELL_TYPES)}
    )
    time_index = observations["Time point"].replace_strict(
        {name: index for index, name in enumerate(TIME_POINTS)}
    )
    columns, variables = [], []
    for cell_type, sizes in module_sizes.items():
        for dynamics, size in sizes.items():
            profile = np.asarray(DYNAMICS[dynamics])[time_index.to_numpy()]
            in_type = (cell_type_index == CELL_TYPES.index(cell_type)).to_numpy()
            # Weak spill-over into the other cell types keeps blocks realistic.
            signal = np.where(in_type, 1.0 + 1.5 * profile, 0.25 * profile)
            gains = rng.lognormal(0.0, 0.3, size)
            values = signal[:, None] * gains[None, :]
            values += rng.normal(0.0, noise, values.shape)
            columns.append(values)
            variables.extend(
                {"module": f"{cell_type} {dynamics}", "Dynamics": dynamics} for _ in range(size)
            )
    cells = np.hstack(columns)
    # Cells of a sample are contiguous rows: average them into its pseudobulk.
    matrix = cells.reshape(-1, CELLS_PER_SAMPLE, cells.shape[1]).mean(axis=1)
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    observations = observations.gather_every(CELLS_PER_SAMPLE)
    variable_frame = pl.DataFrame(variables).with_columns(
        (pl.lit(prefix) + pl.int_range(pl.len()).cast(pl.String)).alias("name")
    )
    return AnnData(
        X=matrix.astype(np.float32),
        obs=observations.to_pandas().astype("category"),
        var=variable_frame.to_pandas().set_index("name").astype("category"),
    )


def name_genes(data: AnnData, genes: dict[str, list[str]]) -> list[str]:
    """
    Rename synthetic genes after the paper's labelled genes, in place.

    Parameters
    ----------
    data : AnnData
        A synthetic modality with a `module` variable column.
    genes : dict[str, list[str]]
        Gene names per module, spread evenly over the module's features.

    Returns
    -------
    list[str]
        The new gene names, in feature order.
    """
    names = list(data.var_names)
    modules = data.var["module"].astype(str).to_numpy()
    for module, module_genes in genes.items():
        (indices,) = np.nonzero(modules == module)
        picks = np.linspace(0, indices.size - 1, len(module_genes) + 2)[1:-1].round()
        for gene, pick in zip(module_genes, picks.astype(int), strict=True):
            names[indices[pick]] = gene
    data.var_names = names
    return [name for name in names if not name.startswith("Gene")]


def panel(
    data: AnnData,
    *,
    colors: list[str],
    title: str,
    key_labels: list[str] | None = None,
    group_labels: dict[str, str] | None = None,
) -> PlotSpec:
    """Build one Fig. 3 heatmap panel from a synthetic modality."""
    data.obs.index = data.obs.index.astype(str)
    # The color bar's ends are named Min and Max, as in the paper.
    palette = scale_fill_gradientn(
        colors=colors,
        name=title,
        breaks=[float(data.X.min()), float(data.X.max())],
        labels=["Min", "Max"],
    )
    # Thousands of features: label only `key_labels` (in italics), or none.
    # `axis_text_y` also styles the top tracks' titles: italics only with gene labels.
    face = "italic" if key_labels else "plain"
    key_axis = theme(axis_text_y=element_text(face=face, size=9), axis_line_x=element_blank())
    return cl.annotated_heatmap(
        data,
        keys=list(data.var_names),
        group_by="Cell type",
        variable_group_by="module",
        transpose=True,
        row_annotations=["Cell type", "Time point"],
        column_annotations=["Dynamics"],
        # "Cell type" / "Time point" on the right of the top tracks, as in the paper.
        row_track_titles=True,
        column_annotation_position="bottom",
        annotation_colors={
            "Cell type": CELL_TYPE_COLORS,
            "Time point": TIME_POINT_COLORS,
            "Dynamics": DYNAMICS_COLORS,
        },
        annotation_legends=False,
        key_labels=key_labels or [],
        group_labels=group_labels,
        group_labels_width=0.4,
        group_labels_kwargs={"size": 5},
        group_lines_color="black",
        group_lines_size=0.8,
        layers=[
            palette,
            # Frame the cells only: a panel border would also box the label margin.
            geom_rect(
                xmin=-0.5,
                xmax=len(CELL_TYPES) * len(TIME_POINTS) - 0.5,
                ymin=-0.5,
                ymax=data.n_vars - 0.5,
                fill="rgba(0,0,0,0)",
                color="black",
                size=1.2,
                inherit_aes=False,
            ),
            key_axis,
        ],
    )


def main() -> None:
    """Render both panels to `figures/`."""
    OUTPUT.mkdir(exist_ok=True)
    rna = synthetic_modality(
        {
            "Astrocyte": {"Early": 40, "Sustained": 70, "Late": 25},
            "Ependymal": {"Early": 45, "Sustained": 40},
            "OPC": {"Early": 20, "Sustained": 45},
            "Oligodendrocyte": {"Early": 12, "Sustained": 6},
            "Microglia": {"Early": 30, "Sustained": 35, "Late": 30},
        },
        prefix="Gene",
        noise=0.6,
        seed=0,
    )
    atac = synthetic_modality(
        {
            "Astrocyte": {"Early": 60, "Sustained": 90, "Late": 15},
            "Ependymal": {"Early": 70, "Sustained": 90, "Late": 80},
            "OPC": {"Early": 5, "Sustained": 25, "Late": 10},
            "Oligodendrocyte": {"Early": 12, "Sustained": 6, "Late": 10},
            "Microglia": {"Early": 8, "Sustained": 110, "Late": 25},
        },
        prefix="Region",
        noise=0.8,
        seed=1,
    )
    magma = ["#000004", "#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"]
    teal = ["#0b0b2b", "#2b3a7a", "#2f6c96", "#33a0a8", "#9ee6c8"]
    genes = name_genes(rna, PAPER_GENES)
    ggsave(
        panel(
            rna,
            colors=magma,
            title=f"RNA ({rna.n_vars:,} genes)",
            key_labels=genes,
            group_labels=PAPER_TERMS,
        ),
        str(OUTPUT / "zamboni_fig3b.png"),
        w=7,
        h=7,
        unit="in",
    )
    ggsave(
        panel(atac, colors=teal, title=f"ATAC ({atac.n_vars:,} regions)"),
        str(OUTPUT / "zamboni_fig3e.png"),
        w=5,
        h=7,
        unit="in",
    )


if __name__ == "__main__":
    main()
