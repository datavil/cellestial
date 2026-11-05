"""
Reproduce Zamboni et al. 2025 (Nat Neurosci) Fig. 3b (RNA) and 3e (ATAC).

The paper's data (GEO GSE304349, GSE304196, GSE304399) is raw multiome
sequencing, so this builds a synthetic stand-in: 5 cell types x 5 time points,
features in modules up-regulated in one cell type with one of three temporal
dynamics, z-scored per feature.

Run:
    poetry run python scripts/zamboni_fig3.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from anndata import AnnData
from ggrust import element_rect, ggsave, scale_fill_gradientn, scale_fill_viridis, theme

import cellestial as cl

CELL_TYPES = ["Ependymal", "Astrocyte", "OPC", "Oligodendrocyte", "Microglia"]
CELL_TYPE_COLORS = {
    "Ependymal": "#7b2a8c",
    "Astrocyte": "#ef5a6a",
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
CELLS_PER_SAMPLE = 40
OUTPUT = Path(__file__).resolve().parent.parent / "figures"


def synthetic_modality(
    module_sizes: dict[str, dict[str, int]], *, prefix: str, noise: float, seed: int
) -> AnnData:
    """
    Simulate one modality as cells x features with module block structure.

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
        Cells ordered by cell type, then time point; features ordered by
        module. Values are z-scored per feature.
    """
    rng = np.random.default_rng(seed)
    observations = pl.DataFrame(
        [
            {"cell_type": cell_type, "time_point": time_point}
            for cell_type in CELL_TYPES
            for time_point in TIME_POINTS
            for _ in range(CELLS_PER_SAMPLE)
        ]
    )
    cell_type_index = observations["cell_type"].replace_strict(
        {name: index for index, name in enumerate(CELL_TYPES)}
    )
    time_index = observations["time_point"].replace_strict(
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
                {"module": f"{cell_type} {dynamics}", "dynamics": dynamics} for _ in range(size)
            )
    matrix = np.hstack(columns)
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    variable_frame = pl.DataFrame(variables).with_columns(
        (pl.lit(prefix) + pl.int_range(pl.len()).cast(pl.String)).alias("name")
    )
    return AnnData(
        X=matrix.astype(np.float32),
        obs=observations.to_pandas().astype("category"),
        var=variable_frame.to_pandas().set_index("name").astype("category"),
    )


def panel(data: AnnData, *, palette) -> object:
    """Build one Fig. 3 heatmap panel from a synthetic modality."""
    data.obs.index = data.obs.index.astype(str)
    return cl.annotated_heatmap(
        data,
        keys=list(data.var_names),
        group_by="cell_type",
        variable_group_by="module",
        transpose=True,
        row_annotations=["cell_type", "time_point"],
        column_annotations=["dynamics"],
        column_annotation_position="bottom",
        annotation_colors={
            "cell_type": CELL_TYPE_COLORS,
            "time_point": TIME_POINT_COLORS,
            "dynamics": DYNAMICS_COLORS,
        },
        # One bin per cell type x time point sample (pseudobulk-like columns).
        max_rows=len(CELL_TYPES) * len(TIME_POINTS),
        group_lines_color="black",
        group_lines_size=0.8,
        layers=[palette, theme(panel_border=element_rect(color="black", size=1.2))],
    )


def main() -> None:
    """Render both panels to `figures/`."""
    OUTPUT.mkdir(exist_ok=True)
    rna = synthetic_modality(
        {
            "Ependymal": {"Early": 40, "Sustained": 70, "Late": 25},
            "Astrocyte": {"Early": 45, "Sustained": 40},
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
            "Ependymal": {"Early": 60, "Sustained": 90, "Late": 15},
            "Astrocyte": {"Early": 70, "Sustained": 90, "Late": 80},
            "OPC": {"Early": 5, "Sustained": 25, "Late": 10},
            "Oligodendrocyte": {"Early": 12, "Sustained": 6, "Late": 10},
            "Microglia": {"Early": 8, "Sustained": 110, "Late": 25},
        },
        prefix="Region",
        noise=0.8,
        seed=1,
    )
    magma = scale_fill_viridis(option="magma", name="RNA")
    teal = scale_fill_gradientn(
        colors=["#0b0b2b", "#2b3a7a", "#2f6c96", "#33a0a8", "#9ee6c8"], name="ATAC"
    )
    ggsave(panel(rna, palette=magma), str(OUTPUT / "zamboni_fig3b.png"), w=5, h=7, unit="in")
    ggsave(panel(atac, palette=teal), str(OUTPUT / "zamboni_fig3e.png"), w=5, h=7, unit="in")


if __name__ == "__main__":
    main()
