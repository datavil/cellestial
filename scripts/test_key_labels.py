"""Render heatmap variants at multiple scales for visual inspection of key labels."""
from __future__ import annotations

import sys
from pathlib import Path

import scanpy as sc

import cellestial as cl

OUT = Path("/tmp/heatmap_test")
OUT.mkdir(parents=True, exist_ok=True)


def grouped(keys: list[str], n_per_group: int = 2) -> dict[str, list[str]]:
    """Split a flat key list into a name->keys mapping for bracket labels."""
    grouped_keys: dict[str, list[str]] = {}
    for index, start in enumerate(range(0, len(keys), n_per_group)):
        grouped_keys[f"Group_{index + 1}"] = keys[start : start + n_per_group]
    return grouped_keys


def make_markers(n: int) -> list[str]:
    base = [
        "C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ",
        "GNLY", "NKG7", "FCER1A", "MS4A7", "FCGR3A", "PPBP",
        "S100A8", "S100A9", "GZMB", "GZMA", "PRF1", "KLRD1",
        "CCL5", "IL7R", "TCF7", "LEF1", "SELL", "CD3D",
        "CD3E", "CD3G", "FOXP3", "RORC", "IFNG", "TNF",
    ]
    return base[:n]


def render_for(suffix: str, data, group_by: str, keys: list[str]):
    print(f"[{suffix}] rendering with {len(keys)} keys, group_by={group_by!r}")
    flat_keys = keys
    dict_keys = grouped(keys, n_per_group=max(1, len(keys) // 5))
    print(f"  groups: {list(dict_keys.keys())} -> sizes {[len(v) for v in dict_keys.values()]}")

    common = dict(data=data, group_by=group_by)

    plots = [
        ("heatmap_flat", cl.heatmap(keys=flat_keys, **common) + cl.borders()),
        ("heatmap_grouped", cl.heatmap(keys=dict_keys, **common) + cl.borders()),
        ("matrixplot_flat", cl.matrixplot(keys=flat_keys, **common) + cl.borders()),
        ("matrixplot_grouped", cl.matrixplot(keys=dict_keys, **common) + cl.borders()),
        ("dotplot_flat", cl.dotplot(keys=flat_keys, **common) + cl.borders()),
        ("dotplot_grouped", cl.dotplot(keys=dict_keys, **common) + cl.borders()),
        ("stacked_violin_flat", cl.stacked_violin(keys=flat_keys, **common) + cl.borders()),
        ("stacked_violin_grouped", cl.stacked_violin(keys=dict_keys, **common) + cl.borders()),
    ]

    for name, plot in plots:
        filename = f"{name}_{suffix}.png"
        try:
            cl.save(plot, filename, path=str(OUT), w=8, h=6)
            print(f"  wrote {filename}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {filename}: {exc}")


def main():
    print("loading pbmc3k via cl.datasets...")
    data_pbmc3k = cl.datasets.pbmc3k(cache_directory="data")
    print("loading pbmc68k_reduced via scanpy...")
    data_pbmc68k = sc.datasets.pbmc68k_reduced()

    pbmc3k_group_by = "cell_type_lvl1"
    pbmc68k_group_by = "bulk_labels"

    pbmc68k_var_names = list(data_pbmc68k.var_names[:30])

    scales = sys.argv[1:] or ["small", "medium", "large"]
    for scale in scales:
        if scale == "small":
            n = 3
        elif scale == "medium":
            n = 10
        elif scale == "large":
            n = 30
        else:
            raise ValueError(scale)
        render_for(f"pbmc3k_{scale}", data_pbmc3k, pbmc3k_group_by, make_markers(n))
        render_for(f"pbmc68k_{scale}", data_pbmc68k, pbmc68k_group_by, pbmc68k_var_names[:n])


if __name__ == "__main__":
    main()
