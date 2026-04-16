"""Ad-hoc benchmark: Lets-Plot render-time speedup experiments for cellestial.

Tests a handful of strategies to close the render-time gap vs scanpy on
50k+ cell UMAPs, all on the same local dataset.

Variants
--------
- baseline         : current benchmark setup (sampling='none', SVG)
- default_sampling : let Lets-Plot auto-sample (~50k cap)
- sampled_30k      : force sampling='pick', n=30000
- coords_round3    : round X_umap coords to 3 decimals, then plot full set
- subsample_30k    : subsample AnnData to 30k cells before plotting
- png_render       : same as baseline but render to PNG instead of SVG
- combined         : subsample_30k + coords_round3 + png_render

Usage
-----
    poetry run python -m benchmarks.bench_letsplot_tweaks --n-cells 50000
"""

from __future__ import annotations

import argparse
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np
from anndata import AnnData

import cellestial as cl
from benchmarks.datasets import load_breast_cancer_atlas


@dataclass
class Variant:
    name: str
    prepare: "callable"
    construct: "callable"
    render: "callable"


@dataclass
class Result:
    variant: str
    construct_s: float
    render_s: float
    total_s: float
    notes: str = ""


def _construct_default(adata: AnnData, ctx: dict):
    return cl.expression(adata, key=ctx["gene"], sampling="none")


def _construct_default_sampling(adata: AnnData, ctx: dict):
    return cl.expression(adata, key=ctx["gene"])


def _construct_pick_30k(adata: AnnData, ctx: dict):
    return cl.expression(adata, key=ctx["gene"], sampling="pick", n=30000)


def _prepare_round_coords(adata: AnnData, ctx: dict) -> AnnData:
    new = adata.copy()
    new.obsm["X_umap"] = np.round(new.obsm["X_umap"].astype(np.float32), 3)
    return new


def _prepare_subsample_30k(adata: AnnData, ctx: dict) -> AnnData:
    if adata.n_obs <= 30000:
        return adata
    rng = np.random.default_rng(0)
    indices = rng.choice(adata.n_obs, size=30000, replace=False)
    indices.sort()
    sub = adata[indices]
    return sub.to_memory() if hasattr(sub, "to_memory") else sub.copy()


def _prepare_combined(adata: AnnData, ctx: dict) -> AnnData:
    sub = _prepare_subsample_30k(adata, ctx)
    sub.obsm["X_umap"] = np.round(sub.obsm["X_umap"].astype(np.float32), 3)
    return sub


def _render_svg(plot) -> int:
    svg = plot.to_svg()
    return len(svg) if isinstance(svg, str) else 0


def _render_png(plot) -> int:
    buf = io.BytesIO()
    try:
        plot.to_png(buf)
    except AttributeError:
        from lets_plot import ggsave

        out = ggsave(plot, "tmp.png", path=str(Path("/tmp")))
        return Path(out).stat().st_size
    return buf.tell()


VARIANTS: list[Variant] = [
    Variant("baseline", lambda a, c: a, _construct_default, _render_svg),
    Variant("default_sampling", lambda a, c: a, _construct_default_sampling, _render_svg),
    Variant("sampled_30k", lambda a, c: a, _construct_pick_30k, _render_svg),
    Variant("coords_round3", _prepare_round_coords, _construct_default, _render_svg),
    Variant("subsample_30k", _prepare_subsample_30k, _construct_default, _render_svg),
    Variant("png_render", lambda a, c: a, _construct_default, _render_png),
    Variant("combined", _prepare_combined, _construct_default, _render_png),
]


def time_once(variant: Variant, adata: AnnData, ctx: dict) -> Result:
    prepared = variant.prepare(adata, ctx)
    t0 = time.perf_counter()
    plot = variant.construct(prepared, ctx)
    t1 = time.perf_counter()
    payload_size = variant.render(plot)
    t2 = time.perf_counter()
    return Result(
        variant=variant.name,
        construct_s=t1 - t0,
        render_s=t2 - t1,
        total_s=t2 - t0,
        notes=f"payload~{payload_size/1e6:.1f}MB" if payload_size else "",
    )


def time_repeated(
    variant: Variant, adata: AnnData, ctx: dict, *, repeats: int, warmup: int
) -> Result:
    for _ in range(warmup):
        time_once(variant, adata, ctx)
    results = [time_once(variant, adata, ctx) for _ in range(repeats)]
    return Result(
        variant=variant.name,
        construct_s=median(result.construct_s for result in results),
        render_s=median(result.render_s for result in results),
        total_s=median(result.total_s for result in results),
        notes=results[-1].notes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[variant.name for variant in VARIANTS],
        choices=[variant.name for variant in VARIANTS],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_breast_cancer_atlas(subsample=args.n_cells)
    ctx = {"gene": spec.gene_keys[0]}
    print(f"dataset={spec.name} n_cells={spec.n_cells} gene={ctx['gene']}")
    print(f"{'variant':<20s}  {'construct':>10s}  {'render':>10s}  {'total':>10s}  notes")
    chosen = [variant for variant in VARIANTS if variant.name in args.variants]
    for variant in chosen:
        try:
            result = time_repeated(
                variant, spec.adata, ctx, repeats=args.repeats, warmup=args.warmup
            )
            print(
                f"{result.variant:<20s}  {result.construct_s*1000:>8.1f}ms  "
                f"{result.render_s*1000:>8.1f}ms  {result.total_s*1000:>8.1f}ms  {result.notes}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{variant.name:<20s}  FAILED: {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
