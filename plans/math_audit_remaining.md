# Math Audit: Remaining Fixes

> **Status: planned.** Findings 1 to 6 of the 2026-08-05 audit of mathematical
> transformations are fixed and shipped in `## [Unreleased]`; see `CHANGELOG.md`
> and the regression tests in `tests/test_audit_regressions.py`. The four items
> below are what is left. Numbering follows the original audit so it maps to the
> report.
>
> Everything here was reproduced against real data (pbmc3k, shapely 2.1.2,
> scanpy). The proposed APIs were checked to exist before being written down.

Ordered by impact: 7 changes what half of all bracket labels say, 10 draws wrong
geometry silently, 9 is an API that ignores its own argument, 8 needs a product
decision before any code.

---

## 7. `p = 0` is printed as a value

**Problem.** `scipy` returns a flat `0.0` once a p-value underflows double
precision. No test yields a true zero, but with the default `prefix_style="="`
the label is built as `f"{prefix} = {pvalue:.3g}"`, and `format(0.0, ".3g")` is
`"0"`, so the bracket asserts something false.

**Evidence.** Not an edge case. Three of six pairwise comparisons on each of
CD3D, MS4A1 and LYZ (pbmc3k, `fill="cell_type_lvl1"`) underflow:

```
CD3D   pairs=6  p==0: 3   0<p<eps: 0
MS4A1  pairs=6  p==0: 3   0<p<eps: 0
LYZ    pairs=6  p==0: 3   0<p<eps: 0
```

The second column matters: nothing lands between zero and double eps, so any
floor in that range flags exactly the same comparisons and only the printed
number differs.

`prefix_style="<"` does not cover this. It ignores the magnitude entirely and
buckets every p-value into the star thresholds, so a true zero and a `5.31e-12`
both render as `p < 0.0001`.

**Fix.** A floor, applied in one place.

Module constant in `cellestial/layers/bracket.py`:

```python
# scipy returns a flat 0.0 once a p-value underflows double precision; no test
# yields a true zero. R's `format.pval` reports anything under double eps as
# below it.
_PVALUE_FLOOR = float(np.finfo(np.float64).eps)
```

New parameter `pvalue_floor: float = _PVALUE_FLOOR` on `bracket()` and
`_compute_bracket_frame()`, consumed at the top of `_format_pvalue`:

```python
def _format_pvalue(pvalue: float) -> str:
    # Below the floor the value is reported as under it rather than printed:
    # an underflowed 0.0 is not a measurement, and "= 0" claims one.
    underflowed = pvalue < pvalue_floor
    if underflowed:
        pvalue = pvalue_floor
    if prefix_style == "<":
        ...unchanged...
    if prefix_style == "=":
        if not prefix and not underflowed:
            return f"{pvalue:{label_format}}"
        return f"{prefix} {'<' if underflowed else '='} {pvalue:{label_format}}".lstrip()
    if prefix_style is None:
        return f"{prefix}{pvalue:{label_format}}"
```

Result:

```
Lymphocytes vs Monocytes    p = 0        ->  p < 2.22e-16
Monocytes   vs B Cells      p = 5.31e-12 ->  p = 5.31e-12   (unchanged)
```

**Properties.** Labels that do not underflow come out byte-identical, so this is
not a formatting change for anyone. The `prefix_style=None` branch only clamps
the number, since its contract says the caller owns the symbol. Stars are
untouched: `p == 0` already yields `****` correctly.

**The one judgment call: the default.** Double eps prints `p < 2.22e-16`,
matching R's `format.pval` and therefore ggpubr, which `geom_bracket` descends
from. The tightest true bound is the smallest subnormal, `p < 5e-324`, which is
strictly more correct and looks absurd on a figure. Since nothing on real data
falls between them, familiarity is the only thing left to decide on, so take
eps. Anyone wanting `p < 1e-100` sets the parameter.

**Scope.** `cellestial/layers/bracket.py` only. Tests: an underflowed label, an
unaffected label, a custom floor. Changelog under `### Fixed`, since it corrects
a false statement rather than changing an API.

---

## 8. Volcano inherits scanpy's log fold change artifact

**Problem.** `cl.volcano` plots every feature in the ranking. scanpy computes
`logfoldchanges` as `log2((expm1(mean_group) + 1e-9) / (expm1(mean_rest) + 1e-9))`,
and for features near zero in both groups that pseudocount dominates, producing
huge fold changes from nothing.

**Evidence.** pbmc3k, B Cells vs rest:

```
features:                    23427
median logFC (all features): -6.7
called "down":               20651  (88%)
  of those, expressed in <1% of cells:  8781
```

The arithmetic is faithful to what scanpy stored. The plot is still dominated by
features carrying no signal.

**Fix.** Filter on detection rate, which scanpy already computes on request.
`sc.tl.rank_genes_groups(..., pts=True)` writes
`adata.uns[key]["pts"]`, a `DataFrame` indexed by var name with one column per
group (verified in the installed scanpy).

Add `min_fraction_expressed: float | None = None` to `volcano` and `volcanos`,
forwarded to `_build_volcano_frame`. When set:

- read `pts` from the stored result and drop features whose value for `group` is
  below it;
- when `pts` is absent, raise naming the fix, e.g. recompute with detection
  rates recorded, rather than silently plotting everything;
- when `group_by` triggers the internal recompute, pass the flag that records
  them so the filter works without the user precomputing.

Default `None` keeps current behaviour, so this is additive.

**Open question before implementing.** Whether the default should stay `None`.
A default of, say, `0.1` would make the out-of-the-box volcano far more useful,
but it silently drops rows and forces a recompute for anyone whose stored
ranking lacks `pts`. Recommend shipping `None` first and revisiting.

**Scope.** `cellestial/single/differential/volcano.py`,
`cellestial/single/differential/utilities.py`. Changelog under `### Added`.

---

## 9. `spatial(scale_axis=...)` ignores which axis it was given

**Problem.** The parameter is typed `Literal[0, 1] | None`, but both values do
the same thing. `cellestial/spatial/spatial.py:409` min-max scales the single
`key` column globally; there is no partition to choose, because a spatial plot
has one value column per plot. Contrast `heatmap`, where `scale_axis` genuinely
selects the partition key. Same in `spatials`.

Nothing is computed wrongly. The API just documents a choice it does not honour.

**Fix (recommended).** Replace it with a boolean, since only one axis exists:

```python
scale: bool = False
```

Breaking. Migration: `scale_axis=0` or `scale_axis=1` becomes `scale=True`. No
collision, `spatial` currently has `greyscale` and `norm` but no `scale`.

**Alternative, lower churn.** Keep the name and reject the meaningless value:
type it `Literal[0] | None` and raise on `1`. This preserves `scale_axis=0` for
existing callers and keeps the vocabulary aligned with `heatmap`, at the cost of
an odd-looking single-member `Literal`.

**Scope.** `cellestial/spatial/spatial.py`, `cellestial/spatial/spatials.py`.
Changelog under `### Breaking` either way, since `scale_axis=1` stops being
accepted.

---

## 10. Polygon holes are merged into the exterior ring

**Problem.** `_polygon_vertex_frame` in `cellestial/spatial/utilities.py:191`
calls `shapely.get_coordinates(geoms)`, which concatenates interior ring
vertices onto the exterior ring, and counts them with `count_coordinates`. Both
rings land under one `instance_id`, so `geom_polygon` draws a single path that
walks out to the hole and back. Only Polygons with holes are affected.

**Evidence.** A 4-vertex square with one square hole plus a plain triangle:

```
current  shapely.get_coordinates(geoms)  -> 14 vertices  (square 5 + hole 5 + triangle 4)
fixed    exterior rings only             ->  9 vertices  (square 5 + triangle 4)
```

**Fix.** Take exterior rings explicitly. Both calls are vectorised in shapely
2.x and were verified against the installed 2.1.2:

```python
rings = shapely.get_exterior_ring(geoms)
coordinates = shapely.get_coordinates(rings)
per_geom_counts = shapely.get_num_coordinates(rings)
```

This also removes the `np.fromiter` loop over `count_coordinates`, so it is
faster on segmentation masks with many cells.

Holes are dropped rather than rendered, because `geom_polygon` has no hole
support and emitting them as separate polygons would fill them in on top, which
is worse than ignoring them. Worth a sentence in the docstring.

**Scope.** `cellestial/spatial/utilities.py` only. Test: a holed polygon yields
exterior vertices only, and a plain polygon is unchanged. Changelog under
`### Fixed`.
