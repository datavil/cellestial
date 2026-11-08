import polars as pl
import pytest
from ggrust import aes
from ggrust.plot.core import PlotSpec

import cellestial as cl
from cellestial.single.heatmap.tracksplot import _group_order
from cellestial.util.errors import (
    DendrogramMismatchError,
    DuplicateKeysError,
    InvalidArgumentError,
    UnsupportedDataTypeError,
    VariableNotFoundError,
)

KEYS = ["CD3D", "MS4A1", "LYZ", "NKG7"]


def _track_frame(plot: PlotSpec) -> pl.DataFrame:
    return cl.retrieve(plot)


def test_tracksplot_returns_plotspec_and_renders(adata, markers, group_key):
    plot = cl.tracksplot(adata, keys=markers, group_by=group_key)
    assert isinstance(plot, PlotSpec)
    assert plot.to_svg() is not None


@pytest.mark.parametrize("geom", ["bar", "area"])
def test_tracksplot_geoms_render_with_group_fill(adata, group_key, geom):
    plot = cl.tracksplot(
        adata, keys=KEYS, group_by=group_key, geom=geom, mapping=aes(fill=group_key)
    )
    assert plot.to_svg() is not None


def test_tracksplot_tracks_scaled_to_track_height(adata):
    frame = _track_frame(cl.tracksplot(adata, keys=KEYS, group_by="leiden"))
    heights = frame.group_by("variable").agg((pl.col("track_top") - pl.col("track_base")).max())
    assert heights["track_top"].to_list() == pytest.approx([0.85] * len(KEYS))
    # First key is the top track.
    bases = dict(frame.select("variable", "track_base").unique().iter_rows())
    assert bases == {key: float(len(KEYS) - 1 - index) for index, key in enumerate(KEYS)}


def test_tracksplot_group_spans_are_contiguous_and_ordered(adata):
    frame = _track_frame(cl.tracksplot(adata, keys=KEYS, group_by="leiden"))
    spans = (
        frame.group_by("leiden")
        .agg(pl.col("position_x").min().alias("start"), pl.col("position_x").max().alias("end"))
        .with_columns(pl.col("leiden").cast(pl.String).cast(pl.Int64))
        .sort("leiden")
    )
    starts = spans["start"].to_list()
    ends = spans["end"].to_list()
    assert starts[0] == 0
    assert all(start == end + 1 for start, end in zip(starts[1:], ends[:-1], strict=True))


def test_tracksplot_max_rows_bins_cells(adata):
    frame = _track_frame(cl.tracksplot(adata, keys=KEYS, group_by="leiden", max_rows=200))
    assert frame["position_x"].n_unique() <= 200 + frame["leiden"].n_unique()
    unbinned = _track_frame(cl.tracksplot(adata, keys=KEYS, group_by="leiden", max_rows=None))
    assert unbinned["position_x"].n_unique() == unbinned["Barcode"].n_unique()


def test_tracksplot_mapping_keys_and_dendrogram(adata, group_key):
    keys = {"T": ["CD3D", "IL7R"], "B": ["MS4A1"], "Mono": ["LYZ", "CST3"]}
    plot = cl.tracksplot(adata, keys=keys, group_by=group_key, dendrogram=True)
    assert plot.to_svg() is not None
    breaks = next(s for s in plot.as_dict()["scales"] if s.get("aesthetic") == "y")
    assert breaks["labels"] == ["CD3D", "IL7R", "MS4A1", "LYZ", "CST3"]


def test_tracksplot_group_bars_off(adata, group_key):
    plot = cl.tracksplot(adata, keys=KEYS, group_by=group_key, group_bars=False, group_lines=False)
    assert plot.to_svg() is not None


def test_tracksplot_requires_keys_and_group_by(adata):
    with pytest.raises(InvalidArgumentError, match="`keys` and `group_by` are required"):
        cl.tracksplot(adata, keys=KEYS)


def test_tracksplot_rejects_unknown_geom(adata):
    with pytest.raises(InvalidArgumentError, match="'bar' or 'area'"):
        cl.tracksplot(adata, keys=KEYS, group_by="leiden", geom="line")


def test_tracksplot_rejects_unsupported_data():
    with pytest.raises(UnsupportedDataTypeError):
        cl.tracksplot(pl.DataFrame({"a": [1]}), keys=["a"], group_by="a")


def test_tracksplot_unknown_keys(adata):
    with pytest.raises(VariableNotFoundError):
        cl.tracksplot(adata, keys=["NOT_A_GENE"], group_by="leiden")
    with pytest.raises(KeyError, match="leidn"):
        cl.tracksplot(adata, keys=KEYS, group_by="leidn")


def test_group_order_numeric_categorical_and_strings():
    assert _group_order(pl.Series(["10", "2", "1", "2"])) == ["1", "2", "10"]
    categorical = pl.Series(["b", "a", "c"], dtype=pl.Enum(["c", "b", "a", "unused"]))
    assert _group_order(categorical) == ["c", "b", "a"]
    assert _group_order(pl.Series(["b", "a"])) == ["a", "b"]


def test_tracksplot_rejects_duplicate_keys(adata):
    with pytest.raises(DuplicateKeysError, match="CD3D"):
        cl.tracksplot(adata, keys=["CD3D", "CD3D"], group_by="leiden")


def test_tracksplot_stale_dendrogram_raises(adata):
    import scanpy as sc

    full = adata.copy()
    sc.tl.dendrogram(full, groupby="leiden")
    subset = full[full.obs["leiden"].isin(["0", "2", "5"])].copy()
    subset.uns = full.uns
    with pytest.raises(DendrogramMismatchError, match="recompute it"):
        cl.tracksplot(subset, keys=KEYS, group_by="leiden", dendrogram=True)
