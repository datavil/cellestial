"""Co-expression blend plots with `cl.blend` (issue #75)."""

import polars as pl
import pytest
from ggrust.plot.core import PlotSpec
from ggrust.plot.subplots import SupPlotsSpec

import cellestial as cl
from cellestial.single.core.blend import _blend_color
from cellestial.util import retrieve
from cellestial.util.errors import (
    ConflictingLengthError,
    DuplicateKeysError,
    InvalidColorError,
    UnsupportedDataTypeError,
    VariableNotFoundError,
)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
GREY = (230, 230, 230)


def _blend(first_level: float, second_level: float) -> str:
    frame = pl.DataFrame({"first": [first_level], "second": [second_level]})
    return frame.select(
        _blend_color(
            pl.col("first"), pl.col("second"), low_rgb=GREY, first_rgb=RED, second_rgb=GREEN
        )
    ).item()


@pytest.mark.parametrize(
    ("first_level", "second_level", "expected"),
    [
        (0.0, 0.0, "#e6e6e6"),  # neither: low color
        (1.0, 0.0, "#ff0000"),  # first only
        (0.0, 1.0, "#00ff00"),  # second only
        (1.0, 1.0, "#ffff00"),  # both: yellow
        (0.5, 0.0, "#f27373"),  # halfway to red (242.5 rounds to even)
    ],
)
def test_blend_color_corners(first_level, second_level, expected):
    """The additive blend reaches each color at the corners of the key."""
    assert _blend(first_level, second_level) == expected


def test_blend_returns_four_panels(adata):
    """Feature panels, blend panel and color key, in that order."""
    plot = cl.blend(adata, keys=("CD3E", "CD8A"))
    assert isinstance(plot, SupPlotsSpec)
    assert len(plot.figures) == 4
    assert all(isinstance(figure, PlotSpec) for figure in plot.figures)
    # Feature panels color by the gene.
    assert plot.figures[0].as_dict()["layers"][0]["mapping"]["color"] == "CD3E"
    assert plot.figures[1].as_dict()["layers"][0]["mapping"]["color"] == "CD8A"
    # The key spans every pair of levels.
    key_frame = retrieve(plot, index=3)
    assert key_frame.height == 32 * 32
    assert key_frame["blend_color"].str.contains(r"^#[0-9a-f]{6}$").all()
    plot.to_svg()


def test_blend_colors_follow_expression(adata):
    """Cells with neither gene keep the low color; the top cell of each gene gets its color."""
    plot = cl.blend(adata, keys=("CD3E", "MS4A1"), colors=("#ff0000", "#0000ff"))
    frame = retrieve(plot, index=2)
    assert frame.height == adata.n_obs
    neither = frame.filter(
        (pl.col("CD3E") == pl.col("CD3E").min()) & (pl.col("MS4A1") == pl.col("MS4A1").min())
    )
    assert neither.height > 0
    assert (neither["blend_color"] == "#e6e6e6").all()
    top_b_cell = frame.filter(pl.col("MS4A1") == pl.col("MS4A1").max()).row(0, named=True)
    # Mutually exclusive markers: the top B cell expresses no CD3E, so it is pure blue.
    if top_b_cell["CD3E"] == frame["CD3E"].min():
        assert top_b_cell["blend_color"] == "#0000ff"


def test_coexpressed_pair_blends_more_than_exclusive_pair(adata):
    """CD3E/CD8A share far more yellow-ish cells than CD3E/MS4A1."""

    def coexpressing_cells(second_key: str) -> int:
        frame = retrieve(cl.blend(adata, keys=("CD3E", second_key)), index=2)
        both = (pl.col("CD3E") > pl.col("CD3E").min()) & (
            pl.col(second_key) > pl.col(second_key).min()
        )
        return frame.filter(both).height

    assert coexpressing_cells("CD8A") > 5 * coexpressing_cells("MS4A1")


def test_blend_skips_cells_with_missing_expression(adata):
    """Cells with a non-finite value are left out of every panel alike."""
    data = adata[:200].copy()
    data.X = data.X.toarray() if hasattr(data.X, "toarray") else data.X.copy()
    gene_index = list(data.var_names).index("CD3E")
    data.X[:3, gene_index] = float("nan")
    plot = cl.blend(data, keys=("CD3E", "CD8A"))
    for index in range(3):
        assert retrieve(plot, index=index).height == 197
    assert retrieve(plot, index=2)["blend_color"].null_count() == 0
    plot.to_svg()


def test_blend_options(adata):
    """Other embeddings, hidden tooltips, arrows, layout and interactivity all build."""
    plot = cl.blend(
        adata,
        keys=["CD3E", "CD8A"],
        dimensions="pca",
        tooltips="none",
        axis_type="arrow",
        ncol=2,
        interactive=True,
        color_low="#fff",
    )
    assert isinstance(plot, SupPlotsSpec)
    blend_layer = plot.figures[2].as_dict()["layers"][0]
    assert blend_layer["mapping"]["x"] == "X_PCA1"
    plot.to_svg()


def test_blend_mudata(mudata):
    """Genes from a modality blend on a joint embedding."""
    plot = cl.blend(mudata, keys=("NKG7", "GNLY"), dimensions="wnn_umap")
    assert isinstance(plot, SupPlotsSpec)
    assert retrieve(plot, index=2).height == mudata.n_obs
    plot.to_svg()


@pytest.mark.parametrize("keys", [("CD3E",), ("CD3E", "CD8A", "MS4A1"), "CD3E"])
def test_blend_needs_two_keys(adata, keys):
    """Anything but two keys is rejected."""
    with pytest.raises(ConflictingLengthError, match="exactly two features"):
        cl.blend(adata, keys=keys)


def test_blend_rejects_duplicate_keys(adata):
    """The same feature cannot be blended with itself."""
    with pytest.raises(DuplicateKeysError, match="'CD3E' twice"):
        cl.blend(adata, keys=("CD3E", "CD3E"))


def test_blend_unknown_gene_suggests_close_match(adata):
    """A typo in a gene name suggests the closest name."""
    with pytest.raises(VariableNotFoundError, match=r"'CD3EE'.*Did you mean 'CD3E'"):
        cl.blend(adata, keys=("CD3EE", "CD8A"))


def test_blend_rejects_metadata_key(adata):
    """Only variable (gene) names can be blended."""
    with pytest.raises(VariableNotFoundError, match="not a variable"):
        cl.blend(adata, keys=("leiden", "CD8A"))


@pytest.mark.parametrize(
    ("options", "parameter"),
    [
        ({"colors": ("red", "#00ff00")}, "colors"),
        ({"colors": ("#ff0000", "#00ff0")}, "colors"),
        ({"color_low": "lightgrey"}, "color_low"),
    ],
)
def test_blend_rejects_non_hex_colors(adata, options, parameter):
    """Colors must be hex strings to be mixed."""
    with pytest.raises(InvalidColorError, match=f"`{parameter}`.*hex"):
        cl.blend(adata, keys=("CD3E", "CD8A"), **options)


def test_blend_needs_two_colors(adata):
    """One color per key is required."""
    with pytest.raises(ConflictingLengthError, match="exactly two colors"):
        cl.blend(adata, keys=("CD3E", "CD8A"), colors=("#ff0000",))


def test_blend_rejects_unsupported_data():
    """Data other than AnnData or MuData is rejected."""
    with pytest.raises(UnsupportedDataTypeError):
        cl.blend(pl.DataFrame({"a": [1]}), keys=("a", "b"))  # ty:ignore[invalid-argument-type]
