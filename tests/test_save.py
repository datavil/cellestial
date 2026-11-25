"""
Tests for cellestial.save, exporting plots to disk.

PNG and PDF export go through cairo, which is the part most likely to break on
Windows, so those formats are exercised explicitly.
"""

from pathlib import Path

import pytest

import cellestial as cl


@pytest.fixture(scope="module")
def plot(adata):
    return cl.umap(adata, "leiden")


@pytest.mark.parametrize("extension", ["png", "svg", "html", "pdf"])
def test_save_creates_file(plot, tmp_path, extension):
    result = cl.save(plot, f"plot.{extension}", path=str(tmp_path))

    output = Path(result)
    assert output.is_absolute()
    assert output.suffix == f".{extension}"
    assert output.exists()
    assert output.stat().st_size > 0


def test_save_grid_to_png(adata, tmp_path):
    grid = cl.expressions(adata, keys=["CD3D", "MS4A1"])
    result = cl.save(grid, "grid.png", path=str(tmp_path))

    output = Path(result)
    assert output.exists()
    assert output.stat().st_size > 0


def test_save_forwards_ggsave_kwargs(plot, tmp_path):
    from PIL import Image

    result = cl.save(plot, "sized.png", path=str(tmp_path), w=2, h=1, unit="in", dpi=100)

    output = Path(result)
    assert output.parent == tmp_path
    assert output.exists()
    with Image.open(output) as image:
        # w * dpi * scale, with the default scale of 2.0
        assert image.size == (400, 200)


def test_save_unknown_kwarg_raises(plot, tmp_path):
    with pytest.raises(TypeError, match="unexpected keyword argument 'width'"):
        cl.save(plot, "plot.png", path=str(tmp_path), width=4)
