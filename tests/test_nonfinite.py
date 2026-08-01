import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData

import cellestial as cl
from cellestial.layers.ondata_legend import _compute_label_positions
from cellestial.layers.outline import _get_density_boundaries
from cellestial.util import retrieve


def _nonfinite_data() -> AnnData:
    data = AnnData(
        X=np.array(
            [
                [1.0, np.nan],
                [np.inf, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
                [6.0, 6.0],
            ]
        ),
        obs=pd.DataFrame(
            {
                "group": pd.Categorical(["a", "a", "b", "b", "b", "b"]),
                "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "tooltip": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            },
            index=[f"cell_{index}" for index in range(6)],
        ),
    )
    data.var_names = ["gene_a", "gene_b"]
    data.obsm["X_umap"] = np.array(
        [[0.0, 0.0], [np.nan, 1.0], [2.0, np.inf], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]
    )
    data.obsm["spatial"] = data.obsm["X_umap"].copy()
    return data


@pytest.mark.parametrize("plotter", [cl.violin, cl.boxplot, cl.histogram])
def test_distributions_silently_filter_only_the_value_column(plotter):
    """Distribution statistics should silently consume only finite values."""
    data = _nonfinite_data()
    frame = pl.DataFrame(
        {
            "gene_a": [1.0, np.nan, np.inf, -np.inf, 2.0],
            "tooltip": [np.nan, 1.0, 2.0, 3.0, np.inf],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        plot = plotter(data, "gene_a", frame=frame, tooltips=["tooltip"])

    plotted = retrieve(plot)
    assert caught == []
    assert plotted["gene_a"].to_list() == [1.0, 2.0]
    assert np.isnan(plotted["tooltip"][0])
    assert np.isinf(plotted["tooltip"][1])


def test_aggregated_plots_use_only_finite_expression():
    """Aggregation should happen after non-finite expression is removed."""
    data = _nonfinite_data()

    dot_frame = retrieve(cl.dotplot(data, ["gene_a", "gene_b"], "group"))
    heatmap_frame = retrieve(
        cl.heatmap(data, ["gene_a", "gene_b"], "group", aggregate=True, geom="tile")
    )

    assert dot_frame["avg_exp"].is_finite().all()
    assert dot_frame["pct_exp"].is_finite().all()
    assert heatmap_frame["value"].is_finite().all()


def test_heatmap_binning_includes_observations_without_finite_tiles():
    """Non-finite cells should not change bin allocation or its row span."""
    finite_data = _nonfinite_data()
    finite_data.X = np.nan_to_num(finite_data.X, nan=0.0, posinf=0.0, neginf=0.0)
    nonfinite_data = finite_data.copy()
    nonfinite_data.X[1, :] = np.nan

    finite_plot = cl.heatmap(
        finite_data,
        ["gene_a", "gene_b"],
        "group",
        aggregate=False,
        geom="tile",
        max_rows=3,
    )
    nonfinite_plot = cl.heatmap(
        nonfinite_data,
        ["gene_a", "gene_b"],
        "group",
        aggregate=False,
        geom="tile",
        max_rows=3,
    )

    finite_y_scale = next(
        scale for scale in finite_plot.as_dict()["scales"] if scale.get("aesthetic") == "y"
    )
    nonfinite_y_scale = next(
        scale for scale in nonfinite_plot.as_dict()["scales"] if scale.get("aesthetic") == "y"
    )
    assert finite_y_scale["limits"] == nonfinite_y_scale["limits"]


def test_coordinate_plots_filter_coordinates_but_not_tooltips():
    """Invalid tooltip values should not remove otherwise valid points."""
    data = _nonfinite_data()
    frame = pl.DataFrame(
        {
            "x": [0.0, np.nan, 2.0, 3.0],
            "y": [0.0, 1.0, np.inf, 3.0],
            "tooltip": [np.nan, 1.0, 2.0, np.inf],
        }
    )

    plotted = retrieve(cl.xyplot(data, "x", "y", frame=frame, tooltips=["tooltip"]))

    assert plotted["x"].to_list() == [0.0, 3.0]
    assert np.isnan(plotted["tooltip"][0])
    assert np.isinf(plotted["tooltip"][1])


def test_dimensional_and_spatial_filter_nonfinite_coordinates():
    """Built-in coordinate plots should retain only finite coordinate pairs."""
    data = _nonfinite_data()

    dimensional_frame = retrieve(cl.umap(data, tooltips="none"))
    spatial_frame = retrieve(cl.spatial(data, image=False, tooltips="none"))

    assert dimensional_frame.select("X_UMAP1", "X_UMAP2").to_numpy().shape == (4, 2)
    assert spatial_frame.select("spatial_x", "spatial_y").to_numpy().shape == (4, 2)


def test_invalid_color_values_do_not_remove_valid_coordinates():
    """Optional color values should not control observation survival."""
    data = _nonfinite_data()
    data.obs["color"] = [0.0, 1.0, 2.0, np.nan, np.inf, 5.0]

    dimensional_frame = retrieve(cl.umap(data, key="color", tooltips="none"))
    spatial_frame = retrieve(cl.spatial(data, key="color", image=False, tooltips="none"))

    assert dimensional_frame.height == 4
    assert spatial_frame.height == 4
    assert dimensional_frame["color"].null_count() == 1
    assert spatial_frame["color"].null_count() == 1
    assert dimensional_frame["color"].is_infinite().any()
    assert spatial_frame["color"].is_infinite().any()


def test_deferred_coordinate_statistics_ignore_nonfinite_rows():
    """Deferred coordinate statistics should receive finite rows only."""
    frame = pl.DataFrame(
        {
            "x": [0.0, 0.2, -0.2, 0.1, -0.1, np.nan, np.inf],
            "y": [0.0, 0.1, 0.1, -0.2, -0.1, 1.0, 2.0],
            "group": ["a"] * 7,
        }
    )

    labels = _compute_label_positions(frame, x="x", y="y", group_by="group")
    boundaries = _get_density_boundaries(frame, "x", "y", "group", "a", grid_size=20)

    assert labels["x"].is_finite().all()
    assert labels["y"].is_finite().all()
    assert boundaries["x"].is_finite().all()
    assert boundaries["y"].is_finite().all()
