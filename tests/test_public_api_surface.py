"""
Pin the public API surface ahead of the v1 freeze.

`tests/snapshots/public_api_signatures.json` records every exported callable's
parameters (name, kind, default). A rename, reorder, new default or kind change
fails here, so breaking changes are deliberate. After an intended change,
regenerate the snapshot with `poetry run python tests/test_public_api_surface.py`
and record the change in `CHANGELOG.md`.
"""

from __future__ import annotations

import inspect
import json
from os import PathLike
from pathlib import Path
from typing import Any

import pytest
from ggrust.plot.core import PlotSpec
from ggrust.plot.subplots import SupPlotsSpec

import cellestial as cl
import cellestial.datasets as datasets

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "public_api_signatures.json"

SINGULAR_PLURAL_PAIRS = [
    ("dimensional", "dimensionals"),
    ("umap", "umaps"),
    ("pca", "pcas"),
    ("tsne", "tsnes"),
    ("expression", "expressions"),
    ("violin", "violins"),
    ("boxplot", "boxplots"),
    ("histogram", "histograms"),
    ("ridge", "ridges"),
    ("xyplot", "xyplots"),
    ("spatial", "spatials"),
    ("volcano", "volcanos"),
]

# Parameters only a grid has, and the key parameter that changes number.
GRID_PARAMETERS = {
    "keys",
    "groups",
    "layers",
    "ncol",
    "sharex",
    "sharey",
    "widths",
    "heights",
    "hspace",
    "vspace",
    "fit",
    "align",
    "guides",
    "share_axis",
    "share_ticks",
    "share_labels",
}


def _default_repr(default: object) -> str | None:
    if default is inspect.Parameter.empty:
        return None
    if isinstance(default, PathLike):
        return "<path>"
    return repr(default)


def _signature_rows(function: Any) -> list[list[str | None]]:
    return [
        [parameter.name, parameter.kind.name, _default_repr(parameter.default)]
        for parameter in inspect.signature(function).parameters.values()
    ]


def _current_surface() -> dict[str, dict[str, list[list[str | None]]]]:
    surface: dict[str, dict[str, list[list[str | None]]]] = {"cellestial": {}}
    for name in sorted(cl.__all__):
        member = getattr(cl, name)
        if callable(member) and not isinstance(member, type):
            surface["cellestial"][name] = _signature_rows(member)
    surface["cellestial.datasets"] = {
        name: _signature_rows(getattr(datasets, name)) for name in sorted(datasets.__all__)
    }
    return surface


def _write_snapshot() -> None:
    surface = _current_surface()
    lines = ["{"]
    for module_index, (module, functions) in enumerate(surface.items()):
        lines.append(f"  {json.dumps(module)}: {{")
        for function_index, (name, rows) in enumerate(functions.items()):
            comma = "," if function_index < len(functions) - 1 else ""
            lines.append(f"    {json.dumps(name)}: {json.dumps(rows)}{comma}")
        lines.append("  }" + ("," if module_index < len(surface) - 1 else ""))
    lines.append("}")
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text("\n".join(lines) + "\n")


def test_exported_names_are_pinned():
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    constants = {name for name in cl.__all__ if not callable(getattr(cl, name))}
    assert set(cl.__all__) - constants == set(snapshot["cellestial"])
    assert constants == {
        "BLUE",
        "CHERRY",
        "LIGHT_GRAY",
        "ORANGE",
        "PINK",
        "PURPLE",
        "RED",
        "SNOW",
        "TEAL",
        "__version__",
        "datasets",
    }
    assert set(datasets.__all__) == set(snapshot["cellestial.datasets"])


@pytest.mark.parametrize("module", ["cellestial", "cellestial.datasets"])
def test_signatures_match_snapshot(module):
    snapshot = json.loads(SNAPSHOT_PATH.read_text())[module]
    current = _current_surface()[module]
    changed = sorted(name for name in current if current[name] != snapshot.get(name))
    assert not changed, (
        f"public signatures changed: {changed}. If intended, regenerate the snapshot "
        "and record the change in CHANGELOG.md."
    )


@pytest.mark.parametrize(("singular", "plural"), SINGULAR_PLURAL_PAIRS)
def test_plural_mirrors_singular_parameters(singular, plural):
    singular_parameters = inspect.signature(getattr(cl, singular)).parameters
    plural_parameters = inspect.signature(getattr(cl, plural)).parameters
    shared_singular = [
        name
        for name in singular_parameters
        if name not in GRID_PARAMETERS | {"key", "group", "frame", "geom_kwargs"}
    ]
    shared_plural = [
        name
        for name in plural_parameters
        if name not in GRID_PARAMETERS | {"key", "group", "frame", "geom_kwargs"}
    ]
    assert shared_singular == shared_plural
    for name in shared_singular:
        assert singular_parameters[name].default == plural_parameters[name].default, name


@pytest.mark.parametrize(("singular", "plural"), SINGULAR_PLURAL_PAIRS)
def test_return_annotations(singular, plural):
    assert inspect.signature(getattr(cl, singular)).return_annotation in (PlotSpec, "PlotSpec")
    assert inspect.signature(getattr(cl, plural)).return_annotation in (
        SupPlotsSpec,
        "SupPlotsSpec",
    )


@pytest.mark.parametrize(
    "plural",
    ["dimensionals", "umaps", "expressions", "violins", "boxplots", "histograms", "ridges"],
)
def test_plural_rejects_a_single_string(adata, group_key, plural):
    arguments = (adata, group_key, group_key) if plural == "ridges" else (adata, group_key)
    with pytest.raises(TypeError):
        getattr(cl, plural)(*arguments)


if __name__ == "__main__":
    _write_snapshot()
