"""IO helpers: SVG paths, feather writers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = ROOT / "figures"
SVG_DIR = ROOT / "svg"


def svg_path(case: str, dataset: str, param_slug: str, library: str, replica: int) -> Path:
    """Return the SVG output path; create parent dirs."""
    directory = SVG_DIR / case / dataset / param_slug / library
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"replica_{replica}.svg"


def write_results(frame: "pl.DataFrame", *, snapshot_label: str | None = None) -> Path:
    """
    Write the consolidated results.feather and a timestamped snapshot.

    Returns the path of the consolidated file.
    """
    import polars as pl

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Snapshot FIRST so the run's data is durable even if the merge step blows up.
    label = snapshot_label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = RUNS_DIR / f"run_{label}.feather"
    frame.write_ipc(snapshot)

    consolidated = RESULTS_DIR / "results.feather"
    merged = frame
    if consolidated.exists():
        try:
            existing = pl.read_ipc(consolidated)
            merged = pl.concat([existing, frame], how="vertical_relaxed")
        except (OSError, pl.exceptions.PolarsError) as exc:
            backup = consolidated.with_suffix(".corrupt.feather")
            consolidated.rename(backup)
            print(
                f"[benchmarks.io] existing {consolidated.name} unreadable "
                f"({exc.__class__.__name__}); moved to {backup.name} and starting fresh"
            )

    partial = consolidated.with_name(consolidated.name + ".part")
    merged.write_ipc(partial)
    partial.replace(consolidated)

    return consolidated


def rebuild_consolidated_from_snapshots() -> Path | None:
    """
    Rebuild `results.feather` by concatenating every snapshot under `runs/`.

    Useful when the consolidated file is corrupt or you want to fold in
    snapshots from past interrupted runs. Returns the consolidated path, or
    None if no snapshots exist.
    """
    import polars as pl

    snapshots = sorted(RUNS_DIR.glob("run_*.feather"))
    if not snapshots:
        return None
    frames = [pl.read_ipc(path) for path in snapshots]
    merged = pl.concat(frames, how="vertical_relaxed")
    consolidated = RESULTS_DIR / "results.feather"
    partial = consolidated.with_name(consolidated.name + ".part")
    merged.write_ipc(partial)
    partial.replace(consolidated)
    return consolidated
