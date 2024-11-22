import shutil
from pathlib import Path


def clean():
    """Cleans the project directory."""
    root = Path("../").resolve()

    for pycache in root.rglob("__pycache__"):
        if pycache.is_dir() and ".venv" not in pycache.parts:
            print(f"Removing {pycache}")
            shutil.rmtree(pycache)

    shutil.rmtree(root / "dist", ignore_errors=True)
    shutil.rmtree(root / ".ruff_cache", ignore_errors=True)
    shutil.rmtree(root / ".mypy_cache", ignore_errors=True)

    return


if __name__ == "__main__":
    clean()
