import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

_CACHE_ROOT = Path(tempfile.mkdtemp(prefix="cellestial-pytest-"))
for _name, _path in {
    "NUMBA_CACHE_DIR": _CACHE_ROOT / "numba",
    "MPLCONFIGDIR": _CACHE_ROOT / "matplotlib",
    "XDG_CACHE_HOME": _CACHE_ROOT,
}.items():
    _path.mkdir(parents=True, exist_ok=True)
    os.environ[_name] = str(_path)
atexit.register(shutil.rmtree, _CACHE_ROOT, ignore_errors=True)

import cellestial as cl  # noqa: E402

# Curated, small set of marker genes that are known to exist in the fixture.
# Kept short so distribution plots stay fast.
MARKERS = ["CD3D", "CD8A", "MS4A1", "NKG7", "CST3", "LYZ"]
GROUP_KEY = "cell_type_lvl1"
CLUSTER_KEY = "leiden"


@pytest.fixture(scope="session")
def adata():
    return cl.datasets.pbmc3k()


@pytest.fixture(scope="session")
def mudata():
    """Real CITE-seq container: 411 cells, `rna` (27) + `prot` (29) modalities."""
    return cl.datasets.pbmc_cite()


@pytest.fixture(scope="session")
def markers():
    return list(MARKERS)


@pytest.fixture(scope="session")
def group_key():
    return GROUP_KEY


@pytest.fixture(scope="session")
def cluster_key():
    return CLUSTER_KEY
