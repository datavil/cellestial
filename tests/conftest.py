import pytest

import cellestial as cl

# Curated, small set of marker genes that are known to exist in the fixture.
# Kept short so distribution plots stay fast.
MARKERS = ["CD3D", "CD8A", "MS4A1", "NKG7", "CST3", "LYZ"]
GROUP_KEY = "cell_type_lvl1"
CLUSTER_KEY = "leiden"


@pytest.fixture(scope="session")
def adata():
    return cl.datasets.pbmc3k()


@pytest.fixture(scope="session")
def markers():
    return list(MARKERS)


@pytest.fixture(scope="session")
def group_key():
    return GROUP_KEY


@pytest.fixture(scope="session")
def cluster_key():
    return CLUSTER_KEY
