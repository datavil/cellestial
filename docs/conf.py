import os
import sys
from datetime import datetime
import unittest.mock as mock

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath('..'))

# Monkey-patch importlib.metadata.version to avoid PackageNotFoundError during build
import importlib.metadata
_original_version = importlib.metadata.version
def _mock_version(package_name):
    if package_name == "cellestial":
        return "0.10.3"
    return _original_version(package_name)
importlib.metadata.version = _mock_version

# Advanced Mocking to handle lets-plot additions and subpackages
class BetterMock(mock.MagicMock):
    def __add__(self, other): return self
    def __radd__(self, other): return self
    def __call__(self, *args, **kwargs): return self
    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattr__(name)
        return self

mock_modules = [
    "anndata", "lets_plot", "lets_plot.plot", "lets_plot.plot.core", 
    "lets_plot.plot.subplots", "polars", "skimage", "scikit-image", 
    "ipykernel", "ruff", "mypy", "pooch", "scanpy", "pycairo", 
    "cairosvg", "scvelo", "ty", "pandas", "tqdm", "pyarrow", 
    "matplotlib", "seaborn", "numpy", "scipy", "sklearn", "IPython"
]

for mod_name in mock_modules:
    sys.modules[mod_name] = BetterMock()

# -- Project information -----------------------------------------------------

project = 'cellestial'
copyright = f'{datetime.now().year}, Zaf4'
author = 'Zaf4'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

autosummary_generate = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': False,
    'member-order': 'bysource',
}

add_module_names = False

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_title = 'cellestial documentation'
html_static_path = ['_static']
html_css_files = ['custom.css']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
