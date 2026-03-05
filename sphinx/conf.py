import os
import sys
from datetime import datetime
import unittest.mock as mock

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(".."))
templates_path = [os.path.join(os.path.abspath("."), "_templates")]

# Monkey-patch importlib.metadata.version to avoid PackageNotFoundError during build
import importlib.metadata

_original_version = importlib.metadata.version


def _mock_version(package_name):
    if package_name == "cellestial":
        return "0.10.3"
    return _original_version(package_name)


importlib.metadata.version = _mock_version


class BetterMock(mock.MagicMock):
    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __bool__(self):
        return True

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self


# Explicitly set __bool__ on the class to prevent MagicMock from overriding it
BetterMock.__bool__ = lambda self: True

# Only mock lets_plot manually, let autodoc_mock_imports handle the rest
for mod_name in ["lets_plot", "lets_plot.plot", "lets_plot.plot.core", "lets_plot.plot.subplots"]:
    sys.modules[mod_name] = BetterMock()

autodoc_mock_imports = [
    "anndata",
    "polars",
    "skimage",
    "scipy",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "tqdm",
    "pyarrow",
    "scanpy",
    "scvelo",
]
"""
# Advanced Mocking to handle lets-plot additions and subpackages
class BetterMock(mock.MagicMock):
    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattr__(name)
        return self


mock_modules = [
    "anndata",
    "lets_plot",
    "lets_plot.plot",
    "lets_plot.plot.core",
    "lets_plot.plot.subplots",
    "polars",
    "skimage",
    "scikit-image",
    "ipykernel",
    "ruff",
    "mypy",
    "pooch",
    "scanpy",
    "pycairo",
    "cairosvg",
    "scvelo",
    "ty",
    "pandas",
    "tqdm",
    "pyarrow",
    "matplotlib",
    "seaborn",
    "numpy",
    "scipy",
    "sklearn",
]

for mod_name in mock_modules:
    sys.modules[mod_name] = BetterMock()

"""
autodoc_mock_imports = [
    "anndata",
    "lets_plot",
    "polars",
    "skimage",
    "scipy",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "tqdm",
    "pyarrow",
    "scanpy",
    "scvelo",
]

# -- Project information -----------------------------------------------------

project = "cellestial"
copyright = f"{datetime.now().year}, datavil"
author = "Zaf4"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    # "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "jupyter_sphinx",
    "sphinx.ext.githubpages",
]


autosummary_generate = True
add_module_names = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": False,
    "member-order": "bysource",
}
autodoc_preserve_defaults = True

# -- Options for HTML output -------------------------------------------------
html_show_sourcelink = False
html_title = ""
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_theme_options = {
    "logo": {"text": "cellestial"},
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/datavil/cellestial",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/cellestial",
            "icon": "_static/pypi.svg",
            "type": "local",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "search_bar_text": "Search",
    "pygments_light_style": "manni",
    "pygments_dark_style": "monokai",
    "show_nav_level": 0,
    "use_edit_page_button": False,
    "announcement": None,
    "show_prev_next": False,
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
}


html_sidebars = {
    "**": [],  # no search, links, etc. on any page
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True


def setup(app):
    # This can be used for custom CSS or JS if needed
    pass
