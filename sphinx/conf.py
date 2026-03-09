import os
import sys
import unittest.mock as mock
from datetime import datetime
from importlib.metadata import version as get_version

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(".."))  # noqa: PTH100
templates_path = [os.path.join(os.path.abspath("."), "_templates")]  # noqa: PTH100, PTH118

# cellestial must be installed in the env
version = get_version("cellestial")
release = get_version("cellestial")
rst_prolog = """
.. role:: gray
   :class: gray
"""
rst_epilog = f".. |version| replace:: :gray:`v{release}`"


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
# BetterMock.__bool__ = lambda self: True

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

# -- Project information -----------------------------------------------------

project = "cellestial"
copyright = f"{datetime.now().year}, datavil"
author = "Zaf4"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "jupyter_sphinx",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_design",
]


autosummary_generate = True
add_module_names = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_preserve_defaults = True
autodoc_typehints_format = "short"


# -- Options for HTML output -------------------------------------------------
html_show_sourcelink = False
html_title = ""
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

html_theme_options = {
    "logo": {
        "text": "Cellestial",
    },
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
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "search_bar_text": "Search",
    "pygments_light_style": "manni",
    "pygments_dark_style": "material",
    "show_nav_level": 0,
    "use_edit_page_button": False,
    "announcement": None,
    "show_prev_next": False,
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
}
html_favicon = "_static/datavil.svg"

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
napoleon_type_aliases = {
    "PlotSpec" : ":py:class:`lets_plot.plot.core.PlotSpec`",
    "FeatureSpec" : ":py:class:`lets_plot.plot.core.FeatureSpec`",
    "LayerSpec" : ":py:class:`lets_plot.plot.core.LayerSpec`",
    "SupPlotsSpec" : ":py:class:`lets_plot.plot.core.SupPlotsSpec`",
}

napoleon_type_aliases = {
    "PlotSpec": ":py:class:`~lets_plot.plot.core.PlotSpec`",
    "DataFrame": ":doc:`polars:reference/dataframe/index`",
    "AnnData" : ":py:class:`~anndata.AnnData`",
}

python_use_unqualified_type_names = True


intersphinx_mapping = {
    "lets_plot": ("https://lets-plot.org/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
}
