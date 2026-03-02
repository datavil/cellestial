import os
import sys
from datetime import datetime

# Add the project root to sys.path so sphinx-autoapi/autodoc can find 'cellestial'
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'cellestial'
copyright = f'{datetime.now().year}, Zaf4'
author = 'Zaf4'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'autoapi.extension',
]

# AutoAPI configuration
autoapi_dirs = ['../cellestial']
autoapi_type = 'python'
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'  # A modern, clean theme
html_static_path = ['_static']
html_title = 'cellestial documentation'

# Napoleon settings for NumPy/Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
