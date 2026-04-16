"""Sphinx configuration for FreeRec documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'FreeRec'
copyright = '2024, MTandHJ'
author = 'MTandHJ'

try:
    from importlib.metadata import version as _get_version
    release = _get_version('freerec')
except Exception:
    release = '0.9.7'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinxcontrib.mermaid',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'src']

# -- Napoleon (NumPy-style docstrings) --------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Autodoc -----------------------------------------------------------------

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

autodoc_mock_imports = [
    'torch', 'torchdata', 'torch_geometric',
    'tensorboard', 'sklearn', 'einops',
    'polars', 'pandas', 'matplotlib', 'tqdm',
    'prettytable', 'yaml', 'numpy', 'requests',
]

# -- Autosummary -------------------------------------------------------------

autosummary_generate = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# -- HTML output -------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_logo = '_static/img/logo_pixel.png'

html_theme_options = {
    "logo": {
        "text": "",
    },
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}
