"""Sphinx configuration for FreeRec documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Mock heavy dependencies before any freerec import -----------------------
# MagicMock causes metaclass conflicts with classes inheriting from mocked
# bases (e.g. IterDataPipe). We use a custom mock module class instead.

import abc
import types


class _MockMeta(abc.ABCMeta):
    """Metaclass (compatible with ABCMeta) that makes any attribute access
    return the class itself. This lets `torch.nn.Module` resolve to
    _MockBase, and `class Foo(X, metaclass=abc.ABCMeta)` work."""

    def __getattr__(cls, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return cls

    def __getitem__(cls, item):
        return cls


class _MockBase(metaclass=_MockMeta):
    """Universal safe base class. Can be inherited from, attribute-chained, etc."""

    def __init__(self, *args, **kwargs):
        pass

    def __init_subclass__(cls, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        # When used as a decorator, return the decorated thing unchanged
        if args and callable(args[0]):
            return args[0]
        return self


class _MockModule(types.ModuleType):
    """A mock module whose attributes are _MockBase (the universal base)."""

    def __init__(self, name="mock"):
        super().__init__(name)
        self.__all__ = []
        self.__path__ = []
        self.__file__ = ""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _MockBase

    def __call__(self, *args, **kwargs):
        return self


_mock_packages = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.distributed",
    "torch.utils",
    "torch.utils.data",
    "torch.utils.data.graph_settings",
    "torch.utils.data.datapipes",
    "torch.utils.data.datapipes.datapipe",
    "torch.utils.tensorboard",
    "torch.backends",
    "torch.backends.cudnn",
    "torchdata",
    "torchdata.datapipes",
    "torchdata.datapipes.iter",
    "torchdata.dataloader2",
    "torchdata.dataloader2.reading_service",
    "torch_geometric",
    "torch_geometric.utils",
    "torch_geometric.utils.num_nodes",
    "torch_geometric.nn",
    "tensorboard",
    "sklearn",
    "sklearn.metrics",
    "einops",
    "polars",
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "tqdm",
    "prettytable",
    "yaml",
    "numpy",
]

for _pkg in _mock_packages:
    sys.modules[_pkg] = _MockModule(_pkg)

# -- Project information -----------------------------------------------------

project = "FreeRec"
copyright = "2024, MTandHJ"
author = "MTandHJ"

try:
    from importlib.metadata import version as _get_version

    release = _get_version("freerec")
except Exception:
    release = "0.9.7"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "src"]

language = "zh_CN"
locale_dirs = ["locales/"]
gettext_compact = False

# -- Napoleon (NumPy-style docstrings) --------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Autodoc -----------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Mock imports are handled manually above via _MockModule to avoid
# metaclass conflicts with IterDataPipe inheritance.

# -- Autosummary -------------------------------------------------------------

autosummary_generate = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "logo": {
        "text": "FreeRec",
    },
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
    "navbar_end": ["theme-switcher", "navbar-icon-links", "components/lang-switcher"],
}
