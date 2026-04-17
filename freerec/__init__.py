r"""FreeRec: A free and open-source recommendation library.

This package provides modular components for building, training,
and evaluating recommendation systems.
"""

from __future__ import annotations

import importlib
from importlib.metadata import version
from typing import TYPE_CHECKING

__version__ = version("freerec")

if TYPE_CHECKING:
    from . import criterions, data, ddp, graph, launcher, metrics, models, parser, utils
    from .dict2obj import Config
    from .utils import infoLogger

_SUBMODULES = {
    "criterions",
    "data",
    "ddp",
    "graph",
    "launcher",
    "metrics",
    "models",
    "parser",
    "utils",
}

_ATTRS = {
    "Config": ("dict2obj", "Config"),
    "infoLogger": ("utils", "infoLogger"),
}

__all__ = [*_SUBMODULES, *_ATTRS]


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _ATTRS:
        module_name, attr_name = _ATTRS[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'freerec' has no attribute {name!r}")


def declare(*, version: str):
    r"""Check that the installed FreeRec version matches the required version.

    Prints a warning if there is a mismatch.

    Parameters
    ----------
    version : str
        The expected FreeRec version string.
    """
    if version != __version__:
        print(
            f"\033[1;31m[Warning] FreeRec version of {version} is required but current version is {__version__} \033[0m"
        )
