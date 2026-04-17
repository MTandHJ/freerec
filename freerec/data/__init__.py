from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# postprocessing must be eagerly loaded: its subclasses register themselves
# as torchdata datapipes via IterDataPipe.__init_subclass__ on import.
from freerec.data import postprocessing

if TYPE_CHECKING:
    from . import datasets, fields, preprocessing, tags, utils

_SUBMODULES = {"datasets", "fields", "preprocessing", "tags", "utils"}

__all__ = [*_SUBMODULES, "postprocessing"]


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'freerec.data' has no attribute {name!r}")
