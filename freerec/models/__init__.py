from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import nn
    from .base import GenRecArch, PredRecArch, RecSysArch, SeqRecArch

_SUBMODULES = {"nn"}

_ATTRS = {
    "GenRecArch": ("base", "GenRecArch"),
    "PredRecArch": ("base", "PredRecArch"),
    "RecSysArch": ("base", "RecSysArch"),
    "SeqRecArch": ("base", "SeqRecArch"),
}

__all__ = [*_SUBMODULES, *_ATTRS]


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _ATTRS:
        module_name, attr_name = _ATTRS[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'freerec.models' has no attribute {name!r}")
