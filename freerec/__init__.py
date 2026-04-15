r"""FreeRec: A free and open-source recommendation library.

This package provides modular components for building, training,
and evaluating recommendation systems.
"""

__version__ = '0.9.7'

from . import data, models, criterions, ddp, graph, launcher, metrics, parser, utils
from .utils import infoLogger
from freerec.dict2obj import Config


def declare(*, version: str):
    r"""Check that the installed FreeRec version matches the required version.

    Prints a warning if there is a mismatch.

    Parameters
    ----------
    version : str
        The expected FreeRec version string.
    """
    if version != __version__:
        print(f"\033[1;31m[Warning] FreeRec version of {version} is required but current version is {__version__} \033[0m")
