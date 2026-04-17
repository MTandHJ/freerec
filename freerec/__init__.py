r"""FreeRec: A free and open-source recommendation library.

This package provides modular components for building, training,
and evaluating recommendation systems.
"""

from importlib.metadata import version

__version__ = version("freerec")

try:
    from freerec.dict2obj import Config

    from . import criterions, data, ddp, graph, launcher, metrics, models, parser, utils
    from .utils import infoLogger
except ModuleNotFoundError:
    # torch/torchdata not yet installed; CLI (e.g., freerec setup) still works.
    pass


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
