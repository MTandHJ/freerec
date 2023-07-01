__version__ = '0.4.3'

from . import data, models, criterions, launcher, metrics, parser, utils
from freerec.dict2obj import Config


def declare(*, version: str):
    """
    This function checks whether the provided version matches the current version of FreeRec package. 
    If they do not match, a warning message would be printed.
    """
    if version != __version__:
        print(f"\033[1;31m[Warning] FreeRec version of {version} is required but current version is {__version__} \033[0m")
