__version__ = '0.1.1'

from . import data, models, layers, criterions, launcher, metrics, utils
from freerec.dict2obj import Config


def check_version(version: str):
    """
    This function checks whether the provided version matches the current version of FreeRec package. 
    If they do not match, a warning message is printed.
    """
    if version != __version__:
        print(f"\033[1;31m [Warning] FreeRec version of {version} is required but current version is {__version__} \033[0m")