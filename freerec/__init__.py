__version__ = '0.0.24'

from . import data, models, layers, criterions, launcher, metrics, utils
from freerec.dict2obj import Config


def check_version(version: str):
    if version != __version__:
        print(f"\033[1;31m [Warning] FreeRec version of {version} is required but current version is {__version__} \033[0m")