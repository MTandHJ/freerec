__version__ = '0.0.20'

import freerec.data as data
import freerec.models as models
import freerec.layers as layers
import freerec.criterions as criterions
import freerec.launcher as launcher
import freerec.metrics as metrics
import freerec.parser as parser
import freerec.utils as utils
from freerec.dict2obj import Config


def check_version(version: str):
    if version != __version__:
        print(f"\033[1;31m [Warning] FreeRec version of {version} is required but current version is {__version__} \033[0m")