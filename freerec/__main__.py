

"""
You can grid search the parameters in the following manner:

    python -m freerec NAME_OF_EXPERIMENT CONFIG.yaml

"""

from .parser import CoreParser
from .launcher import Adapter


cfg = CoreParser()
cfg.compile()

tuner = Adapter()
tuner.compile(cfg)
tuner.fit()

