

# Grid search for hyper-parameters can be conducted by:
#
#    python -m freerec NAME_OF_EXPERIMENT CONFIG.yaml
#

from .parser import CoreParser
from .launcher import Adapter


cfg = CoreParser()
cfg.compile()

tuner = Adapter()
tuner.compile(cfg)
tuner.fit()

