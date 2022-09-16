

from .parser import CoreParser
from .launcher import Adapter


cfg = CoreParser()
cfg.compile()

tuner = Adapter()
tuner.compile(cfg)
tuner.fit()

