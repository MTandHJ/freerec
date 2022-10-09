

from typing import Union, Dict

import numpy as np

from ..datasets import BaseSet, RecDataSet


__all__ = ['Postprocessor', 'ModeError']


class Postprocessor(BaseSet):

    def __init__(self, datapipe: Union[RecDataSet, 'Postprocessor']) -> None:
        super().__init__()
        self.source = datapipe
        self.fields = self.source.fields

    def train(self):
        super().train()
        self.source.train()
        return self

    def valid(self):
        super().valid()
        self.source.valid()
        return self

    def test(self):
        super().test()
        self.source.test()
        return self

    def at_least_2d(self, array: np.array):
        return array[:, None] if array.ndim == 1 else array

    @property
    def datasize(self):
        return self.source.datasize

    def __len__(self):
        return len(self.source)

class ModeError(Exception): ...

