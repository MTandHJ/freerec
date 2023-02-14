

from typing import Union, Dict, Iterator

import numpy as np
from ..fields import FieldList, BufferField

from ..datasets import BaseSet, RecDataSet


__all__ = ['Postprocessor', 'ModeError']


class Postprocessor(BaseSet):

    def __init__(self, datapipe: Union[RecDataSet, 'Postprocessor']) -> None:
        super().__init__()
        self.source = datapipe

    @property
    def fields(self):
        return self.source.fields

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

    @property
    def VALID_IS_TEST(self):
        return self.source.VALID_IS_TEST

    @property
    def DEFAULT_CHUNK_SIZE(self):
        return self.source.DEFAULT_CHUNK_SIZE

class ModeError(Exception): ...

