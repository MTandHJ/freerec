

from typing import Union

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

    def valid(self):
        super().valid()
        self.source.valid()

    def test(self):
        super().test()
        self.source.test()


class ModeError(Exception): ...