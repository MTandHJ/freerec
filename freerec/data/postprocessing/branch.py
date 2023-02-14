

from typing import Iterator, Optional

import torchdata.datapipes as dp

from .base import Postprocessor
from ..fields import FieldList, BufferField


__all__ = ['Wrapper']


@dp.functional_datapipe("wrap_")
class Wrapper(Postprocessor):
    """Wrap trainpipe, validpipe and testpipe together."""

    def __init__(
        self, datapipe: Postprocessor,
        validpipe: Optional[Postprocessor] = None,
        testpipe: Optional[Postprocessor] = None,
    ) -> None:
        """
        Parameters:
        ---

        datapipe: trainpipe
        validpipe: validpipe or None
            - `None`: Set validpipe == trainpipe

        testpipe: testpipe
            - `None`: Set testpipe == validpipe

        Examples:
        ---

        >>> from freerec.data.datasets import MovieLens1M
        >>> basepipe = MovieLens1M("../../data/MovieLens1M")
        >>> datapipe = basepipe.shuffle_().shard_()
        >>> trainpipe = datapipe.negatives_for_train_(num_negatives=4)
        >>> validpipe = datapipe.negatives_for_eval_(num_negatives=99) # 1:99
        >>> dataset = trainpipe.wrap_(validpipe).tensor_().chunk_(cfg.batch_size).group_()
        """
        super().__init__(datapipe)
        self.validpipe = datapipe if validpipe is None else validpipe
        self.testpipe = self.validpipe if testpipe is None else testpipe
        self.train()

    def train(self):
        super().train()
        self.source.train()
        return self

    def valid(self):
        super().valid()
        self.validpipe.valid()
        return self

    def test(self):
        super().test()
        self.testpipe.test()
        return self

    def __len__(self):
        if self.mode == 'train':
            return len(self.source)
        elif self.mode == 'valid':
            return len(self.validpipe)
        else:
            return len(self.testpipe)

    def forward(self) -> Iterator[FieldList[BufferField]]:
        if self.mode == 'train':
            yield from self.source
        elif self.mode == 'valid':
            yield from self.validpipe
        else:
            yield from self.testpipe
