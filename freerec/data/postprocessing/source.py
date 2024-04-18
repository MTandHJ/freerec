

from typing import Any, Iterable, Dict

import torchdata.datapipes as dp

from .base import Source
from ..datasets.base import RecDataSet
from ..fields import Field


__all__ = [
    'RandomChoicedSource', 'RandomShuffledSource', 'OrderedSource',
]


@dp.functional_datapipe("choiced_source_")
class RandomChoicedSource(Source):
    r"""
    DataPipe that generates random items from given source.
    Note that this sampling is with replacement.

    Parameters:
    -----------
    source: Iterable 
        The source data to start.
    datasize: int 
        Datasize.
    """

    def __init__(
        self, dataset: RecDataSet, source: Iterable[Dict[Field, Any]]
    ) -> None:
        super().__init__(dataset)

        self.datasize = dataset.datasize
        self.source = tuple(source)

    def __len__(self):
        return self.datasize

    def __iter__(self):
        for _ in range(self.datasize):
            yield self._rng.choice(self.source).copy()


@dp.functional_datapipe("shuffled_source_")
class RandomShuffledSource(Source):
    r"""
    DataPipe that generates shuffled source.
    In this vein, every sample will be selected once per epoch.

    Parameters:
    -----------
    source: Iterable 
        The source data to start.
    """

    def __init__(self, dataset: RecDataSet, source: Iterable[Dict[Field, Any]]) -> None:
        super().__init__(dataset)

        self.source = list(source)

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        self._rng.shuffle(self.source)
        for row in self.source:
            yield row.copy()


@dp.functional_datapipe("ordered_source_")
class OrderedSource(Source):
    r"""
    DataPipe that generates ordered items from given source.

    Parameters:
    -----------
    source: Sequence 
        The source data to start.
    """

    def __init__(self, dataset: RecDataSet, source: Iterable[Dict[Field, Any]]) -> None:
        super().__init__(dataset)

        self.source = tuple(source)

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        for row in self.source:
            yield row.copy()