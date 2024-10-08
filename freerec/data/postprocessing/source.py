

from typing import Any, Iterable, Dict

import random
import torchdata.datapipes as dp

from .base import Source
from ..datasets.base import RecDataSet
from ..fields import Field


__all__ = [
    'RandomChoicedSource', 'RandomShuffledSource', 'OrderedSource', 'PipedSource'
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
        super().__init__(dataset, source, dataset.datasize, shuffle=False)

        self._rng = random.Random()
        self.set_seed(0)

    def set_seed(self, seed: int):
        self._rng.seed(seed)

    def __iter__(self):
        for _ in self.launcher:
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
        super().__init__(dataset, source, shuffle=True)

    def __iter__(self):
        for i in self.launcher:
            yield self.source[i].copy()


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
        super().__init__(dataset, source, shuffle=False)

    def __iter__(self):
        for i in self.launcher:
            yield self.source[i].copy()


@dp.functional_datapipe("piped_source_")
class PipedSource(Source):
    r"""
    DataPipe that yields from the given source.

    Parameters:
    -----------
    source: IterDataPipe
    """

    def __init__(self, dataset: RecDataSet, source: dp.iter.IterDataPipe) -> None:
        super().__init__(dataset, source)
        assert isinstance(source, dp.iter.IterDataPipe), f"PipedSource needs `IterDataPipe` but {type(source)} received ..."

    def __iter__(self):
        for row in self.launcher:
            yield row