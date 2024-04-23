

from typing import Any, Iterable, Dict

import random
import torchdata.datapipes as dp

from .base import Source
from ..datasets.base import RecDataSet
from ..fields import Field


__all__ = [
    'RandomChoicedSource', 'RandomShuffledSource', 'OrderedSource',
]


class Launcher(dp.iter.IterDataPipe):

    def __init__(self, datasize: int, shuffle: bool = True):
        super().__init__()

        self.source = list(range(datasize))
        self.shuffle = shuffle

        self._rng = random.Random()
        self.set_seed(0)

    def set_seed(self, seed: int):
        self._rng.seed(seed)

    def __iter__(self):
        if self.shuffle:
            self._rng.shuffle(self.source)
        yield from iter(self.source)


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
        super().__init__(dataset, source)

        self.datasize = dataset.datasize
        self.launcher = Launcher(self.datasize, shuffle=False).sharding_filter()

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
        super().__init__(dataset, source)

        self.datasize = len(self.source)
        self.launcher = Launcher(self.datasize, shuffle=True).sharding_filter()

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
        super().__init__(dataset, source)

        self.datasize = len(self.source)
        self.launcher = Launcher(self.datasize, shuffle=False).sharding_filter()

    def __iter__(self):
        for i in self.launcher:
            yield self.source[i].copy()