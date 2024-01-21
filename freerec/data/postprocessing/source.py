

from typing import Iterable, Sequence

import random
import torchdata.datapipes as dp

from .base import BaseProcessor
from ..fields import SparseField


__all__ = [
    'RandomChoicedSource', 'RandomShuffledSource', 'OrderedSource',
    'RandomIDs', 'OrderedIDs', 'DummySource'
]


class RandomChoicedSource(BaseProcessor):
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
        self, source: Iterable, datasize: int
    ) -> None:
        super().__init__(None)

        self.source = tuple(source)
        self.datasize = datasize
        self._rng = random.Random()
        self.set_seed(1)

    def set_seed(self, seed: int):
        self._rng.seed(seed)

    def __iter__(self):
        for _ in range(self.datasize):
            yield self._rng.choice(self.source)


class RandomShuffledSource(BaseProcessor):
    r"""
    DataPipe that generates shuffled source.
    In this vein, every sample will be selected once per epoch.

    Parameters:
    -----------
    source: Iterable 
        The source data to start.
    """

    def __init__(self, source) -> None:
        super().__init__(None)

        self.source = list(source)
        self._rng = random.Random()
        self.set_seed(1)

    def set_seed(self, seed: int):
        self._rng.seed(seed)

    def __iter__(self):
        self._rng.shuffle(self.source)
        yield from iter(self.source)


class OrderedSource(BaseProcessor):
    r"""
    DataPipe that generates ordered items from given source.

    Parameters:
    -----------
    source: Sequence 
        The source data to start.
    datasize: int 
        Datasize.
    """

    def __init__(self, source: Sequence) -> None:
        super().__init__(None)

        assert isinstance(source, Sequence), f"Sequence type is required but received {type(source)} type."
        self.source = tuple(source)

    def __iter__(self):
        yield from iter(self.source)


class RandomIDs(RandomChoicedSource):
    r"""
    DataPipe that generates random IDs according to SparseField.

    Parameters:
    -----------
    field: SparseField 
        ID values to select from.
    datasize: int 
        Number of IDs to generate.
    """

    def __init__(
        self, field: SparseField,
        datasize: int,
    ) -> None:
        super().__init__(field.enums, datasize)


class OrderedIDs(OrderedSource):
    r"""
    DataPipe that generates ordered IDs.

    Parameters:
    -----------
    field: SparseField 
        ID values to select from.
    """

    def __init__(self, field: SparseField) -> None:
        super().__init__(field.enums)


class DummySource(OrderedSource):
    r"""
    DataPipe that generates dummy data.

    Parameters:
    -----------
    datasize: int 
        Number of data to generate.
    """
    def __init__(self, datasize: int) -> None:
        super().__init__(range(datasize))


@dp.functional_datapipe("dummy_")
class _DummySource(DummySource):
    r"""
    Functional DataPipe wrapper for DummySource.

    Parameters:
    -----------
    source_dp: dp.iter.IterDataPipe 
        The source DataPipe (No use here).
    datasize: int 
        Number of data to generate.
    """
    def __init__(self, source_dp: dp.iter.IterDataPipe, datasize: int) -> None:
        super().__init__(datasize)