

from typing import Iterable, Sequence

import random
import torchdata.datapipes as dp
from functools import partial

from .base import BaseProcessor
from ..fields import SparseField


__all__ = [
    'RandomSource', 'OrderedSource',
    'RandomIDs', 'OrderedIDs', 'DummySource'
]


class RandomSource(BaseProcessor):
    """DataPipe that generates random items from given source.

    Args:
        source (Iterable): The source data to start.
        datasize (int): Datasize.
    """

    def __init__(
        self, source: Iterable, datasize: int
    ) -> None:
        super().__init__(None)

        self._rng = partial(
            random.choice, seq=tuple(source)
        )
        self.datasize = datasize

    def __iter__(self):
        for _ in range(self.datasize):
            yield self._rng()


class OrderedSource(BaseProcessor):
    """DataPipe that generates ordered items from given source.

    Args:
        source (Sequence): The source data to start.
        datasize (int): Datasize.
    """

    def __init__(self, source: Sequence) -> None:
        super().__init__(None)

        assert isinstance(source, Sequence), f"Sequence type is required but received {type(source)} type."
        self.source = tuple(source)

    def __iter__(self):
        yield from iter(self.source)


class RandomIDs(RandomSource):
    """DataPipe that generates random IDs according to SparseField.

    Args:
        field SparseField: ID values to select from.
        datasize (int): Number of IDs to generate.
    """

    def __init__(
        self, field: SparseField,
        datasize: int,
    ) -> None:
        super().__init__(field.ids, datasize)


class OrderedIDs(OrderedSource):
    """DataPipe that generates ordered IDs.

    Args:
        low (Optional[int]): Lowest possible ID value. Default: None.
        high (Optional[int]): Highest possible ID value. Default: None.
        ids (Union[None, SparseField, Iterable]): ID values to select from. Default: None.
    """

    def __init__(self, field: SparseField) -> None:
        super().__init__(field.ids)


class DummySource(OrderedSource):
    """DataPipe that generates dummy data.

    Args:
        datasize (int): Number of data to generate.
    """
    def __init__(self, datasize: int) -> None:
        super().__init__(range(datasize))


@dp.functional_datapipe("dummy_")
class _DummySource(DummySource):
    """Functional DataPipe wrapper for DummySource.

    Args:
        source_dp (dp.iter.IterDataPipe): The source DataPipe (No use here).
        datasize (int): Number of data to generate.
    """
    def __init__(self, source_dp: dp.iter.IterDataPipe, datasize: int) -> None:
        super().__init__(datasize)