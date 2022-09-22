

from typing import Callable, Iterator, Union

import torchdata.datapipes as dp

from .base import Postprocessor
from ..datasets import RecDataSet


__all__ = ['Batcher', 'Mapper', 'ShardingFilter']


@dp.functional_datapipe("batch_")
class Batcher(Postprocessor):

    def __init__(
        self, datapipe: Union[RecDataSet, Postprocessor],
        batch_size: int, drop_last: bool = False
    ) -> None:
        super().__init__(datapipe)
        self.datapipe = datapipe.batch(batch_size, drop_last)

    def __iter__(self) -> Iterator:
        yield from self.datapipe


@dp.functional_datapipe("map_")
class Mapper(Postprocessor):

    def __init__(
        self, datapipe: Union[RecDataSet, Postprocessor], fn: Callable
    ) -> None:
        super().__init__(datapipe)
        self.datapipe = datapipe.map(fn)
    
    def __iter__(self) -> Iterator:
        yield from self.datapipe


@dp.functional_datapipe("sharding_filter_")
class ShardingFilter(Postprocessor):

    def __init__(self, datapipe: Union[RecDataSet, Postprocessor]) -> None:
        super().__init__(datapipe)
        self.datapipe = datapipe.sharding_filter()

    def __iter__(self) -> Iterator:
        yield from self.datapipe