
from typing import Callable, Iterable, Iterator, Optional, Union

import torch
import torchdata.datapipes as dp
import pandas as pd

from .base import Postprocessor
from ..datasets import RecDataSet


__all__ = [
    "DataFrame", "Sharder", "PinMemory", "SubFielder", "Chunker", 
    "Frame2Dict", "Frame2List", "ToTensor"
]


@dp.functional_datapipe("dataframe_")
class DataFrame(Postprocessor):
    """Make pd.DataFrame from source data."""

    def __init__(
        self, datapipe: Union[RecDataSet, Postprocessor],
        columns: Optional[Iterable] = None
    ) -> None:
        super().__init__(datapipe)
        self.columns = columns

    def __iter__(self) -> Iterator:
        for batch in self.source:
            yield pd.DataFrame(batch, columns=self.columns)


@dp.functional_datapipe("shard_")
class Sharder(Postprocessor):
    """For num_workers != 0."""

    def __iter__(self) -> Iterator:
        worker_infos = torch.utils.data.get_worker_info()
        if worker_infos:
            id_, nums = worker_infos.id, worker_infos.num_workers
            for idx, item in enumerate(self.source):
                if idx % nums == id_:
                    yield item
        else:
            yield from self.source


@dp.functional_datapipe("pin_")
class PinMemory(Postprocessor):
    """Pin buffer_size data into memory for efficiency."""

    def __init__(
        self, datapipe: RecDataSet,
        buffer_size: Optional[int] = None, shuffle: bool = True
    ) -> None:
        super().__init__(datapipe)

        self.buffer_size = buffer_size if buffer_size else float('inf')
        self.shuffle = shuffle

    def __iter__(self) -> Iterator:
        _buffer = None
        for df in self.source:
            _buffer = pd.concat((_buffer, df))
            if len(_buffer) >= self.buffer_size:
                if self.mode == 'train' and self.shuffle:
                    yield _buffer.sample(frac=1)
                else:
                    yield _buffer
                _buffer = None

        if _buffer is not None and not _buffer.empty:
            if self.mode == 'train' and self.shuffle:
                yield _buffer.sample(frac=1)
            else:
                yield _buffer


@dp.functional_datapipe("subfield_")
class SubFielder(Postprocessor):
    """Select subfields."""

    def __init__(
        self, datapipe: Postprocessor,
        filter_: Optional[Callable] = None
    ) -> None:
        super().__init__(datapipe)
        if filter_:
            self.fields = [field for field in self.fields if filter_(field)]
        self.columns = [field.name for field in self.fields]

    def __iter__(self) -> Iterator:
        for df in self.source:
            yield df[self.columns]


@dp.functional_datapipe("chunk_")
class Chunker(Postprocessor):
    """A special batcher for dataframe only"""

    def __init__(
        self, datapipe: Postprocessor, 
        batch_size: int, drop_last: bool = False
    ) -> None:
        super().__init__(datapipe)

        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator:
        _buffer = None
        for df in self.source:
            _buffer = pd.concat((_buffer, df))
            for _ in range(self.batch_size, len(_buffer) + 1, self.batch_size):
                yield _buffer.head(self.batch_size)
                _buffer = _buffer[self.batch_size:]

        if not self.drop_last and _buffer is not None and not _buffer.empty:
            yield _buffer


@dp.functional_datapipe("dict_")
class Frame2Dict(Postprocessor):
    """Convert dataframe into Dict[str, List]."""
    def __iter__(self) -> Iterator:
        for df in self.source:
            yield {name: df[name].values.tolist() for name in df.columns}


@dp.functional_datapipe("list_")
class Frame2List(Postprocessor):
    """Convert dataframe into List[List]."""
    def __iter__(self) -> Iterator:
        for df in self.source:
            yield [df[name].values.tolist() for name in df.columns]


@dp.functional_datapipe("tensor_")
class ToTensor(Postprocessor):
    """Convert Dict[str, List] into Dict[str, torch.Tensor]."""
    def at_least_2d(self, val: torch.Tensor):
        return val.unsqueeze(1) if val.ndim == 1 else val

    def __iter__(self) -> Iterator:
        for batch in self.source:
            yield {field.name: self.at_least_2d(torch.tensor(batch[field.name], dtype=field.dtype)) for field in self.fields}
