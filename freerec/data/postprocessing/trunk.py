

from typing import Iterator, Sized

import torch, random
import torchdata.datapipes as dp
from ..fields import BufferField

from .base import Postprocessor


__all__ = [
    "Sharder", "Shuffle", "Spliter", "ToTensor"
]


@dp.functional_datapipe("shard_")
class Sharder(Postprocessor):
    """For num_workers != 0."""

    def forward(self):
        worker_infos = torch.utils.data.get_worker_info()
        if worker_infos:
            id_, nums = worker_infos.id, worker_infos.num_workers
            for idx, item in enumerate(self.source):
                if idx % nums == id_:
                    yield item
        else:
            yield from self.source


@dp.functional_datapipe("shuffle_")
class Shuffle(Postprocessor):
    """Shuffling each dataframe."""

    def shuffle(self, field, state):
        random.setstate(state) # for same indices
        random.shuffle(field.data)

    def forward(self):
        for chunk in self.source:
            state = random.getstate()
            self.listmap(
                self.shuffle,
                chunk,
                [state] * len(chunk)
            )
            yield chunk


@dp.functional_datapipe("tensor_")
class ToTensor(Postprocessor):
    """Convert List into torch.Tensor."""

    def at_least_2d(self, vals: torch.Tensor):
        return vals.unsqueeze(1) if vals.ndim == 1 else vals

    def to_tensor(self, field: BufferField):
        try:
            return field.buffer(self.at_least_2d(
                torch.tensor(field.data, dtype=field.dtype)
            ))
        except ValueError: # avoid ragged List
            return field

    def forward(self) -> Iterator:
        for chunk in self.source:
            yield self.listmap(self.to_tensor, chunk)


@dp.functional_datapipe("split_")
class Spliter(Postprocessor):
    """A special batcher for DataFrame only."""

    def __init__(
        self, datapipe: Postprocessor, batch_size: int
    ) -> None:
        """
        Parameters:
        ---

        datapipe: Postprocessor
            It must yield tensors or array !
        batch_size: int
            If the given batch size is not evenly divisible by _DEFAULT_CHUNK_SIZE,
            minor chunk will yielded frequently. Fortunately, _DEFAULT_CHUNK_SIZE=51200 satisfies almost all popular batch sizes. For example, 128, 1024 ... 
        """
        super().__init__(datapipe)

        self.batch_size = batch_size

    def split(self, field: BufferField, start, end):
        return field.buffer(field[start:end])

    def forward(self) -> Iterator:

        for chunk in self.source:
            for start, end in zip(
                range(0, chunk.datasize, self.batch_size),
                range(self.batch_size, chunk.datasize + self.batch_size, self.batch_size)
            ):
                yield self.listmap(
                    self.split, chunk, [start] * len(chunk), [end] * len(chunk)
                )