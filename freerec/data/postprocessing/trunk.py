

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

    def __init__(self, datapipe):
        super().__init__(datapipe)
        self.num_of_instances = 1
        self.instance_id = 0

    def is_shardable(self):
        return True

    def apply_sharding(self, num_of_instances, instance_id):
        self.num_of_instances = num_of_instances
        self.instance_id = instance_id

    def forward(self):
        for i, item in enumerate(self.source):
            if i % self.num_of_instances == self.instance_id:
                yield item

    def __len__(self):
        if isinstance(self.source, Sized):
            return len(self.source) // self.num_of_instances +\
                (1 if (self.instance_id < len(self.source) % self.num_of_instances) else 0)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))


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
    """Convert Dict[str, List] into Dict[str, torch.Tensor]."""

    def at_least_2d(self, vals: torch.Tensor):
        return vals.unsqueeze(1) if vals.ndim == 1 else vals

    def to_tensor(self, field: BufferField):
        return field.buffer(self.at_least_2d(
            torch.tensor(field.data, dtype=field.dtype)
        ))

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