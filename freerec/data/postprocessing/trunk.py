
from typing import Callable, Iterator, Optional, Dict

import torch
import torchdata.datapipes as dp
import numpy as np

from .base import Postprocessor
from ..datasets import RecDataSet


__all__ = [
    "Sharder", "Shuffle", "SubFielder", "Chunker", "ToTensor"
]


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


@dp.functional_datapipe("shuffle_")
class Shuffle(Postprocessor):

    def shuffle(self, chunk: Dict):
        indices = None
        for key in chunk:
            if indices is None:
                indices = np.arange(len(chunk[key]))
                np.random.shuffle(indices)
            chunk[key] = chunk[key][indices]
        return chunk

    def __iter__(self):
        for chunk in self.source:
            yield self.shuffle(chunk)


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

    def __iter__(self) -> Iterator:
        for chunk in self.source:
            yield {field.name: chunk[field.name] for field in self.fields}


@dp.functional_datapipe("tensor_")
class ToTensor(Postprocessor):
    """Convert Dict[str, List] into Dict[str, torch.Tensor].
    
    Returns:
        Dict[field, torch.Tensor],
    """
    def at_least_2d(self, vals: torch.Tensor):
        return vals.unsqueeze(1) if vals.ndim == 1 else vals

    def __iter__(self) -> Iterator:
        for batch in self.source:
            yield {field.name: self.at_least_2d(torch.tensor(batch[field.name], dtype=field.dtype)) for field in self.fields}


@dp.functional_datapipe("chunk_")
class Chunker(Postprocessor):
    """A special batcher for Dict[str, Tensor] only.
    
    NOTE: If the given batch size is not evenly divisible by _DEFAULT_CHUNK_SIZE,
    minor chunk will yielded frequently. Fortunately, _DEFAULT_CHUNK_SIZE=51200 satisfies
    almost all popular batch sizes. For example, 128, 1024 ... 
    """

    def __init__(
        self, datapipe: Postprocessor, batch_size: int
    ) -> None:
        super().__init__(datapipe)

        self.batch_size = batch_size

    def __iter__(self) -> Iterator:
        for chunk in self.source:
            chunk = {key: torch.split(vals, self.batch_size, dim=0) for key, vals in chunk.items()}
            try:
                k = 0
                while True:
                    yield {key: chunk[key][k] for key in chunk}
                    k += 1
            except IndexError:
                pass

