

from typing import Iterator, Iterable, List, Union, TypeVar, Dict

import torch
import torchdata.datapipes as dp

from .base import PostProcessor


__all__ = [
    "Batcher_", "ToTensor"
]


@dp.functional_datapipe("batch_")
class Batcher_(PostProcessor):
    r"""
    A postprocessor that converts a batch of rows into:
        Dict[Field, List[Any]]

    Parameters:
    -----------
    source: dp.IterDataPipe 
        A datapipe that yields a batch samples.
    batch_size: int
    drop_last: bool, default False
    """

    def __init__(
        self, source: PostProcessor, 
        batch_size: int, drop_last: bool = False
    ) -> None:
        super().__init__(source)

        self.input_fields = tuple(self.sure_input_fields())
        self.source = source.batch(batch_size, drop_last)

    def __iter__(self):
        for batch in self.source:
            yield {field: [row[field] for row in batch] for field in self.input_fields}


@dp.functional_datapipe("tensor_")
class ToTensor(PostProcessor):
    r"""
    A datapipe that converts lists into torch Tensors.
    This class converts a List into a torch.Tensor.

    Notes:
    ------
    The returned tensor is at least 2d. 
    """

    def at_least_2d(self, vals: torch.Tensor):
        """Reshape tensor to 2D if needed."""
        return vals.unsqueeze(1) if vals.ndim == 1 else vals

    def to_tensor(self, col: List) -> Union[List, torch.Tensor]:
        """Convert the List to a torch.Tensor."""
        try:
            return self.at_least_2d(torch.tensor(col))
        except ValueError: # skip ragged List
            return col

    def __iter__(self) -> Iterator:
        for cols in self.source:
            yield {field: self.to_tensor(col) for field, col in cols.items()}