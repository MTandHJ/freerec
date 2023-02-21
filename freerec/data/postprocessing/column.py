

from typing import Iterator, List, Union, Iterable, TypeVar

import torch
import torchdata.datapipes as dp

from .base import Postprocessor
from ..fields import BufferField, FieldList


__all__ = ["Columner", "ToTensor", "Fielder"]


T = TypeVar("T")


@dp.functional_datapipe("column_")
class Columner(Postprocessor):
    """A postprocessor that converts a batch of samples into columns.

    Columner takes a datapipe that yields a batch of samples, and converts them into columns. 
    This can be useful for the following transformation (by column).

    Args:
        source_dp (dp.IterDataPipe): A datapipe that yields a batch samples.

    """

    def __iter__(self):
        for batch in self.source:
            yield tuple(zip(*batch))


@dp.functional_datapipe("tensor_")
class ToTensor(Postprocessor):
    """A datapipe that converts lists into torch Tensors.

    This class converts a List into a torch.Tensor.

    """

    def at_least_2d(self, vals: torch.Tensor):
        """Reshape tensor to 2D if needed."""
        return vals.unsqueeze(1) if vals.ndim == 1 else vals

    def to_tensor(self, col: List) -> Union[List, torch.Tensor]:
        """Convert the List to a torch.Tensor."""
        try:
            return self.at_least_2d(
                torch.tensor(col)
            )
        except ValueError: # avoid ragged List
            return col

    def __iter__(self) -> Iterator:
        for chunk in self.source:
            yield self.listmap(self.to_tensor, chunk)


@dp.functional_datapipe("field_")
class Fielder(Postprocessor):
    """Convert column data to field-style data.
    Then filtering by tags can be possible.
    """

    def __iter__(self) -> Iterator[FieldList[BufferField]]:
        for cols in self.source:
            yield FieldList(map(
                lambda field, col: field.buffer(col),
                self.fields,
                cols
            ))

