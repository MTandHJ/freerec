from typing import Iterable, Iterator, Union

import torch
import torchdata.datapipes as dp

from freerec.data.fields import Field
from freerec.data.postprocessing.base import PostProcessor
from freerec.data.tags import SIZE

__all__ = ["Batcher_", "ToTensor"]


@dp.functional_datapipe("batch_")
class Batcher_(PostProcessor):
    r"""Convert a stream of row dicts into batched column dicts.

    Each yielded dict maps :class:`~Field` to a list of values collected
    from the batch, plus a special ``SIZE`` field recording the batch size.

    Parameters
    ----------
    source : :class:`~PostProcessor`
        A datapipe that yields individual row dicts.
    batch_size : int
        Number of rows per batch.
    drop_last : bool, optional
        Whether to drop the last incomplete batch. Default is ``False``.
    """

    def __init__(
        self, source: PostProcessor, batch_size: int, drop_last: bool = False
    ) -> None:
        r"""Initialize the Batcher_."""
        super().__init__(source)

        self.Size = Field(SIZE.name, SIZE)
        self.source = source.batch(batch_size, drop_last)

    def __iter__(self):
        r"""Yield batched column dicts."""
        for batch in self.source:
            batch_data = {field: [row[field] for row in batch] for field in batch[0]}
            batch_data[self.Size] = len(batch)
            yield batch_data


@dp.functional_datapipe("tensor_")
class ToTensor(PostProcessor):
    r"""Convert list columns into :class:`torch.Tensor` objects.

    Notes
    -----
    The returned tensor is at least 2-dimensional. Ragged data that cannot
    be stacked into a tensor is passed through unchanged.
    """

    def at_least_2d(self, vals: torch.Tensor):
        r"""Reshape a 1-D tensor to 2-D by adding a trailing dimension."""
        return vals.unsqueeze(1) if vals.ndim == 1 else vals

    def to_tensor(self, col: Iterable) -> Union[Iterable, torch.Tensor]:
        r"""Convert column data to a :class:`torch.Tensor` if possible.

        Parameters
        ----------
        col : iterable
            Column data to convert.

        Returns
        -------
        :class:`torch.Tensor` or iterable
            A tensor when conversion succeeds, the original iterable otherwise.
        """
        try:
            if isinstance(col, Iterable):
                col = self.at_least_2d(torch.tensor(col))
        except ValueError:  # skip ragged data
            pass
        finally:
            return col

    def __iter__(self) -> Iterator:
        r"""Yield dicts with list columns converted to tensors."""
        for cols in self.source:
            yield {field: self.to_tensor(col) for field, col in cols.items()}
