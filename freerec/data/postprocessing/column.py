

from typing import Iterator, Iterable, List, Union, TypeVar, Any, Callable, Optional

import numpy as np
import torch, warnings
import torchdata.datapipes as dp
from functools import partial

from .base import Postprocessor
from ..fields import BufferField, FieldList, Field


__all__ = [
    "Columner", "ToTensor", "Fielder",
    'LeftPaddingCol', 'RightPaddingCol'
]


def _to_array(x: Union[np.ndarray, Iterable]) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else np.array(x)


T = TypeVar("T")


@dp.functional_datapipe("column_")
class Columner(Postprocessor):
    r"""
    A postprocessor that converts a batch of samples into columns.
    Columner takes a datapipe that yields a batch of samples, and converts them into columns. 
    This can be useful for the following transformation (by column).

    Parameters:
    -----------
    source_dp: dp.IterDataPipe 
        A datapipe that yields a batch samples.
    """

    def __iter__(self):
        for batch in self.source:
            yield list(zip(*batch))


@dp.functional_datapipe("tensor_")
class ToTensor(Postprocessor):
    r"""
    A datapipe that converts lists into torch Tensors.
    This class converts a List into a torch.Tensor.
    """

    def __init__(
        self, source_dp: dp.iter.IterDataPipe, *, 
        idtype = torch.long, fdtype = torch.float32
    ) -> None:
        super().__init__(source_dp)
        self.idtype = idtype
        self.fdtype = fdtype

    def at_least_2d(self, vals: torch.Tensor):
        """Reshape tensor to 2D if needed."""
        return vals.unsqueeze(1) if vals.ndim == 1 else vals

    def to_tensor(self, col: List) -> Union[List, torch.Tensor]:
        """Convert the List to a torch.Tensor."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arr = np.stack(col)
            if arr.dtype.kind == 'i':
                ter = torch.as_tensor(arr, dtype=self.idtype)
            elif arr.dtype.kind == 'f': 
                ter = torch.as_tensor(arr, dtype=self.fdtype)
            else:
                ter = torch.as_tensor(arr)
            return self.at_least_2d(ter)
        except ValueError: # skip ragged List
            return col
        except TypeError: # skip objects
            return col

    def __iter__(self) -> Iterator:
        for cols in self.source:
            yield self.listmap(self.to_tensor, cols)


@dp.functional_datapipe("field_")
class Fielder(Postprocessor):
    r"""
    Convert column data to field-style data.
    Then filtering by tags can be possible.
    """

    def __init__(
        self, source_dp: dp.iter.IterDataPipe, 
        *fields: Field
    ) -> None:
        super().__init__(source_dp, fields=fields)

    @staticmethod
    def _buffer(field: Field, col: Any):
        return field.buffer(col)

    def __iter__(self) -> Iterator[FieldList[BufferField]]:
        for cols in self.source:
            yield FieldList(map(
                self._buffer,
                self.fields,
                cols
            ))


class ColMapper(dp.iter.IterDataPipe):
    r"""
    Apply a function to specific indices of some columns in an IterableDataset.

    Parameters:
    -----------
    source_dp : tud.IterableDataset
        The source IterableDataset.
    fn : Callable
        The function to apply to the specified indices of each row.
    indices : Iterable[int]
        The indices of the elements in each row to which `fn` should be applied.

    Raises:
    -------
    TypeError
        If `indices` is not an iterable.

    Attributes:
    -----------
    source : tud.IterableDataset
        The source IterableDataset.
    fn : Callable
        The function to apply to the specified indices of each row.
    indices : list of int
        The indices of the elements in each row to which `fn` should be applied.
    """

    def __init__(
        self, source_dp: dp.iter.IterableWrapper,
        fn: Callable, indices: Iterable[int]
    ):
        super().__init__()

        assert isinstance(indices, Iterable), \
            f"{self.__class__.__name__} requires iterable indices but {type(indices)} recevied ..."

        self.source = source_dp
        self.fn = fn
        self.indices = sorted(set(indices))

    def _apply_fn(self, cols: Iterable):
        r"""
        Apply the specified function to the elements of the row at the specified indices.

        Parameters:
        -----------
        cols : Iterable[Union[List, Tuple, Tensor]]
            The col of data to transform.

        Returns:
        --------
        list
            The transformed cols.
        """
        if isinstance(cols, torch.Tensor):
            cols = cols.tolist()
        elif not isinstance(cols, List):
            cols = list(cols)
        for i in self.indices:
            cols[i] = self.fn(cols[i])
        return cols

    def __iter__(self):
        for cols in self.source:
            yield self._apply_fn(cols)

        
@dp.functional_datapipe("lpad_col_")
class LeftPaddingCol(ColMapper):
    r"""
    A functional data pipeline component that left pads sequences to a maximum length.

    Parameters:
    -----------
    source_dp : dp.iter.IterDataPipe
        The source data pipeline to operate on.
    indices : Iterable[int]
        The indices of the elements in each row to which `fn` should be applied.
    maxlen : int, optional
        The maximum length to pad the sequences to.
        `None`: `maxlen' will be the maximum length of the current batch
    padding_value : int, optional (default=0)
        The value to use for padding.
    """

    def __init__(
        self, source_dp: dp.iter.IterDataPipe, 
        indices: Union[None, int, Iterable[int]],
        maxlen: Optional[int] = None, padding_value: int = 0,
    ) -> None:

        self.maxlen = maxlen
        self.padding_value = padding_value

        super().__init__(
            source_dp=source_dp, 
            fn=self._lpad,
            indices=indices
        )

    def _zero_like(self, x: Union[Iterable, int, float]):
        if isinstance(x, Iterable):
            return [self._zero_like(item) for item in x]
        else:
            return self.padding_value

    def _lpad_row(self, x: Iterable, maxlen: int):
        x = _to_array(x)
        if len(x) >= maxlen:
            return x
        elif len(x) > 0:
            p = np.empty_like(x[0])
            p.fill(self.padding_value)
            p = np.stack(
                [p] * (maxlen - len(x)),
                axis=0
            )
            return np.concatenate((p, x), axis=0)
        else:
            raise ValueError("Cannot pad an empty element")

    def _lpad(self, rows: Iterable):
        rows = rows if isinstance(rows, List) else list(rows)
        maxlen = maxlen if self.maxlen is not None else max([len(row) for row in rows])
        return list(map(
            partial(self._lpad_row, maxlen=maxlen),
            rows
        ))


@dp.functional_datapipe("rpad_col_")
class RightPaddingCol(ColMapper):
    r"""
    A functional data pipeline component that right pads sequences to a maximum length.

    Parameters:
    -----------
    source_dp : dp.iter.IterDataPipe
        The source data pipeline to operate on.
    indices : Iterable[int]
        The indices of the elements in each row to which `fn` should be applied.
    maxlen : int
        The maximum length to pad the sequences to.
        `None`: `maxlen' will be the maximum length of the current batch
    padding_value : int, optional (default=0)
        The value to use for padding.
    """

    def __init__(
        self, source_dp: dp.iter.IterDataPipe, 
        indices: Union[None, int, Iterable[int]],
        maxlen: Optional[int] = None, padding_value: int = 0,
    ) -> None:

        self.maxlen = maxlen
        self.padding_value = padding_value

        super().__init__(
            source_dp=source_dp, 
            fn=self._rpad,
            indices=indices
        )

    def _zero_like(self, x: Union[Iterable, int, float]):
        if isinstance(x, Iterable):
            return [self._zero_like(item) for item in x]
        else:
            return self.padding_value

    def _rpad_row(self, x: Iterable, maxlen: int):
        x = _to_array(x)
        if len(x) >= maxlen:
            return x
        elif len(x) > 0:
            p = np.empty_like(x[0])
            p.fill(self.padding_value)
            p = np.stack(
                [p] * (maxlen - len(x)),
                axis=0
            )
            return np.concatenate((x, p), axis=0)
        else:
            raise ValueError("Cannot pad an empty element")

    def _rpad(self, rows: Iterable):
        rows = rows if isinstance(rows, List) else list(rows)
        maxlen = maxlen if self.maxlen is not None else max([len(row) for row in rows])
        return list(map(
            partial(self._rpad_row, maxlen=maxlen),
            rows
        ))