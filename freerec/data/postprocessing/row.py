

from typing import Iterable, Union, Any, Callable, List, Tuple, Dict

import numpy as np
import torchdata.datapipes as dp

from .base import Postprocessor
from ..fields import Field


__all__ = [
    "LeftPruningRow", "RightPruningRow",
    "AddingRow",
    "LeftPaddingRow", "RightPaddingRow",
]


def _to_array(x: Union[np.ndarray, Iterable]) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else np.array(x)

#==================================Filter==================================
class RowFilter(Postprocessor):
    r"""
    Apply a function to specific indices of each row in an IterableDataset.

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
        self, source: dp.iter.IterableWrapper,
        fn: Callable, checked_fields: Iterable[Field]
    ):
        super().__init__(source)
        self.fn = fn
        self.checked_fields = set(self.sure_input_fields()) & set(checked_fields)

    def _check(self, row: Dict[Field, Any]) -> bool:
        return all(self.fn(row[field]) for field in self.checked_fields)

    def __iter__(self):
        for row in self.source:
            if self._check(row):
                yield row


#==================================Mapper==================================
class RowMapper(Postprocessor):
    r"""
    Apply a function to specific indices of each row in an IterableDataset.

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
    modified_fields : Itertable[Field]
        The fields to be modified by given `fn`
    """

    def __init__(
        self, source: dp.iter.IterableWrapper,
        fn: Callable, modified_fields: Iterable[Field]
    ):
        super().__init__(source)

        self.fn = fn
        self.modified_fields = set(self.sure_input_fields()) & set(modified_fields)

    def __iter__(self):
        for row in self.source:
            for field in self.modified_fields:
                row[field] = self.fn(row[field])
            yield row


@dp.functional_datapipe("lprune_")
class LeftPruningRow(RowMapper):
    r"""
    A functional datapipe that prunes the left side of a given datapipe to a specified maximum length.

    Parameters:
    -----------
    source_dp: IterDataPipe 
        The input datapipe to prune.
    indices : Iterable[int]
        The indices of the elements in each row to which `fn` should be applied.
    maxlen: int 
        The maximum length to prune the input data to.
    """
    def __init__(
        self, source: dp.iter.IterDataPipe, maxlen: int, 
        *, modified_fields: Iterable[Field]
    ) -> None:

        self.maxlen = maxlen

        super().__init__(
            source, self._lprune,
            modified_fields
        )

    def _lprune(self, x: Iterable) -> Iterable:
        return x[-self.maxlen:]


@dp.functional_datapipe("rprune_")
class RightPruningRow(RowMapper):
    r"""
    A functional datapipe that prunes the right side of a given datapipe to a specified maximum length.

    Parameters:
    -----------
    source_dp: IterDataPipe 
        The input datapipe to prune.
    indices : Iterable[int]
        The indices of the elements in each row to which `fn` should be applied.
    maxlen: int 
        The maximum length to prune the input data to.
    """

    def __init__(
        self, source: dp.iter.IterDataPipe, maxlen: int, 
        *, modified_fields: Iterable[Field]
    ) -> None:

        self.maxlen = maxlen

        super().__init__(
            source, self._rprune,
            modified_fields
        )

    def _rprune(self, x: Iterable) -> Iterable:
        return x[:self.maxlen]


@dp.functional_datapipe("add_")
class AddingRow(RowMapper):
    r"""
    Mapper that adds the input data by a specified offset.

    Parameters:
    -----------
    source_dp: dp.iter.IterDataPipe
        Input datapipeline.
    indices : Iterable[int]
        The indices of the elements in each row to which `fn` should be applied.
    offset: int
        Amount to add the input data by.   
    """

    def __init__(
        self, source: dp.iter.IterDataPipe, offset: int, 
        *, modified_fields: Iterable[Field]
    ) -> None:

        self.offset = offset

        super().__init__(
            source, self._add,
            modified_fields
        )

    def _add(self, x: Union[np.ndarray, Iterable]) -> np.ndarray:
        r"""
        Examples:
        ---------
        >>> x = [1, 2, 3] 
        >>> _add(x) # self.offset = 1
        [2, 3, 4]
        >>> _add(x) # self.offset = -1
        [0, 1, 2]
        """
        return _to_array(x) + self.offset


@dp.functional_datapipe("lpad_")
class LeftPaddingRow(RowMapper):
    r"""
    A functional data pipeline component that left pads sequences to a maximum length.

    Parameters:
    -----------
    source_dp : dp.iter.IterDataPipe
        The source data pipeline to operate on.
    indices : Iterable[int]
        The indices of the elements in each row to which `fn` should be applied.
    maxlen : int
        The maximum length to pad the sequences to.
    padding_value : int, optional (default=0)
        The value to use for padding.

    Raises:
    -------
    ValueError:
        To pad an empty element.
    """

    def __init__(
        self, source: dp.iter.IterDataPipe, maxlen: int, 
        *, modified_fields: Iterable[Field], padding_value: int = 0
    ) -> None:

        self.maxlen = maxlen
        self.padding_value = int(padding_value)

        super().__init__(
            source, self._lpad,
            modified_fields
        )

    def _lpad(self, x: Union[np.ndarray, Iterable]) -> np.ndarray:
        x = _to_array(x)
        if len(x) >= self.maxlen:
            return x
        elif len(x) > 0: 
            p = np.empty_like(x[0])
            p.fill(self.padding_value)
            p = np.stack(
                [p] * (self.maxlen - len(x)),
                axis=0
            )
            return np.concatenate((p, x), axis=0)
        else:
            raise ValueError("Cannot pad an empty element")


@dp.functional_datapipe("rpad_")
class RightPaddingRow(RowMapper):
    r"""
    A functional data pipeline component that right pads sequences to a maximum length.

    Parameters:
    -----------
    source_dp : dp.iter.IterDataPipe
        The source data pipeline to operate on.
    indices : Iterable[int]
        The indices of the elements in each row to which `fn` should be applied.
    maxlen : int
        The maximum length to right pad the sequences to.
    padding_value : int, optional (default=0)
        The value to use for padding.

    Raises:
    -------
    ValueError:
        To pad an empty element.
    """

    def __init__(
        self, source: dp.iter.IterDataPipe, maxlen: int, 
        *, modified_fields: Iterable[Field], padding_value: int = 0,
    ) -> None:

        self.maxlen = maxlen
        self.padding_value = padding_value

        super().__init__(
            source, self._rpad,
            modified_fields
        )

    def _rpad(self, x: Union[np.ndarray, Iterable]) -> np.ndarray:
        x = _to_array(x)
        if len(x) >= self.maxlen:
            return x
        elif len(x) > 0:
            p = np.empty_like(x[0])
            p.fill(self.padding_value)
            p = np.stack(
                [p] * (self.maxlen - len(x)),
                axis=0
            )
            return np.concatenate((x, p), axis=0)
        else:
            raise ValueError("Cannot pad an empty element")