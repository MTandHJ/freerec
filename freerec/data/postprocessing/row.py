

from typing import Iterable, Union, Any, Callable, List, Tuple

import torchdata.datapipes as dp
from itertools import repeat, chain


__all__ = [
    "DropEmpty", 
    "LeftPruningRow", "RightPruningRow",
    "DropingDuplicates",
    "LeftShiftingRow", "RightShiftingRow",
    "LeftPaddingRow", "RightPaddingRow",
]


#==================================Filter==================================
class RowFilter(dp.iter.IterDataPipe):
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
        self, source_dp: dp.iter.IterableWrapper,
        fn: Callable, indices: Iterable[int]
    ):
        super().__init__()

        assert isinstance(indices, Iterable), \
            f"{self.__class__.__name__} requires iterable indices but {type(indices)} recevied ..."

        self.source = source_dp
        self.fn = fn
        self.indices = sorted(set(indices))

    def _apply_fn(self, row: Union[List, Tuple]):
        r"""
        Apply the specified function to the elements of the row at the specified indices.

        Parameters:
        -----------
        row : list or tuple
            The row of data to check.

        Returns:
        --------
        bool
            The final results over specified indices.
        """
        return all(self.fn(row[i]) for i in self.indices)

    def __iter__(self):
        for row in self.source:
            if self._apply_fn(row):
                yield row


@dp.functional_datapipe("drop_empty_")
class DropEmpty(RowFilter):
    r"""
    A functional datapipe that drops empty data from a given datapipe.

    Parameters:
    -----------
    source_dp: IterDataPipe 
        The input datapipe to filter.
    indices : Iterable[int]
        The indices of the elements in each row to which `fn` should be applied.
    """
    def __init__(
        self, 
        source_dp: dp.iter.IterDataPipe, 
        indices: Iterable[int]
    ) -> None:

        super().__init__(
            source_dp=source_dp,
            fn=self._check,
            indices=indices
        )

    def _check(self, x: Any) -> bool:
        return (not isinstance(x, Iterable)) or len(x) > 0


#==================================Mapper==================================
class RowMapper(dp.iter.IterDataPipe):
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
        self, source_dp: dp.iter.IterableWrapper,
        fn: Callable, indices: Iterable[int]
    ):
        super().__init__()

        assert isinstance(indices, Iterable), \
            f"{self.__class__.__name__} requires iterable indices but {type(indices)} recevied ..."

        self.source = source_dp
        self.fn = fn
        self.indices = sorted(set(indices))

    def _apply_fn(self, row: Union[List, Tuple]) -> List:
        r"""
        Apply the specified function to the elements of the row at the specified indices.

        Parameters:
        -----------
        row : list or tuple
            The row of data to transform.

        Returns:
        --------
        list
            The transformed row.
        """
        row = row if isinstance(row, list) else list(row)
        for i in self.indices:
            row[i] = self.fn(row[i])
        return row

    def __iter__(self):
        for row in self.source:
            yield self._apply_fn(row)


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
        self, source_dp: dp.iter.IterDataPipe, 
        indices: Union[None, int, Iterable[int]],
        maxlen: int
    ) -> None:

        self.maxlen = maxlen

        super().__init__(
            source_dp=source_dp, 
            fn=self._lprune,
            indices=indices
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
        self, source_dp: dp.iter.IterDataPipe, 
        indices: Union[None, int, Iterable[int]],
        maxlen: int
    ) -> None:

        self.maxlen = maxlen

        super().__init__(
            source_dp=source_dp, 
            fn=self._rprune,
            indices=indices
        )

    def _rprune(self, x: Iterable) -> Iterable:
        return x[:self.maxlen]


@dp.functional_datapipe("drop_duplicates_")
class DropingDuplicates(RowMapper):
    r"""
    A functional datapipe that drop the duplicates.

    Parameters:
    -----------
    source_dp: IterDataPipe 
        The input datapipe to prune.
    indices : Iterable[int]
        The indices of the elements in each row to which `fn` should be applied.
    ordered: bool
        `True`: keeping order.
    """

    def __init__(
        self, source_dp: dp.iter.IterDataPipe, 
        indices: Union[None, int, Iterable[int]],
        ordered: bool = True
    ) -> None:

        self.ordered = ordered

        super().__init__(
            source_dp=source_dp, 
            fn=self._drop_in_order if self.ordered else self._drop,
            indices=indices
        )

    def _drop(self, x: Iterable) -> List:
        return list(set(x))

    def _drop_in_order(self, x: Iterable) -> List:
        return sorted(set(x), key=x.index)


@dp.functional_datapipe("lshift_")
class LeftShiftingRow(RowMapper):
    r"""
    Mapper that left shifts the input data by a specified offset.

    Parameters:
    -----------
    source_dp: dp.iter.IterDataPipe
        Input datapipeline.
    indices : Iterable[int]
        The indices of the elements in each row to which `fn` should be applied.
    offset: int
        Amount to shift the input data by.   
    """

    def __init__(
        self, source_dp: dp.iter.IterDataPipe, 
        indices: Union[None, int, Iterable[int]],
        offset: int
    ) -> None:

        super().__init__(
            source_dp=source_dp, 
            fn=self._lshift,
            indices=indices,
        )

        self.offset = offset

    def _reduce(self, item: int):
        return item - self.offset

    def _lshift(self, x: Iterable) -> List:
        return list(map(self._reduce, x))


@dp.functional_datapipe("rshift_")
class RightShiftingRow(RowMapper):
    r"""
    Mapper that right shifts the input data by a specified offset.

    Parameters:
    -----------
    source_dp: dp.iter.IterDataPipe
        Input datapipeline.
    indices : Iterable[int]
        The indices of the elements in each row to which `fn` should be applied.
    offset: int
        Amount to shift the input data by.   
    """

    def __init__(
        self, source_dp: dp.iter.IterDataPipe, 
        indices: Union[None, int, Iterable[int]],
        offset: int
    ) -> None:

        self.offset = offset

        super().__init__(
            source_dp=source_dp, 
            fn=self._rshift,
            indices=indices
        )

    def _add(self, item: int):
        return item + self.offset

    def _rshift(self, x: Iterable) -> List:
        return list(map(self._add, x))


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
    """

    def __init__(
        self, source_dp: dp.iter.IterDataPipe, 
        indices: Union[None, int, Iterable[int]],
        maxlen: int, padding_value: int = 0,
    ) -> None:

        self.maxlen = maxlen
        self.padding_value = padding_value

        super().__init__(
            source_dp=source_dp, 
            fn=self._lpad,
            indices=indices
        )

    def _lpad(self, x: Iterable) -> List:
        return list(chain(repeat(self.padding_value, self.maxlen - len(x)), x))


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
    """

    def __init__(
        self, source_dp: dp.iter.IterDataPipe, 
        indices: Union[None, int, Iterable[int]],
        maxlen: int, padding_value: int = 0,
    ) -> None:

        self.maxlen = maxlen
        self.padding_value = padding_value

        super().__init__(
            source_dp=source_dp, 
            fn=self._rpad,
            indices=indices
        )

    def _rpad(self, x: Iterable) -> List:
        return list(chain(x, repeat(self.padding_value, self.maxlen - len(x))))