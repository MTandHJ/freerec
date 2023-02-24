

from typing import Iterable, Union, Any, Callable, List, Tuple

import torchdata.datapipes as dp
from itertools import repeat, chain


__all__ = [
    "DropEmpty", 
    "LeftPruningRow", "RightPruningRow",
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

        def _check(x: Any) -> bool:
            return (not isinstance(x, Iterable)) or len(x) > 0

        super().__init__(
            source_dp=source_dp,
            fn=_check,
            indices=indices
        )


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

    def _apply_fn(self, row: Union[List, Tuple]):
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


dp.iter.Mapper
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
        def _lprune(x: Iterable) -> Iterable:
            return x[-maxlen:]

        super().__init__(
            source_dp=source_dp, 
            fn=_lprune,
            indices=indices
        )


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
        def _rprune(x):
            return x[:maxlen]

        super().__init__(
            source_dp=source_dp, 
            fn=_rprune,
            indices=indices
        )


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

        def _lshift(x):
            return list(map(lambda item: item - offset, x))

        super().__init__(
            source_dp=source_dp, 
            fn=_lshift,
            indices=indices,
        )


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

        def _rshift(x):
            return list(map(lambda item: item + offset, x))

        super().__init__(
            source_dp=source_dp, 
            fn=_rshift,
            indices=indices
        )


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
    max_len : int
        The maximum length to pad the sequences to.
    padding_value : int, optional (default=0)
        The value to use for padding.
    """

    def __init__(
        self, source_dp: dp.iter.IterDataPipe, 
        indices: Union[None, int, Iterable[int]],
        maxlen: int, padding_value: int = 0,
    ) -> None:

        def _lpad(x):
            return list(chain(repeat(padding_value, maxlen - len(x)), x))

        super().__init__(
            source_dp=source_dp, 
            fn=_lpad,
            indices=indices
        )


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
    max_len : int
        The maximum length to right pad the sequences to.
    padding_value : int, optional (default=0)
        The value to use for padding.
    """

    def __init__(
        self, source_dp: dp.iter.IterDataPipe, 
        indices: Union[None, int, Iterable[int]],
        maxlen: int, padding_value: int = 0,
    ) -> None:

        def _rpad(x):
            return list(chain(x, repeat(padding_value, maxlen - len(x))))

        super().__init__(
            source_dp=source_dp, 
            fn=_rpad,
            indices=indices
        )