

from typing import Iterable, Union, Any, Callable, List, Tuple, Dict

import numpy as np
import torchdata.datapipes as dp
from itertools import repeat, chain

from .base import PostProcessor
from ..fields import Field


__all__ = [
    "LeftPruningRow", "RightPruningRow",
    "AddingRow",
    "LeftPaddingRow", "RightPaddingRow",
]


#==================================Filter==================================
class RowFilter(PostProcessor):
    r"""
    Apply a function to specific indices of each row in an IterableDataset.

    Parameters:
    -----------
    source_dp : tud.IterableDataset
        The source IterableDataset.
    fn : Callable
        The function to apply to the specified indices of each row.
    checked_fields : Iterable[Field]
        The fields to be checked.

    Raises:
    -------
    TypeError
        If `indices` is not an iterable.
    """

    def __init__(
        self, source: dp.iter.IterableWrapper,
        fn: Callable, checked_fields: Iterable[Field]
    ):
        super().__init__(source)
        self.fn = fn
        self.checked_fields = set(self.sure_input_fields()) & set(checked_fields)

    def _check(self, row: Dict[Field, Any]) -> bool:
        return all(self.fn(field, row[field]) for field in self.checked_fields)

    def __iter__(self):
        for row in self.source:
            if self._check(row):
                yield row


#==================================Mapper==================================
class RowMapper(PostProcessor):
    r"""
    Apply a function to specific indices of each row in an IterableDataset.

    Parameters:
    -----------
    source_dp : tud.IterableDataset
        The source IterableDataset.
    fn : Callable
        The function to apply to the specified indices of each row.
    modified_fields : Iterable[Field]
        The fields to be modified.

    Raises:
    -------
    TypeError
        If `indices` is not an iterable.
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
                row[field] = self.fn(field, row[field])
            yield row


@dp.functional_datapipe("lprune_")
class LeftPruningRow(RowMapper):
    r"""
    A functional datapipe that prunes the left side of a given datapipe to a specified maximum length.

    Parameters:
    -----------
    maxlen: int 
        The maximum length to prune the input data to.
    modifields_fields: Iterable[Field]
        The fields to be modified.

    Flows:
    ------
    [1, 2, 3, 4] --(maxlen=3)--> [2, 3, 4]
    [3, 4] --(maxlen=3)--> [3, 4]

    Examples:
    ---------
    >>> dataset: RecDataSet
    >>> ISeq = dataset[ITEM, ID].fork(SEQUENCE)
    >>> datapipe = dataset.valid().ordered_user_ids_source(
    ).valid_sampling_().lprune_(
        3, modified_fields=(ISeq,)
    )
    >>> next(iter(datapipe))
    {Field(USER:ID,USER): 0,
    Field(ITEM:ID,ITEM,SEQUENCE): (9839, 10076, 11155),
    Field(ITEM:ID,ITEM,UNSEEN): (11752,),
    Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
    """

    def __init__(
        self, source: dp.iter.IterDataPipe, maxlen: int, modified_fields: Iterable[Field]
    ) -> None:

        self.maxlen = maxlen

        super().__init__(
            source, self._prune,
            modified_fields
        )

    def _prune(self, field: Field, x: Iterable) -> Iterable:
        return x[-self.maxlen:]


@dp.functional_datapipe("rprune_")
class RightPruningRow(LeftPruningRow):
    r"""
    A functional datapipe that prunes the right side of a given datapipe to a specified maximum length.

    Parameters:
    -----------
    maxlen: int 
        The maximum length to prune the input data to.
    modifields_fields: Iterable[Field]
        The fields to be modified.

    Flows:
    ------
    [1, 2, 3, 4] --(maxlen=3)--> [1, 2, 3]
    [3, 4] --(maxlen=3)--> [3, 4]

    Examples:
    ---------
    >>> dataset: RecDataSet
    >>> ISeq = dataset[ITEM, ID].fork(SEQUENCE)
    >>> datapipe = dataset.valid().ordered_user_ids_source(
    ).valid_sampling_().rprune_(
        3, modified_fields=(ISeq,)
    )
    >>> next(iter(datapipe))
    {Field(USER:ID,USER): 0,
    Field(ITEM:ID,ITEM,SEQUENCE): (9449, 9839, 10076),
    Field(ITEM:ID,ITEM,UNSEEN): (11752,),
    Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
    """

    def _prune(self, field: Field, x: Iterable) -> Iterable:
        return x[:self.maxlen]


@dp.functional_datapipe("add_")
class AddingRow(RowMapper):
    r"""
    Mapper that adds the input data by a specified offset.

    Parameters:
    -----------
    offset: int
        Amount to add the input data by.   
    modifields_fields: Iterable[Field]
        The fields to be modified.

    Flows:
    ------
    [1, 2, 3, 4] --(offset=1)--> [2, 3, 4, 5]

    Examples:
    ---------
    >>> dataset: RecDataSet
    >>> ISeq = dataset[ITEM, ID].fork(SEQUENCE)
    >>> datapipe = dataset.valid().ordered_user_ids_source(
    ).valid_sampling_().rprune_(
        3, modified_fields=(ISeq,)
    ).add_(
        1, modified_fields=(ISeq,)
    )
    """

    def __init__(
        self, source: dp.iter.IterDataPipe, offset: int, modified_fields: Iterable[Field]
    ) -> None:

        self.offset = offset

        super().__init__(
            source, self._add,
            modified_fields
        )

    def _add(self, field: Field, x: Iterable) -> List:
        r"""
        Examples:
        ---------
        >>> x = [1, 2, 3] 
        >>> _add(x) # self.offset = 1
        [2, 3, 4]
        >>> _add(x) # self.offset = -1
        [0, 1, 2]
        """
        return (np.array(x, copy=False) + self.offset).tolist()


@dp.functional_datapipe("lpad_")
class LeftPaddingRow(RowMapper):
    r"""
    A functional data pipeline component that left pads sequences to a maximum length.

    Parameters:
    -----------
    maxlen : int
        The maximum length to pad the sequences to.
    modifields_fields: Iterable[Field]
        The fields to be modified.
    padding_value : int, optional (default=0)
        The value to use for padding.

    Flows:
    ------
    [1, 2, 3, 4] --(maxlen=7, padding_value=0)--> [0, 0, 0, 1, 2, 3, 4]
    [1, 2, 3, 4] --(maxlen=4, padding_value=0)--> [1, 2, 3, 4]

    Examples:
    ---------
    >>> dataset: RecDataSet
    >>> ISeq = dataset[ITEM, ID].fork(SEQUENCE)
    >>> datapipe = dataset.valid().ordered_user_ids_source(
    ).valid_sampling_().lprune_(
        3, modified_fields=(ISeq,)
    ).add_(
        1, modified_fields=(ISeq,)
    ).lpad_(
        5, modified_fields=(ISeq,)
    )
    >>> next(iter(datapipe))
    {Field(USER:ID,USER): 0,
    Field(ITEM:ID,ITEM,SEQUENCE): [0, 0, 9840, 10077, 11156],
    Field(ITEM:ID,ITEM,UNSEEN): (11752,),
    Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
    """

    def __init__(
        self, source: dp.iter.IterDataPipe, maxlen: int, modified_fields: Iterable[Field], padding_value: int = 0
    ) -> None:

        self.maxlen = maxlen
        self.padding_value = int(padding_value)

        super().__init__(
            source, self._pad,
            modified_fields
        )

        self.sure_zero_elems()

    def sure_zero_elems(self):
        def guess_zero(field: Field, value: Iterable):
            if isinstance(value, Iterable):
                if isinstance(value[0], Iterable):
                    return (self.padding_value,) * len(value[0])
                else:
                    return self.padding_value
            else:
                raise ValueError(f"{value} for {field} is not non-iterable ...")

        row = next(iter(self.source))
        self.zeros = dict()
        for field in self.modified_fields:
            self.zeros[field] = guess_zero(field, row[field])

    def _pad(self, field: Field, x: Iterable) -> List:
        return list(chain(repeat(self.zeros[field], self.maxlen - len(x)), x))


@dp.functional_datapipe("rpad_")
class RightPaddingRow(LeftPaddingRow):
    r"""
    A functional data pipeline component that right pads sequences to a maximum length.

    Parameters:
    -----------
    maxlen : int
        The maximum length to pad the sequences to.
    modifields_fields: Iterable[Field]
        The fields to be modified.
    padding_value : int, optional (default=0)
        The value to use for padding.

    Flows:
    ------
    [1, 2, 3, 4] --(maxlen=7, padding_value=0)--> [1, 2, 3, 4, 0, 0, 0]
    [1, 2, 3, 4] --(maxlen=4, padding_value=0)--> [1, 2, 3, 4]

    Examples:
    ---------
    >>> dataset: RecDataSet
    >>> ISeq = dataset[ITEM, ID].fork(SEQUENCE)
    >>> datapipe = dataset.valid().ordered_user_ids_source(
    ).valid_sampling_().rprune_(
        3, modified_fields=(ISeq,)
    ).add_(
        1, modified_fields=(ISeq,)
    ).rpad_(
        5, modified_fields=(ISeq,)
    )
    >>> next(iter(datapipe))
    {Field(USER:ID,USER): 0,
    Field(ITEM:ID,ITEM,SEQUENCE): [9450, 9840, 10077, 0, 0],
    Field(ITEM:ID,ITEM,UNSEEN): (11752,),
    Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
    """

    def _pad(self, field: Field, x: Iterable) -> List:
        return list(chain(x, repeat(self.zeros[field], self.maxlen - len(x))))