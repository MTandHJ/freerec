from itertools import chain, repeat
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
import torchdata.datapipes as dp

from ..fields import Field
from .base import PostProcessor

__all__ = [
    "LeftPruningRow",
    "RightPruningRow",
    "AddingRow",
    "LeftPaddingRow",
    "RightPaddingRow",
]


# ==================================Filter==================================
class RowFilter(PostProcessor):
    r"""Filter rows by applying a predicate to specified fields.

    Parameters
    ----------
    source : :class:`~PostProcessor`
        The source datapipe.
    fn : callable
        A function ``fn(field, value) -> bool`` applied to each checked field.
    checked_fields : iterable of :class:`~Field`
        The fields to be checked.

    Raises
    ------
    TypeError
        If ``checked_fields`` is not an iterable.
    """

    def __init__(
        self,
        source: dp.iter.IterableWrapper,
        fn: Callable,
        checked_fields: Iterable[Field],
    ):
        r"""Initialize the RowFilter."""
        super().__init__(source)
        self.fn = fn
        self.checked_fields = set(checked_fields)

    def _check(self, row: Dict[Field, Any]) -> bool:
        r"""Return whether all checked fields pass the predicate."""
        return all(
            self.fn(field, row.get(field, None)) for field in self.checked_fields
        )

    def __iter__(self):
        r"""Yield rows that pass the filter."""
        for row in self.source:
            if self._check(row):
                yield row


# ==================================Mapper==================================
class RowMapper(PostProcessor):
    r"""Apply a mapping function to specified fields of each row.

    Parameters
    ----------
    source : :class:`~PostProcessor`
        The source datapipe.
    fn : callable
        A function ``fn(field, value) -> new_value`` applied to each modified field.
    modified_fields : iterable of :class:`~Field`
        The fields to be modified.

    Raises
    ------
    TypeError
        If ``modified_fields`` is not an iterable.
    """

    def __init__(
        self,
        source: dp.iter.IterableWrapper,
        fn: Callable,
        modified_fields: Iterable[Field],
    ):
        r"""Initialize the RowMapper."""
        super().__init__(source)
        self.fn = fn
        self.modified_fields = set(modified_fields)

    def __iter__(self):
        r"""Yield rows with the specified fields transformed."""
        for row in self.source:
            for field in self.modified_fields:
                row[field] = self.fn(field, row[field])
            yield row


@dp.functional_datapipe("lprune_")
class LeftPruningRow(RowMapper):
    r"""Prune sequences from the left to a maximum length.

    Parameters
    ----------
    source : :class:`~IterDataPipe`
        The source datapipe.
    maxlen : int
        The maximum length to prune the input data to.
    modified_fields : iterable of :class:`~Field`
        The fields to be modified.

    Notes
    -----
    ::

        [1, 2, 3, 4] --(maxlen=3)--> [2, 3, 4]
        [3, 4]       --(maxlen=3)--> [3, 4]

    Examples
    --------
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
        self,
        source: dp.iter.IterDataPipe,
        maxlen: int,
        modified_fields: Iterable[Field],
    ) -> None:
        r"""Initialize the LeftPruningRow."""

        self.maxlen = maxlen

        super().__init__(source, self._prune, modified_fields)

    def _prune(self, field: Field, x: Iterable) -> Iterable:
        r"""Return the last ``maxlen`` elements of *x*."""
        return x[-self.maxlen :]


@dp.functional_datapipe("rprune_")
class RightPruningRow(LeftPruningRow):
    r"""Prune sequences from the right to a maximum length.

    Parameters
    ----------
    source : :class:`~IterDataPipe`
        The source datapipe.
    maxlen : int
        The maximum length to prune the input data to.
    modified_fields : iterable of :class:`~Field`
        The fields to be modified.

    Notes
    -----
    ::

        [1, 2, 3, 4] --(maxlen=3)--> [1, 2, 3]
        [3, 4]       --(maxlen=3)--> [3, 4]

    Examples
    --------
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
        r"""Return the first ``maxlen`` elements of *x*."""
        return x[: self.maxlen]


@dp.functional_datapipe("add_")
class AddingRow(RowMapper):
    r"""Add a constant offset to specified fields.

    Parameters
    ----------
    source : :class:`~IterDataPipe`
        The source datapipe.
    offset : int
        Amount to add to the input data.
    modified_fields : iterable of :class:`~Field`
        The fields to be modified.

    Notes
    -----
    ::

        [1, 2, 3, 4] --(offset=1)--> [2, 3, 4, 5]

    Examples
    --------
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
        self,
        source: dp.iter.IterDataPipe,
        offset: int,
        modified_fields: Iterable[Field],
    ) -> None:
        r"""Initialize the AddingRow."""

        self.offset = offset

        super().__init__(source, self._add, modified_fields)

    def _add(self, field: Field, x: Iterable) -> List:
        r"""Add ``self.offset`` element-wise to *x*.

        Examples
        --------
        >>> x = [1, 2, 3]
        >>> _add(x)  # self.offset = 1
        [2, 3, 4]
        >>> _add(x)  # self.offset = -1
        [0, 1, 2]
        """
        return np.add(x, self.offset).tolist()


@dp.functional_datapipe("lpad_")
class LeftPaddingRow(RowMapper):
    r"""Left-pad sequences to a maximum length.

    Parameters
    ----------
    source : :class:`~IterDataPipe`
        The source datapipe.
    maxlen : int
        The maximum length to pad the sequences to.
    modified_fields : iterable of :class:`~Field`
        The fields to be modified.
    padding_value : int, optional
        The value to use for padding. Default is ``0``.

    Notes
    -----
    ::

        [1, 2, 3, 4] --(maxlen=7, padding_value=0)--> [0, 0, 0, 1, 2, 3, 4]
        [1, 2, 3, 4] --(maxlen=4, padding_value=0)--> [1, 2, 3, 4]

    Examples
    --------
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
        self,
        source: dp.iter.IterDataPipe,
        maxlen: int,
        modified_fields: Iterable[Field],
        padding_value: int = 0,
    ) -> None:
        r"""Initialize the LeftPaddingRow."""

        self.maxlen = maxlen
        self.padding_value = int(padding_value)

        super().__init__(source, self._pad, modified_fields)

        self.sure_zero_elems()

    def sure_zero_elems(self):
        r"""Infer the padding element shape from the first row."""

        def guess_zero(field: Field, value: Iterable):
            if isinstance(value, Iterable):
                if len(value) == 0:
                    return self.padding_value
                elif isinstance(value[0], Iterable):
                    return (self.padding_value,) * len(value[0])
                else:
                    return self.padding_value
            else:
                raise ValueError(f"{value} for {field} is non-iterable ...")

        row = next(iter(self.source))
        self.zeros = dict()
        for field in self.modified_fields:
            self.zeros[field] = guess_zero(field, row[field])

    def _pad(self, field: Field, x: Iterable) -> List:
        r"""Left-pad *x* to ``self.maxlen`` with the appropriate zero element."""
        return list(chain(repeat(self.zeros[field], self.maxlen - len(x)), x))


@dp.functional_datapipe("rpad_")
class RightPaddingRow(LeftPaddingRow):
    r"""Right-pad sequences to a maximum length.

    Parameters
    ----------
    source : :class:`~IterDataPipe`
        The source datapipe.
    maxlen : int
        The maximum length to pad the sequences to.
    modified_fields : iterable of :class:`~Field`
        The fields to be modified.
    padding_value : int, optional
        The value to use for padding. Default is ``0``.

    Notes
    -----
    ::

        [1, 2, 3, 4] --(maxlen=7, padding_value=0)--> [1, 2, 3, 4, 0, 0, 0]
        [1, 2, 3, 4] --(maxlen=4, padding_value=0)--> [1, 2, 3, 4]

    Examples
    --------
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
        r"""Right-pad *x* to ``self.maxlen`` with the appropriate zero element."""
        return list(chain(x, repeat(self.zeros[field], self.maxlen - len(x))))
