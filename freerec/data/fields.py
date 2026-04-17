from itertools import chain
from typing import Iterable, Iterator, Literal, Tuple, TypeVar, Union

import numpy as np
import polars as pl
import torch

from .normalizer import NORMALIZERS
from .tags import FieldTags

__all__ = ["Field", "FieldTuple", "FieldModule", "FieldModuleList"]


T = TypeVar("T")


class Field:
    r"""A field determined by a name and a collection of tags.

    A :class:`~Field` represents a single column or feature in a dataset,
    identified by a human-readable *name* and annotated with one or more
    :class:`~FieldTags`.  Fields support tag-based matching, forking
    (creating derived fields), normalization, and conversion to sparse
    tensors.

    Parameters
    ----------
    name : str
        Human-readable name for the field (e.g. ``"User"``).
    *tags : :class:`~FieldTags`
        One or more tags that describe the field.

    Examples
    --------
    >>> User = Field('User', USER)
    >>> User
    Field(User:USER)
    >>> UserID = User.fork(ID)
    >>> UserID
    Field(User:ID,USER)
    >>> Field.issubfield(UserID, User)
    True
    >>> Field.issuperfield(User, UserID)
    True
    >>> UserID.match(USER)
    True
    >>> UserID.match(ID)
    True
    >>> UserID.match(UserID, ITEM)
    False
    >>> UserID.match_all()
    True
    >>> UserID.match_any()
    False
    >>> list(UserID)
    [<FieldTags.ID: 'ID'>, <FieldTags.USER: 'USER'>]
    >>> UserID1 = Field('UserID1', USER, ID)
    >>> UserID2 = Field('UserID2', USER, ID)
    >>> UserID1 == UserID2
    False
    """

    def __init__(self, name: str, *tags: FieldTags) -> None:
        r"""Initialize field with a name and tags."""
        self.__name = str(name)
        self.__tags = set(tags)
        self.__identifier = (name,) + tuple(
            sorted(self.__tags, key=lambda tag: tag.value)
        )
        self.__hash_value = hash(self.identifier)
        self.count = None

    @property
    def name(self) -> str:
        r"""Return the human-readable name of the field."""
        return self.__name

    @property
    def tags(self) -> Tuple:
        r"""Return the sorted tags of the field as a tuple."""
        return self.__identifier[1:]

    @property
    def identifier(self) -> Tuple:
        r"""Return the full identifier tuple ``(name, *sorted_tags)``."""
        return self.__identifier

    def fork(self: T, *tags: FieldTags) -> T:
        r"""Create a derived field by appending additional tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Additional tags to include in the forked field.

        Returns
        -------
        :class:`~Field`
            A new field that carries the original name, the original tags,
            and the newly supplied tags.
        """
        field = type(self)(self.name, *self.tags, *tags)
        field.count = self.count
        return field

    def to_module(self) -> "FieldModule":
        r"""Convert this field to a :class:`~FieldModule`.

        Returns
        -------
        :class:`~FieldModule`
            A module-wrapped copy of this field.
        """
        field = FieldModule(self.name, *self.tags)
        field.count = self.count
        return field

    def match(self, *tags: FieldTags) -> bool:
        r"""Check whether this field matches all given tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags to check against.

        Returns
        -------
        bool
            ``True`` if the field's tag set is a superset of *tags*.
        """
        return self.__tags.issuperset(tags)

    def match_all(self, *tags: FieldTags) -> bool:
        r"""Check whether this field matches all given tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags to check against.

        Returns
        -------
        bool
            ``True`` if the field's tag set is a superset of *tags*.
        """
        return self.match(*tags)

    def match_any(self, *tags: FieldTags) -> bool:
        r"""Check whether this field matches any of the given tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags to check against.

        Returns
        -------
        bool
            ``True`` if the field matches at least one of the given tags.
        """
        return any(self.match(tag) for tag in tags)

    def issubfield(self, other: "Field") -> bool:
        r"""Check whether this field is a sub-field of *other*.

        Parameters
        ----------
        other : :class:`~Field`
            The candidate super-field.

        Returns
        -------
        bool
            ``True`` if this field's tags are a superset of *other*'s tags.
        """
        return self.match(*other.tags)

    def issuperfield(self, other: "Field") -> bool:
        r"""Check whether this field is a super-field of *other*.

        Parameters
        ----------
        other : :class:`~Field`
            The candidate sub-field.

        Returns
        -------
        bool
            ``True`` if *other*'s tags are a superset of this field's tags.
        """
        return other.match(*self.tags)

    def __hash__(self) -> int:
        r"""Return the pre-computed hash based on the identifier."""
        return self.__hash_value

    def __eq__(self, other) -> bool:
        r"""Check equality by comparing hashes."""
        return isinstance(other, Field) and hash(self) == hash(other)

    def __repr__(self) -> str:
        r"""Return a developer-friendly string representation."""
        return f"{self.__class__.__name__}({str(self)})"

    def __str__(self) -> str:
        r"""Return a concise string of the form ``name:count:TAG1,TAG2``."""
        return (
            f"{self.name}:{self.count}:{','.join(map(lambda tag: tag.name, self.tags))}"
        )

    def set_normalizer(
        self,
        dtype: Union[None, str, pl.DataType] = None,
        fill_null_strategy: Literal[
            "forward", "backward", "min", "max", "zero", "one"
        ] = "zero",
        normalizer: Union[None, Literal["standard", "minmax", "reindex"]] = None,
        **kwargs,
    ):
        r"""Configure casting, null-filling and normalization for this field.

        Parameters
        ----------
        dtype : str, :class:`polars.DataType`, or None
            Target data type.  When a string is given it is resolved via
            ``getattr(pl, dtype)``.  ``None`` means no casting.
        fill_null_strategy : {'forward', 'backward', 'min', 'max', 'zero', 'one'}
            Strategy passed to :meth:`polars.Series.fill_null`.
        normalizer : {'standard', 'minmax', 'reindex'} or None
            Normalization method.  ``None`` falls back to a simple counter.
        **kwargs
            Extra keyword arguments forwarded to the normalizer constructor.

        Raises
        ------
        KeyError
            If *normalizer* is not registered or *kwargs* are invalid.
        """
        self._dtype = getattr(pl, dtype) if isinstance(dtype, str) else dtype
        self._fill_null_strategy = fill_null_strategy

        normalizer = "counter" if normalizer is None else normalizer
        try:
            self._normalizer = NORMALIZERS[normalizer.upper()](**kwargs)
        except KeyError:
            availables = "; ".join(NORMALIZERS.keys())
            raise KeyError(
                f"Receive an invalid normalizer not existing in: [{availables}]. "
                f"You should register this via `register_normalizer(normalizer, name)` ..."
            )
        except TypeError:
            raise KeyError(f"Receive invalid kwargs for {normalizer}: {kwargs}")

    def cast(self, data: pl.Series, strict: bool = False) -> pl.Series:
        r"""Cast and null-fill a :class:`polars.Series`.

        Parameters
        ----------
        data : :class:`polars.Series`
            Input series.
        strict : bool, optional
            Whether to use strict casting (default ``False``).

        Returns
        -------
        :class:`polars.Series`
            The series after optional casting, NaN-to-null conversion, and
            null filling.
        """
        if self._dtype is not None:
            data = data.cast(self._dtype, strict=strict)
        try:
            data = data.fill_nan(None)
        except Exception:
            # Skip `fill_nan` for String data
            pass
        finally:
            data = data.fill_null(strategy=self._fill_null_strategy)
            return data

    def fit(
        self, data: Union[pl.Series, pl.DataFrame, pl.LazyFrame], partial: bool = True
    ) -> pl.Series:
        r"""Fit the normalizer on *data* and return the cast series.

        Parameters
        ----------
        data : :class:`polars.Series`, :class:`polars.DataFrame`, or :class:`polars.LazyFrame`
            Input data.  DataFrames and LazyFrames are converted to a
            :class:`polars.Series` before processing.
        partial : bool, optional
            If ``True`` (default), partially fit on the given data.
            If ``False``, reset the normalizer before fitting.

        Returns
        -------
        :class:`polars.Series`
            The cast (but not normalized) series.
        """
        if isinstance(data, pl.LazyFrame):
            data = data.collect().to_series()
        elif isinstance(data, pl.DataFrame):
            data = data.to_series()

        data = self.cast(data)

        if not partial:
            self._normalizer.reset()
        self._normalizer.partial_fit(data)

        try:
            self.count = self._normalizer.count
        except AttributeError:
            pass

        return data

    def normalize(
        self,
        data: Union[pl.Series, pl.DataFrame, pl.LazyFrame],
    ) -> pl.Series:
        r"""Cast and normalize *data* using the fitted normalizer.

        The processing pipeline is: ``data -> cast -> normalization -> data``.

        Parameters
        ----------
        data : :class:`polars.Series`, :class:`polars.DataFrame`, or :class:`polars.LazyFrame`
            Input data.

        Returns
        -------
        :class:`polars.Series`
            The normalized series.
        """
        if isinstance(data, pl.LazyFrame):
            data = data.collect().to_series()
        elif isinstance(data, pl.DataFrame):
            data = data.to_series()

        data = self.cast(data)
        data = self._normalizer(data)
        return data

    def to_csr(self, data: Iterable) -> torch.Tensor:
        r"""Convert a batch of variable-length index lists to a CSR tensor.

        Parameters
        ----------
        data : iterable of list or tuple
            A 2-D iterable where each inner sequence contains column indices.

        Returns
        -------
        :class:`torch.Tensor`
            A sparse CSR tensor of shape ``(B, self.count)`` with ones at
            the indicated positions.

        Examples
        --------
        >>> Item: Field
        >>> Item.count = 5
        >>> data = [[1, 2], [3, 4]]
        >>> Item.to_csr(data)
        tensor(crow_indices=tensor([0, 2, 4]),
            col_indices=tensor([1, 2, 3, 4]),
            values=tensor([1, 1, 1, 1]), size=(2, 5), nnz=4,
            layout=torch.sparse_csr)
        """
        if isinstance(data, (torch.Tensor, np.ndarray)):
            data = data.tolist()
        assert isinstance(data[0], (list, tuple)), (
            f"Each row of data should be `list'|`tuple' but `{type(data[0])}' received ..."
        )

        crow_indices = np.cumsum([0] + list(map(len, data)), dtype=np.int64)
        col_indices = list(chain(*data))

        values = np.ones_like(col_indices, dtype=np.int64)
        return torch.sparse_csr_tensor(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            size=(len(data), self.count),  # B x Num of Items
        )


class FieldModule(Field, torch.nn.Module):
    r"""A :class:`~Field` that is also a :class:`torch.nn.Module`.

    This allows a field to own learnable parameters (e.g. embedding
    tables) and participate in the standard PyTorch parameter-collection
    mechanism.

    Parameters
    ----------
    name : str
        Human-readable name for the field.
    *tags : :class:`~FieldTags`
        One or more tags that describe the field.

    Notes
    -----
    The fixed hash value inherited from :class:`~Field` means that two
    :class:`~FieldModule` instances with the same name and tags will hash
    equally.  When both are registered as sub-modules of the same parent
    :class:`torch.nn.Module`, only one will be discovered by
    :meth:`~torch.nn.Module.parameters` because duplicates are removed
    during collection.

    For example, in the class below, ``self.field`` and ``self.field2``
    share the same hash, so only the parameters of ``self.field`` will
    be found::

        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.field = FieldModule('a').fork()
                self.field2 = FieldModule('a').fork()
                self.field.add_module("embeddings", nn.Embedding(3, 4))
                self.field2.add_module("embeddings", nn.Embedding(1, 4))
    """

    embeddings: torch.nn.Embedding

    def __init__(self, name: str, *tags: FieldTags) -> None:
        r"""Initialize both the Module and Field bases."""
        torch.nn.Module.__init__(self)
        Field.__init__(self, name, *tags)


class FieldTuple(tuple):
    r"""An immutable tuple of :class:`~Field` instances with tag-based access.

    Supports filtering by :class:`~FieldTags` and attribute-style look-up
    in addition to ordinary positional indexing.
    """

    def match(self, *tags: FieldTags) -> "FieldTuple":
        r"""Return fields that match all given tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags to match against.

        Returns
        -------
        :class:`~FieldTuple`
            Subset of fields whose tags are a superset of *tags*.
        """
        return FieldTuple(field for field in self if field.match(*tags))

    def match_all(self, *tags: FieldTags) -> "FieldTuple":
        r"""Return fields that match all given tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags to match against.

        Returns
        -------
        :class:`~FieldTuple`
            Subset of fields whose tags are a superset of *tags*.
        """
        return FieldTuple(field for field in self if field.match_all(*tags))

    def match_any(self, *tags: FieldTags) -> "FieldTuple":
        r"""Return fields that match any of the given tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags to match against.

        Returns
        -------
        :class:`~FieldTuple`
            Subset of fields that match at least one of the given tags.
        """
        return FieldTuple(field for field in self if field.match_any(*tags))

    def match_not(self, *tags: FieldTags) -> "FieldTuple":
        r"""Return fields that do **not** match all given tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags to match against.

        Returns
        -------
        :class:`~FieldTuple`
            Subset of fields that fail to match all given tags simultaneously.
        """
        return FieldTuple(field for field in self if not field.match_all(*tags))

    def copy(self) -> "FieldTuple":
        r"""Return a shallow copy of this tuple.

        Returns
        -------
        :class:`~FieldTuple`
            A new :class:`~FieldTuple` containing the same fields.
        """
        return FieldTuple(self)

    def index(self, *tags) -> int:
        r"""Return the index of the field matching the given tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags identifying the target field.

        Returns
        -------
        int
            Positional index of the matching field.

        Examples
        --------
        >>> from freerec.data.tags import USER, ITEM, ID
        >>> User = SparseField('User', None, int, tags=(USER, ID))
        >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
        >>> fields = FieldTuple([User, Item])
        >>> fields.index(USER, ID)
        0
        >>> fields.index(ITEM, ID)
        1
        """
        return super().index(self[tags])

    def __iter__(self) -> Iterator[FieldModule]:
        r"""Iterate over the fields in the tuple."""
        return super().__iter__()

    def __getitem__(
        self, index: Union[int, str, slice, FieldTags, Iterable[FieldTags]]
    ) -> Union[Field, "FieldTuple", None]:
        r"""Retrieve fields by position, name, slice, or tag(s).

        Parameters
        ----------
        index : int, str, slice, :class:`~FieldTags`, or iterable of :class:`~FieldTags`
            - int: return the field at that position.
            - str: return the field whose name equals the string.
            - slice: return the fields at the given slice positions.
            - :class:`~FieldTags`: return fields matching that single tag.
            - iterable of :class:`~FieldTags`: return fields matching all
              given tags.

        Returns
        -------
        :class:`~Field`, :class:`~FieldTuple`, or None
            A single :class:`~Field` when exactly one field matches,
            ``None`` when no field matches, or a :class:`~FieldTuple`
            when multiple fields match.

        Examples
        --------
        >>> from freerec.data.tags import USER, ITEM, ID
        >>> User = Field('User', USER, ID)
        >>> Item = Field('Item', ITEM, ID)
        >>> fields = FieldTuple([User, Item])
        >>> fields[USER, ID] is User
        True
        >>> fields[0] is User
        True
        >>> fields[0:1] is User
        True
        >>> fields[Item, ID] is Item
        True
        >>> fields[1] is Item
        True
        >>> fields['Item'] is Item
        True
        >>> fields[1:] is Item
        True
        >>> len(fields[ID])
        2
        >>> len(fields[:])
        2
        >>> isinstance(fields[ID], FieldTuple)
        True
        """
        if isinstance(index, int):
            return super().__getitem__(index)
        elif isinstance(index, str):
            fields = FieldTuple(field for field in self if field.name == index)
        elif isinstance(index, slice):
            fields = FieldTuple(super().__getitem__(index))
        elif isinstance(index, FieldTags):
            fields = self.match(index)
        else:
            fields = self.match(*index)
        if len(fields) == 1:
            return fields[0]
        elif len(fields) == 0:
            return None  # for a safety purpose
        else:
            return fields


class FieldModuleList(torch.nn.ModuleList):
    r"""A :class:`torch.nn.ModuleList` that holds :class:`~FieldModule` instances.

    Provides the same tag-based filtering interface as :class:`~FieldTuple`
    while retaining PyTorch module semantics.

    Parameters
    ----------
    fields : iterable of :class:`~FieldModule`
        The field modules to store.

    Raises
    ------
    AssertionError
        If any element is not a :class:`~FieldModule`.
    """

    def __init__(self, fields: Iterable[FieldModule]) -> None:
        r"""Initialize with an iterable of :class:`~FieldModule` instances."""
        super().__init__(fields)
        assert all(isinstance(field, FieldModule) for field in self), (
            "'FieldModuleList' receives 'FieldModule' only ..."
        )

    def match(self, *tags: FieldTags) -> "FieldModuleList":
        r"""Return field modules that match all given tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags to match against.

        Returns
        -------
        :class:`~FieldModuleList`
            Subset of field modules whose tags are a superset of *tags*.
        """
        return FieldModuleList(field for field in self if field.match(*tags))

    def match_all(self, *tags: FieldTags) -> "FieldModuleList":
        r"""Return field modules that match all given tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags to match against.

        Returns
        -------
        :class:`~FieldModuleList`
            Subset of field modules whose tags are a superset of *tags*.
        """
        return FieldModuleList(field for field in self if field.match_all(*tags))

    def match_any(self, *tags: FieldTags) -> "FieldModuleList":
        r"""Return field modules that match any of the given tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags to match against.

        Returns
        -------
        :class:`~FieldModuleList`
            Subset of field modules that match at least one of the given tags.
        """
        return FieldModuleList(field for field in self if field.match_any(*tags))

    def match_not(self, *tags: FieldTags) -> "FieldModuleList":
        r"""Return field modules that do **not** match all given tags.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags to match against.

        Returns
        -------
        :class:`~FieldModuleList`
            Subset of field modules that fail to match all given tags
            simultaneously.
        """
        return FieldModuleList(field for field in self if not field.match_all(*tags))

    def insert(self, index: int, field: FieldModule) -> None:
        r"""Insert a :class:`~FieldModule` at the given position.

        Parameters
        ----------
        index : int
            Position before which to insert.
        field : :class:`~FieldModule`
            The field module to insert.

        Raises
        ------
        AssertionError
            If *field* is not a :class:`~FieldModule`.
        """
        assert isinstance(field, FieldModule), (
            "'FieldModuleList' receives 'FieldModule' only ..."
        )
        return super().insert(index, field)

    def append(self, field: FieldModule) -> "FieldModuleList":
        r"""Append a :class:`~FieldModule` to the end.

        Parameters
        ----------
        field : :class:`~FieldModule`
            The field module to append.

        Returns
        -------
        :class:`~FieldModuleList`
            This list (for chaining).

        Raises
        ------
        AssertionError
            If *field* is not a :class:`~FieldModule`.
        """
        assert isinstance(field, FieldModule), (
            "'FieldModuleList' receives 'FieldModule' only ..."
        )
        return super().append(field)

    def extend(self, fields: Iterable[FieldModule]) -> "FieldModuleList":
        r"""Extend the list with an iterable of :class:`~FieldModule` instances.

        Parameters
        ----------
        fields : iterable of :class:`~FieldModule`
            The field modules to add.

        Returns
        -------
        :class:`~FieldModuleList`
            This list (for chaining).

        Raises
        ------
        AssertionError
            If any element is not a :class:`~FieldModule`.
        """
        fields = list(fields)
        assert all(isinstance(field, FieldModule) for field in fields), (
            "'FieldModuleList' receives 'FieldModule' only ..."
        )
        return super().extend(fields)

    def __iter__(self) -> Iterator[FieldModule]:
        r"""Iterate over the field modules."""
        return super().__iter__()

    def __getitem__(
        self, index: Union[int, str, FieldTags, Iterable[FieldTags]]
    ) -> Union[FieldModule, "FieldModuleList", None]:
        r"""Retrieve field modules by position, name, slice, or tag(s).

        Parameters
        ----------
        index : int, str, slice, :class:`~FieldTags`, or iterable of :class:`~FieldTags`
            - int: return the field module at that position.
            - str: return the field module whose name equals the string.
            - slice: return the field modules at the given slice positions.
            - :class:`~FieldTags`: return field modules matching that single
              tag.
            - iterable of :class:`~FieldTags`: return field modules matching
              all given tags.

        Returns
        -------
        :class:`~FieldModule`, :class:`~FieldModuleList`, or None
            A single :class:`~FieldModule` when exactly one field matches,
            ``None`` when no field matches, or a :class:`~FieldModuleList`
            when multiple fields match.

        Examples
        --------
        >>> from freerec.data.tags import USER, ITEM, ID
        >>> User = FieldModule('User', USER, ID)
        >>> Item = FieldModule('Item', ITEM, ID)
        >>> fields = FieldModuleList([User, Item])
        >>> fields[USER, ID] is User
        True
        >>> fields[0] is User
        True
        >>> fields[0:1] is User
        True
        >>> fields[Item, ID] is Item
        True
        >>> fields[1] is Item
        True
        >>> fields['Item'] is Item
        True
        >>> fields[1:] is Item
        True
        >>> len(fields[ID])
        2
        >>> len(fields[:])
        2
        >>> isinstance(fields[ID], FieldTuple)
        True
        """
        if isinstance(index, int):
            return super().__getitem__(index)
        elif isinstance(index, str):
            fields = FieldModuleList(field for field in self if field.name == index)
        elif isinstance(index, slice):
            fields = FieldModuleList(super().__getitem__(index))
        elif isinstance(index, FieldTags):
            fields = self.match(index)
        else:
            fields = self.match(*index)
        if len(fields) == 1:
            return fields[0]
        elif len(fields) == 0:
            return None  # for a safety purpose
        else:
            return fields
