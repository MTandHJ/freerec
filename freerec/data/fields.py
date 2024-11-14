

from typing import Iterable, Tuple, Union, Literal, Iterator, TypeVar

import torch
import numpy as np
import polars as pl
from itertools import chain

from .tags import FieldTags
from .normalizer import NORMALIZERS


__all__ = [
    'Field', 'FieldTuple', 'FieldModule', 'FieldModuleList'
]


T = TypeVar('T')

class Field:
    r"""
    Field determined by a `name` and a collection of tags (FieldTags).

    Parameters:
    -----------
    name: str
    *tags: FieldTags

    Examples:
    ---------
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

    def __init__(self, name: str,  *tags: FieldTags) -> None:
        self.__name = str(name)
        self.__tags = set(tags)
        self.__identifier = (name,) + tuple(sorted(self.__tags, key=lambda tag: tag.value))
        self.__hash_value = hash(self.identifier)
        self.count = None

    @property
    def name(self) -> str:
        return self.__name

    @property
    def tags(self) -> Tuple:
        return self.__identifier[1:]

    @property
    def identifier(self) -> Tuple:
        return self.__identifier

    def fork(self: T, *tags: FieldTags) -> T:
        field = type(self)(self.name, *self.tags, *tags)
        field.count = self.count
        return field

    def to_module(self) -> 'FieldModule':
        field = FieldModule(
            self.name, *self.tags
        )
        field.count = self.count
        return field

    def match(self, *tags: FieldTags) -> bool:
        """True if `self` matches all tags."""
        return self.__tags.issuperset(tags)

    def match_all(self, *tags: FieldTags) -> bool:
        """True if `self` matches all tags."""
        return self.match(*tags)
    
    def match_any(self, *tags: FieldTags) -> bool:
        """True if `self` matches any tags."""
        return any(self.match(tag) for tag in tags)

    def issubfield(self, other: 'Field') -> bool:
        """True if `self` matches all tags of `other`."""
        return self.match(*other.tags)

    def issuperfield(self, other: 'Field') -> bool:
        """True if `other` matches all tags of `self`."""
        return other.match(*self.tags)

    def __hash__(self) -> int:
        return self.__hash_value

    def __eq__(self, other) -> bool:
        return isinstance(other, Field) and hash(self) == hash(other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"

    def __str__(self) -> str:
        return f"{self.name}:{self.count}:{','.join(map(lambda tag: tag.name, self.tags))}"

    def set_normalizer(
        self, 
        dtype: Union[None, str, pl.DataType] = None,
        fill_null_strategy: Literal['forward', 'backward', 'min', 'max', 'zero', 'one'] = 'zero',
        normalizer: Union[None, Literal['standard', 'minmax', 'reindex']] = None,
        **kwargs
    ):
        r"""
        Field processor for casting, normalization and tokenization.

        Parameters:
        -----------
        field: Field
        dtype: Union[None, str, pl.DataType]
            - `None`: no operation
            - `str`: getattr(str, pl)
        fill_null_strategy: the null_strategy used in polars
            - `None`: no operation
        normalizer: Union[str, Callable]
            - `None`: no operation
            - `standardscaler`: applying standard normalization
            - `minmaxscaler`: applying minmax normalization
            - `reindexer`: mapping each to a number
        kwargs:
            other args for normalizer
        """
        self._dtype = getattr(pl, dtype) if isinstance(dtype, str) else dtype
        self._fill_null_strategy = fill_null_strategy

        normalizer = 'counter' if normalizer is None else normalizer
        try:
            self._normalizer = NORMALIZERS[normalizer.upper()](**kwargs)
        except KeyError:
            availables = '; '.join(NORMALIZERS.keys())
            raise KeyError(
                f"Receive an invalid normalizer not existing in: [{availables}]. "
                f"You should register this via `register_normalizer(normalizer, name)` ..."
            )
        except TypeError:
            raise KeyError(
                f"Receive invalid kwargs for {normalizer}: {kwargs}"
            )

    def cast(
        self, data: pl.Series, strict: bool = False
    ) -> pl.Series:
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
        self, 
        data: Union[pl.Series, pl.DataFrame, pl.LazyFrame],
        partial: bool = True
    ) -> pl.Series:
        r"""
        Fit the whole data column and return the processed.

        Parameters:
        -----------
        data: pl.Series or pl.DataFrame or pl.LazyFrame
        partial: bool
            `True`: partially fitting the given data
            `False`: fitting the given data
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
        r"""
        Normalizer the data.

        Parameters:
        -----------
        data: pl.Series or pl.DataFrame or pl.LazyFrame

        Flows:
        ------
        data -> cast ---> normalization ---> data
        """
        if isinstance(data, pl.LazyFrame):
            data = data.collect().to_series()
        elif isinstance(data, pl.DataFrame):
            data = data.to_series()

        data = self.cast(data)
        data = self._normalizer(data)
        return data

    def to_csr(self, data: Iterable) -> torch.Tensor:
        r"""
        Convert batch of data to CSR Tensor.

        Parameters:
        -----------
        data: 2-d array

        Returns:
        --------
        data: torch.Tensor, (B, #Items)

        Examples:
        ---------
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
        assert isinstance(data[0], (list, tuple)), f"Each row of data should be `list'|`tuple' but `{type(data[0])}' received ..."

        crow_indices = np.cumsum([0] + list(map(len, data)), dtype=np.int64)
        col_indices = list(chain(*data))

        values = np.ones_like(col_indices, dtype=np.int64)
        return torch.sparse_csr_tensor(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            size=(len(data), self.count) # B x Num of Items
        )


class FieldModule(Field, torch.nn.Module):
    r"""
    Field module.

    Notes:
    ------
    The 'fixed' hash value of field/fieldmodule may raise some errors.
    For example, 'a = A()' below. 
    The parameters in 'a.field' can be found by 'a.parameters()'
    but not the case for 'a.field2'. This is because 'duplicate' will be removed during collection ('field1' and 'field2' share the same hash value).

    class A(nn.Module):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.field = FieldModule('a').fork()
            self.field2 = FieldModule('a').fork()

            self.field.add_module(
                "embeddings", nn.Embedding(3, 4)
            )
            self.field2.add_module(
                "embeddings", nn.Embedding(1, 4)
            )
    """

    embeddings: torch.nn.Embedding

    def __init__(self, name: str, *tags: FieldTags) -> None:
        torch.nn.Module.__init__(self)
        Field.__init__(self, name, *tags)


class FieldTuple(tuple):
    """A tuple of fields, which support attribute access and filtering by tags."""

    def match(self, *tags: FieldTags) -> 'FieldTuple':
        """Return those fields matching all given tags."""
        return FieldTuple(field for field in self if field.match(*tags))

    def match_all(self, *tags: FieldTags) -> 'FieldTuple':
        """Return those fields matching all given tags."""
        return FieldTuple(field for field in self if field.match_all(*tags))

    def match_any(self, *tags: FieldTags) -> 'FieldTuple':
        """Return those fields matching any given tags."""
        return FieldTuple(field for field in self if field.match_any(*tags))

    def match_not(self, *tags: FieldTags) -> 'FieldTuple':
        """Return those fields not matching all given tags."""
        return FieldTuple(field for field in self if not field.match_all(*tags))

    def copy(self) -> 'FieldTuple':
        r"""
        Return a copy of the FieldTuple.

        Returns:
        --------
        A new FieldTuple with the same fields as this one.
        """
        return FieldTuple(self)
    
    def index(self, *tags) -> int:
        r"""
        Get index by tags.

        Parameters:
        -----------
        *tags: FieldTags

        Returns:
        --------
        Index: int 
            Index of the field accordance with the given tags.
        
        Examples:
        ---------
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
        return super().__iter__()

    def __getitem__(self, index: Union[int, str, slice, FieldTags, Iterable[FieldTags]]) -> Union[Field, 'FieldTuple', None]:
        r"""
        Get fields by index.

        Parameters:
        -----------
        index: Union[int, slice, FieldTags, Iterable[FieldTags]]
            - int: Return the field at position `int`.
            - str: Return the field with a name of `str`.
            - slice: Return the fields at positions of `slice`.
            - FieldTags: Return the fields matching `FieldTags`.
            - Iterable[FieldTags]: Return the fields matching `Iterable[FieldTags]`.

        Returns:
        --------
        Fields: Union[Field, FieldTuple, None]
            - Field: If only one field matches given tags.
            - None: If none of fields matches given tags.
            - FieldTuple: If more than one field match given tags.

        Examples:
        ---------
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
            fields = FieldTuple(
                super().__getitem__(index)
            )
        elif isinstance(index, FieldTags):
            fields =  self.match(index)
        else:
            fields = self.match(*index)
        if len(fields) == 1:
            return fields[0]
        elif len(fields) == 0:
            return None # for a safety purpose
        else:
            return fields


class FieldModuleList(torch.nn.ModuleList):
    r"""A collection of fields."""

    def __init__(self, fields: Iterable[FieldModule]) -> None:
        super().__init__(fields)
        assert all(isinstance(field, FieldModule) for field in self), "'FieldModuleList' receives 'FieldModule' only ..."

    def match(self, *tags: FieldTags) -> 'FieldModuleList':
        """Return those fields matching all given tags."""
        return FieldModuleList(field for field in self if field.match(*tags))

    def match_all(self, *tags: FieldTags) -> 'FieldModuleList':
        """Return those fields matching all given tags."""
        return FieldModuleList(field for field in self if field.match_all(*tags))

    def match_any(self, *tags: FieldTags) -> 'FieldModuleList':
        """Return those fields matching any given tags."""
        return FieldModuleList(field for field in self if field.match_any(*tags))

    def match_not(self, *tags: FieldTags) -> 'FieldModuleList':
        """Return those fields not matching all given tags."""
        return FieldModuleList(field for field in self if not field.match_all(*tags))

    def insert(self, index: int, field: FieldModule) -> None:
        assert isinstance(field, FieldModule), "'FieldModuleList' receives 'FieldModule' only ..."
        return super().insert(index, field)
    
    def append(self, field: FieldModule) -> 'FieldModuleList':
        assert isinstance(field, FieldModule), "'FieldModuleList' receives 'FieldModule' only ..."
        return super().append(field)
    
    def extend(self, fields: Iterable[FieldModule]) -> 'FieldModuleList':
        fields = list(fields)
        assert all(isinstance(field, FieldModule) for field in fields), "'FieldModuleList' receives 'FieldModule' only ..."
        return super().extend(fields)

    def __iter__(self) -> Iterator[FieldModule]:
        return super().__iter__()

    def __getitem__(self, index: Union[int, str, FieldTags, Iterable[FieldTags]]) -> Union[FieldModule, 'FieldModuleList', None]:
        r"""
        Get fields by index.

        Parameters:
        -----------
        index: Union[int, slice, FieldTags, Iterable[FieldTags]]
            - int: Return the field at position `int`.
            - str: Return the field with a name of `str`.
            - slice: Return the fields at positions of `slice`.
            - FieldTags: Return the fields matching `FieldTags`.
            - Iterable[FieldTags]: Return the fields matching `Iterable[FieldTags]`.

        Returns:
        --------
        Fields: Union[Field, FieldTuple, None]
            - FieldModule: If only one field matches given tags.
            - None: If none of fields matches given tags.
            - FieldModuleList: If more than one field match given tags.

        Examples:
        ---------
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
            fields = FieldModuleList(
                super().__getitem__(index)
            )
        elif isinstance(index, FieldTags):
            fields =  self.match(index)
        else:
            fields = self.match(*index)
        if len(fields) == 1:
            return fields[0]
        elif len(fields) == 0:
            return None # for a safety purpose
        else:
            return fields