

from typing import Iterable, Tuple, Union, Literal

import torch
import numpy as np
import polars as pl
from itertools import chain

from .tags import FieldTags


__all__ = [
    'Field', 'FieldTuple', 'FieldModule'
]


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

    def cast(
        self, 
        data: pl.LazyFrame, 
        dtype: Union[None, int, float, str, pl.DataType] = None,
        strict: bool = False,
        fill_null_strategy: Literal['forward', 'backward', 'min', 'max', 'zero', 'one'] = 'zero'
    ) -> pl.Series:
        if self.match(FieldTags.ID):
            dtype, strict = int, True
        elif self.match(FieldTags.RATING) or self.match(FieldTags.TIMESTAMP):
            dtype = float

        data = data.collect().to_series()
        if dtype is not None:
            data = data.cast(dtype, strict=strict)
        try:
            data = data.fill_nan(None)
        except Exception:
            # Skip `fill_nan` for String data
            pass
        finally:
            data = data.fill_null(strategy=fill_null_strategy)

        return data

    def standard_scale(self, data: pl.Series, eps: float = 1.e-8) -> pl.Series:
        return (data - data.mean()) / (data.std() + eps)

    def minmax_scale(self, data: pl.Series, eps: float = 1.e-8) -> pl.Series:
        return (data - data.min()) / (data.max() - data.min() + eps)

    def fork(self, *tags: FieldTags) -> 'Field':
        field = Field(self.name, *self.tags, *tags)
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

    def __iter__(self):
        return iter(self.tags)

    def __hash__(self) -> int:
        return self.__hash_value

    def __eq__(self, other) -> bool:
        return isinstance(other, Field) and hash(self) == hash(other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"

    def __str__(self) -> str:
        return f"{self.name}:{','.join(map(lambda tag: tag.name, self.tags))}"

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