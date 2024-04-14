

from typing import Iterable, Tuple, Union

import torch
import pandas as pd
from functools import lru_cache

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
    >>> UserID == User
    False
    >>> UserID >= User
    False
    >>> UserID <= User
    True
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
    True
    >>> hash(UserID1) == hash(UserID2)
    False
    """

    def __init__(self, name: str,  *tags: FieldTags) -> None:
        self.__name = name
        self.__tags = set(tags)
        self.__identifier = (name,) + tuple(sorted(self.__tags, key=lambda tag: tag.value))
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

    def try_to_numeric(self, data: Iterable) -> Tuple:
        data = pd.Series(data)
        if self.match(FieldTags.ID):
            data = data.astype(int)
        elif self.match(FieldTags.RATING) or self.match(FieldTags.TIMESTAMP):
            data = data.astype(float)
        else:
            try:
                data = pd.to_numeric(data, errors='raise')
            except ValueError:
                pass
        data = tuple(data.to_list())
        self._enums |= set(data)
        self.count = len(self._enums)
        return data

    def fork(self, *tags: FieldTags) -> 'Field':
        field = Field(self.name, *self.tags, *tags)
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

    def __eq__(self, other) -> bool:
        return isinstance(other, Field) and self.tags == other.tags

    def __le__(self, other: 'Field') -> bool:
        return self.issubfield(other)

    def __ge__(self, other: 'Field') -> bool:
        return self.issuperfield(other)

    def __hash__(self) -> int:
        return hash(self.identifier)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"

    def __str__(self) -> str:
        return f"{self.name}:{','.join(map(lambda tag: tag.name, self.tags))}"


class FieldModule(Field, torch.nn.Module):

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
    r"""A collection of fields.
    
    Attributes:
        fields (nn.ModuleList): A list of fields.

    Examples:
        >>> from freerec.data.datasets import Gowalla_m1
        >>> basepipe = Gowalla_m1("../data")
        >>> fields = basepipe.fields
        >>> fields = FieldModule(fields)
    """

    def __init__(self, fields: Iterable[FieldModule]) -> None:
        super().__init__(fields)

    @lru_cache(maxsize=4)
    def groupby(self, *tags: FieldTags) -> FieldTuple[FieldModule]:
        r"""
        Return those fields matching given tags.

        Parameters:
        -----------
        *tags: FieldTags 
            Variable length argument list of FieldTags to filter the fields by.

        Returns:
        --------
        A new FieldTuple that contains only the fields whose tags match the given tags.

        Examples:
        ---------
        >>> from freerec.data.tags import USER, ID
        >>> User = fields.groupby(USER, ID)
        >>> isinstance(User, List)
        True
        >>> Item = fields.groupby(ITEM, ID)[0]
        >>> Item.match(ITEM)
        True
        >>> Item.match(ID)
        True
        >>> Item.match(User)
        False
        """
        return FieldTuple(field for field in self.fields if field.match(*tags))

    @lru_cache(maxsize=4)
    def groupbynot(self, *tags: FieldTags) -> FieldTuple:
        r"""
        Return those fields not matching given tags.

        Parameters:
        -----------
        *tags: FieldTags 
            Variable length argument list of FieldTags to filter the fields by.

        Returns:
        --------
            A new FieldTuple that contains only the fields whose tags do not match the given tags.
        """
        return FieldTuple(field for field in self.fields if not field.match(*tags))

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, index: Union[int, str, FieldTags, Iterable[FieldTags]]) -> Union[FieldModule, FieldTuple, None]:
        r"""
        Get fields by index.

        Parameters:
        -----------
        index: Union[int, FieldTags, Iterable[FieldTags]]
            - int: Return the field at position `int`.
            - str: Return the field with a name of `str`.
            - slice: Return the fields at positions of `slice`.
            - FieldTags: Return the fields matching `FieldTags`.
            - Iterable[FieldTags]: Return the fields matching `Iterable[FieldTags]`.

        Returns:
        --------
        Fields: Union[FieldModule, FieldList, None]
            - Field: If only one field matches given tags.
            - None: If none of fields matches given tags.
            - FieldTuple: If more than one field match given tags.

        Examples:
        ---------
        >>> from freerec.data.tags import USER, ITEM, ID
        >>> User = SparseField('User', None, int, tags=(USER, ID))
        >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
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
        >>> isinstance(fields[ID], FieldList)
        True
        """

        if isinstance(index, int):
            return super().__getitem__(index)
        elif isinstance(index, slice):
            fields = FieldTuple(
                super().__getitem__(index)
            )
        elif isinstance(index, str):
            fields = FieldTuple(field for field in self if field.name == index)
        elif isinstance(index, FieldTags):
            fields =  self.groupby(index)
        else:
            fields = self.groupby(*index)
        if len(fields) == 1:
            return fields[0]
        elif len(fields) == 0:
            return None # for a safety purpose
        else:
            return fields