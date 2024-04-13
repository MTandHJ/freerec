

from typing import Callable, Iterable, Tuple, Union, Dict, Optional, Any

import torch, abc
import numpy as np
import pandas as pd
from functools import partial, lru_cache, reduce
from itertools import chain

from .utils import safe_cast
from .transformation import Identifier, Indexer, MaxIndexer
from .tags import FieldTags
from ..utils import warnLogger


__all__ = [
    'Field', 
    'BufferField', 'SparseField', 'DenseField', 'AffiliateField', 
    'FieldTuple', 'FieldList', 'FieldModule'
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

    def fit(self, data: Iterable):
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
        self.count = len(data.unique())
        return tuple(data.to_list())

    def fork(self, *tags):
        field = Field(self.name, *self.tags, *tags)
        field.count = self.count
        return field

    def match(self, *tags):
        return self.__tags.issuperset(tags)

    def issubfield(self, other: 'Field'):
        return self.match(*other.tags)

    def issuperfield(self, other: 'Field'):
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


# class BufferField(Field):
#     r"""
#     For buffering data, which should be re-created once the data changes.
    
#     Parameters:
#     -----------
#     data: Any
#         Any column data.
#     tags: Union[FieldTags, Iterable[FieldTags]] 
#         Tags for filtering.
#     root: Field, optional
#         Inherit some attribuites from the root.
#     """

#     def __init__(
#         self, data: Any, 
#         tags: Union[FieldTags, Iterable[FieldTags]] = tuple(),
#         *, root: Optional[Field] = None
#     ) -> None:
#         super().__init__(data, tags)
#         self.data = data
#         if isinstance(tags, FieldTags):
#             self.add_tag(tags)
#         else:
#             self.add_tag(*tags)
#         self.inherit(root)

#     def inherit(self, root: Union[None, 'BufferField', 'FieldModule']):
#         if root is None:
#             pass
#         elif root.match(SPARSE):
#             self.count = root.count
#             self.add_tag(*root.tags)
#         elif root.match(DENSE):
#             self.add_tag(*root.tags)
#         elif isinstance(root, Field):
#             self.add_tag(*root.tags)
#         else:
#             raise ValueError(
#                 f"root should be `None|BufferField|FieldMoudle' but {type(root)} received ..."
#             )

#     def __getitem__(self, *args, **kwargs):
#         return self.data.__getitem__(*args, **kwargs)

#     def __iter__(self):
#         yield from iter(self.data)

#     def to(
#         self, device: Optional[Union[int, torch.device]] = None,
#         dtype: Optional[Union[torch.dtype, str]] = None,
#         non_blocking: bool = False
#     ):
#         if isinstance(self.data, torch.Tensor):
#             self.data = self.data.to(device, dtype, non_blocking)
#         return self

#     def to_csr(self, length: Optional[int] = None) -> torch.Tensor:
#         r"""
#         Convert List to CSR Tensor.

#         Notes:
#         ------
#         Each row in self.data should be the col indices !
#         """
#         data = self.data
#         if isinstance(data, torch.Tensor):
#             data = data.tolist()
#         elif isinstance(data, np.ndarray):
#             data = data.tolist()
#         assert isinstance(data[0], (list, tuple)), f"Each row of data should be `list'|`tuple' but `{type(data[0])}' received ..."

#         length = self.count if length is None else length
#         crow_indices = np.cumsum([0] + list(map(len, self.data)), dtype=np.int64)
#         col_indices = list(chain(
#             *self.data
#         ))
#         values = np.ones_like(col_indices, dtype=np.int64)
#         return torch.sparse_csr_tensor(
#             crow_indices=crow_indices,
#             col_indices=col_indices,
#             values=values,
#             size=(len(data), length) # B x Num of Items
#         )


class FieldModule(Field, torch.nn.Module):

    def __init__(self, name: str, *tags: FieldTags) -> None:
        torch.nn.Module.__init__(self)
        Field.__init__(self, name, *tags)

        self.set_caster()
        self.set_transformer()

        self.dimension: int
        self.embeddings: Union[torch.nn.Embedding, torch.nn.Linear]

    def set_caster(self):
        if FieldTags.INT in self:
            self.caster = partial(safe_cast, dest_type=int, default=None)
        elif FieldTags.FLOAT in self:
            self.caster = partial(safe_cast, dest_type=float, default=None)
        else:
            self.caster = partial(safe_cast, dest_type=str, default=None)

    def set_transformer(self):
        if FieldTags.TOKEN in self:
            self.transformer = Indexer()
        elif FieldTags.INT in self:
            self.transformer = MaxIndexer()
        else:
            self.transformer = Identifier()

    @property
    def count(self) -> Optional[int]:
        """Return the number of classes."""
        return self.transformer.count

    def partial_fit(self, col) -> None:
        r"""
        Updates the transformer with a new batch of data.

        Parameters:
        -----------
        col: Iterable 
            The data to fit the transformer with.

        Returns:
        --------
        None
        """
        return self.transformer.partial_fit(col)

    def transform(self, col):
        r"""
        Applies the transformer to a new batch of data.

        Parameters:
        -----------
        col: Iterable
            The data to transform.

        Returns:
        --------
        The processed data.
        """
        return self.transformer.transform(col)


class FieldTuple(tuple):
    """A tuple of fields, which support attribute access and filtering by tags."""

    @lru_cache(maxsize=4)
    def groupby(self, *tags: FieldTags) -> 'FieldTuple':
        r"""
        Return those fields matching given tags.

        Parameters:
        -----------
        *tags: FieldTags 
            Variable length argument list of FieldTags to filter the fields by.

        Returns:
        --------
        A new FieldTuple that contains only the fields whose tags match the given tags.
        """
        return FieldTuple(field for field in self if field.match(*tags))

    @lru_cache(maxsize=4)
    def groupbynot(self, *tags: FieldTags) -> 'FieldTuple':
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
        return FieldTuple(field for field in self if not field.match(*tags))

    def state_dict(self) -> Dict:
        r"""
        Return state dict of fields.

        Returns:
        --------
            A dictionary containing the name and transformer of each field.
        """
        return {field.name: field.transformer for field in self}

    def load_state_dict(self, state_dict: Dict, strict: bool = False):
        r"""
        Load state dict of fields.

        Parameters:
        -----------
        state_dict: Dict 
            A dictionary containing the state of the fields.
        strict: bool 
            Whether to strictly enforce that the keys in the state dict match the names of the fields.
        """
        for field in self:
            field.transformer = state_dict.get(field.name, field.transformer)

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
        >>> User = SparseField('User', None, int, tags=(USER, ID))
        >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
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
        >>> isinstance(fields[ID], FieldList)
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
            fields =  self.groupby(index)
        else:
            fields = self.groupby(*index)
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