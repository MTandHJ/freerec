

from typing import Callable, Iterable, Tuple, Union, Dict, List, Any, Optional

import torch, abc
from functools import partial, lru_cache

from .utils import safe_cast
from .preprocessing import Identifier, Indexer, StandardScaler, MinMaxScaler
from .tags import FieldTags, SPARSE, DENSE
from ..utils import warnLogger


__all__ = ['Field', 'BufferField', 'SparseField', 'DenseField', 'FieldTuple', 'FieldList', 'FieldModule']


TRANSFORMATIONS = {
    "none": Identifier,
    'label2index': Indexer,
    'minmax': MinMaxScaler,
    'standard': StandardScaler,
}



class Field(metaclass=abc.ABCMeta):
    """Fielding data by column.
    
    Attributes:
    ---

    name: str
        Name of this field.
    tags: set
        A set of FieldTags.
    dtype: torch.long|torch.float32
    caster: Callable
        Convert elements to specified dtype (int, float or str) with na_value.
    """
    
    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Union[int, str, float],
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple()
    ):
        """
        Parameters:
        ---

        name: str 
            The name of the field.
        na_value: str, int or float
            Fill 'na' with na_value.
        dtype: str|int|float
        transformer: 'none'|'label2index'|'binary'|'minmax'|'standard'
        tags: Union[FieldTags, Iterable[FieldTags]]
            For quick retrieve.

        Examples:
        ---

        >>> from freerec.data.tags import USER, ITEM, ID, TARGET
        >>> User = SparseField('User', None, int, tags=(USER, ID))
        >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
        >>> Target = SparseField('Label', 0, int, transformer='none', tags=TARGET)
        """

        self.__name = name
        self.dtype = dtype
        self.__na_value = na_value
        self.__tags = set()
        if isinstance(tags, FieldTags):
            self.add_tag(tags)
        else:
            self.add_tag(*tags)
        self.data = None

    def add_tag(self, *tags: FieldTags):
        """Add some tags.

        Parameters:
        ---
        *tags: FieldTags

        Examples:
        ---

        >>> from freerec.data.tags import ID
        >>> User = SparseField('User', -1, int)
        >>> User.add_tag(ID)

        """
        for tag in tags:
            if not isinstance(tag, FieldTags):
                warnLogger(f"FieldTags is expected but {type(tags)} received ...")
            self.__tags.add(tag)

    @property
    def tags(self):
        return self.__tags

    def match(self, *tags: FieldTags):
        """If current field matches the given tags, return True.
        
        Parameters:
        ---
        *tags: FieldTags


        Returns:
        ---

        - `True`: If all given tags are matched.
        - 'False': If any tag is not matched.

        Examples:
        ---

        >>> from freerec.data.tags import ID
        >>> User = SparseField('User', -1, int)
        >>> User.add_tag(ID)
        >>> User.match(ID)
        True
        >>> User.match(ID, Feature)
        False
        >>> User.match(Feature)
        False

        """
        return all(tag in self.__tags for tag in tags)

    @property
    def name(self):
        return self.__name

    @property
    def na_value(self):
        return self.__na_value

    @property
    def dtype(self):
        return self.__dtype

    @dtype.setter
    def dtype(self, val):
        if val in (int, str):
            self.__dtype = torch.long
        elif val is float:
            self.__dtype = torch.float32
        else:
            self.__dtype = val

    def __str__(self) -> str:
        tags = ','.join(map(str, self.tags))
        return f"{self.name}: [dtype: {self.dtype}, na_value: {self.na_value}, tags: {tags}]"

    def __len__(self) -> int:
        try:
            return len(self.data)
        except TypeError:
            return 0

    def buffer( # copy for multiprocessing
        self, data: Any = None,
        dtype: Union[None, int, str, float] = None,
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple()
    ):
        return BufferField(self, data, dtype, tags)


class BufferField(Field):
    """For buffering data, which should be re-created once the data changes.
    
    Attributes:
    ---

    name: str
        Name of this field.
    tags: set
        A set of FieldTags.
    dtype: torch.long|torch.float32
    root: TopField
        the root field for `.look_up` embeddings
    """

    def __init__(
        self, parent: Field, data: Any = None,
        dtype: Union[None, int, str, float] = None, 
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple()
    ):
        super().__init__(
            name = parent.name, 
            na_value = parent.na_value, 
            dtype = dtype if dtype else parent.dtype,
            tags = parent.tags
        )
        if isinstance(tags, FieldTags):
            self.add_tag(tags)
        else:
            self.add_tag(*tags)
        self.root = parent
        self.data = data if data is not None else parent.data

    @property
    def root(self):
        return self.__root

    @root.setter
    def root(self, field: Field):
        if isinstance(field, TopField):
            self.__root = field
        else:
            self.__root = field.root

    def __iter__(self):
        yield from iter(self.data)

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)

    def look_up(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Look up embeddings from indices `x` if it is not None.
        Otherwise, self-look-up will be done.
        """
        return self.__root.look_up(x if x else self.data)


class TopField(Field, torch.nn.Module):

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable,
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple(),
        transformer: Union[str, Callable] = 'none'
    ):
        """
        Parameters:
        ---

        name: str 
            The name of the field.
        na_value: str, int or float
            Fill 'na' with na_value.
        dtype: str|int|float for safe_cast
        transformer: 'none'|'label2index'|'minmax'|'standard'
        tags: Union[FieldTags, Iterable[FieldTags]]
            For quick retrieve.

        Examples:
        ---

        >>> from freerec.data.tags import USER, ITEM, ID, TARGET
        >>> User = SparseField('User', None, int, tags=(USER, ID))
        >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
        >>> Target = SparseField('Label', 0, int, transformer='none', tags=TARGET)
        """

        self.transformer = TRANSFORMATIONS[transformer]() if isinstance(transformer, str) else transformer
        self.caster = partial(safe_cast, dest_type=dtype, default=na_value)
        super().__init__(
            name=name,
            na_value=na_value,
            dtype=dtype,
            tags=tags
        )

    def partial_fit(self, col):
        return self.transformer.partial_fit(col)
    
    def transform(self, col):
        return self.transformer.transform(col)

    def embed(self, dim: int, **kwargs):
        NotImplementedError("'embed' method should be implemented ...")

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        NotImplementedError("'look_up' method should be implemented ...")



class SparseField(TopField):

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable, 
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple(),
        transformer: Union[str, Callable] = 'label2index'
    ):
        assert transformer == 'label2index', "SparseField supports 'Label2Index' only !"
        super().__init__(name, na_value, dtype, tags, transformer)
        self.add_tag(SPARSE)

    @property
    def count(self) -> Optional[int]:
        """Return the number of classes."""
        return self.transformer.count

    @property
    def ids(self) -> Optional[Tuple[int]]:
        """Return the list of IDs."""
        return self.transformer.ids

    def embed(self, dim: int, **kwargs):
        """Create nn.Embedding.

        Parameters:
        ---

        dim: int
            Dimension.
        **kwargs: other kwargs for nn.Embedding

        """

        self.dimension = dim
        self.embeddings = torch.nn.Embedding(self.count, dim, **kwargs)

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for categorical features.

        Parameters:
        ---

        x: (B, *), torch.Tensor

        Returns:
        ---

        embeddings: (B, *, d)

        Examples:
        ---

        >>> User: SparseField
        >>> ids = torch.arange(3).view(-1, 1)
        >>> User.look_up(ids).ndim
        3
        """
        return self.embeddings(x)


class DenseField(TopField):

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable, 
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple(),
        transformer: Union[str, Callable] = 'none'
    ):
        super().__init__(name, na_value, dtype, tags, transformer)
        self.add_tag(DENSE)


    def embed(self, dim: int = 1, bias: bool = False, linear: bool = False):
        """Create Embedding in nn.Linear manner.

        Parameters:
        ---

        dim: int
            Dimension.
        bias: bool
            Add bias or not.
        linear: bool
            - `True`: nn.Linear.
            - `False`: using nn.Identity instead.

        """
        if linear:
            self.dimension = dim
            self.embeddings = torch.nn.Linear(1, dim, bias=bias)
        else:
            self.dimension = 1
            self.embeddings = torch.nn.Identity()

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for numeric features.
        Note that the return embeddings' shape is in accordance with
        that of categorical features.

        Parameters:
        ---

        x: (B, *), torch.Tensor

        Returns:
        ---

        embeddings: (B, *, 1) or (B, *, d)
            - If `linear` is True, it returns (B, *, d).
            - If `linear` is False, it returns (B, *, 1).

        >>> Field: DenseField
        >>> vals = torch.rand(3, 1)
        >>> Field.look_up(vals).ndim
        3
        """
        return self.embeddings(x.unsqueeze(-1))
   

class FieldTuple(tuple):
    """A tuple of fields."""

    @lru_cache(maxsize=4)
    def groupby(self, *tags: FieldTags) -> 'FieldTuple':
        """Return those fields matching given tags."""
        return FieldTuple(field for field in self if field.match(*tags))

    @lru_cache(maxsize=4)
    def groupbynot(self, *tags: FieldTags) -> 'FieldTuple':
        """Return those fields not matching given tags."""
        return FieldTuple(field for field in self if not field.match(*tags))

    def state_dict(self) -> Dict:
        """Return state dict of fields."""
        return {field.name: field.transformer for field in self}

    def load_state_dict(self, state_dict: Dict, strict: bool = False):
        for field in self:
            field.transformer = state_dict.get(field.name, field.transformer)

    def copy(self) -> 'FieldTuple':
        return FieldTuple(self)

    def __getitem__(self, index: Union[int, FieldTags, Iterable[FieldTags]]) -> Union[Field, 'FieldTuple', None]:
        """Get fields by index.

        Parameters:
        ---

        index: Union[int, FieldTags, Iterable[FieldTags]]
            - `int`: Return the field at position `int`.
            - `FieldTags`: Return the fields matching `FieldTags`.
            - `Iterable[FieldTags]`: Return the fields matching `Iterable[FieldTags]`.
        
        Returns:
        ---

        - `Field`: If only one field matches given tags.
        - `None`: If none of fields matches given tags.
        - `FieldTuple`: If more than one field match given tags.

        Examples:
        ---

        >>> from freerec.data.tags import USER, ITEM, ID
        >>> User = SparseField('User', None, int, tags=(USER, ID))
        >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
        >>> fields = FieldTuple([User, Item])
        >>> fields[USER, ID] is User
        True
        >>> fields[Item, ID] is Item
        True
        >>> len(fields[ID])
        2
        >>> isinstance(fields[ID], FieldTuple)
        True
        """
        if isinstance(index, int):
            return super().__getitem__(index)
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


class FieldList(list):
    """A list of (buffer)fields."""

    @property
    def datasize(self):
        return len(self[0])

    def groupby(self, *tags: FieldTags) -> 'FieldList':
        """Return those fields matching given tags."""
        return FieldList(field for field in self if field.match(*tags))

    def groupbynot(self, *tags: FieldTags) -> 'FieldList':
        """Return those fields not matching given tags."""
        return FieldList(field for field in self if not field.match(*tags))

    def copy(self) -> 'FieldList':
        return FieldList(self)

    def __getitem__(self, index: Union[int, FieldTags, Iterable[FieldTags]]) -> Union[Field, 'FieldList', None]:
        """Get fields by index.

        Parameters:
        ---

        index: Union[int, FieldTags, Iterable[FieldTags]]
            - `int`: Return the field at position `int`.
            - `FieldTags`: Return the fields matching `FieldTags`.
            - `Iterable[FieldTags]`: Return the fields matching `Iterable[FieldTags]`.
        
        Returns:
        ---

        - `Field`: If only one field matches given tags.
        - `None`: If none of fields matches given tags.
        - `FieldTuple`: If more than one fields match given tags.

        Examples:
        ---

        >>> from freerec.data.tags import USER, ITEM, ID
        >>> User = SparseField('User', None, int, tags=(USER, ID))
        >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
        >>> fields = FieldList([User, Item])
        >>> fields[USER, ID] is User
        True
        >>> fields[Item, ID] is Item
        True
        >>> len(fields[ID])
        2
        >>> isinstance(fields[ID], FieldList)
        True
        """

        if isinstance(index, int):
            return super().__getitem__(index)
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

    def look_up(self):
        """Self-Look-Up for a collection of embeddings.
        """
        return list(map(
            lambda field: field.look_up(),
            self
        ))


class FieldModule(torch.nn.Module):
    """A collection of fields.
    
    Attributes:
    ---

    fields: nn.ModuleList

    """

    def __init__(self, fields: Iterable[TopField]) -> None:
        """
        Examples:
        ---

        >>> from freerec.data.datasets import Gowalla_m1
        >>> basepipe = Gowalla_m1("../data")
        >>> fields = basepipe.fields
        >>> fields = FieldModule(fields)
        """
        super().__init__()

        self.fields = torch.nn.ModuleList(fields)

    @lru_cache(maxsize=4)
    def groupby(self, *tags: FieldTags) -> List[TopField]:
        """Group by given tags and return list of fields matched.

        Examples:
        ---
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
        return [field for field in self.fields if field.match(*tags)]

    def embed(self, dim: int, *tags: FieldTags, **kwargs):
        """Create embeddings.

        Parameters:
        ---
        dim: int
            Dimension.
        *tags: FieldTags
        **kwargs: kwargs for nn.Embeddings or nn.Linear

        Examples:
        ---

        >>> from freerec.data.tags import DENSE, SPARSE
        >>> fields.embed(8, SPARSE) # nn.Embedding(count, 8)
        >>> fields.embed(8, DENSE, bias=False) # nn.Linear(1, 8, bias=False)
        """
        for field in self.groupby(*tags):
            field.embed(dim, **kwargs)

    def calculate_dimension(self, *tags: FieldTags):
        """Return the total dimension of fields matching tiven tags."""
        return sum(field.dimension for field in self.groupby(*tags))

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, index: Union[int, FieldTags, Iterable[FieldTags]]) -> Union[TopField, 'FieldModule', None]:
        """Get fields by index.

        Parameters:
        ---

        index: Union[int, FieldTags, Iterable[FieldTags]]
            - `int`: Return the fields at position `int`.
            - `FieldTags`: Return the fields matching `FieldTags`.
            - `Iterable[FieldTags]`: Return the fields matching `Iterable[FieldTags]`.
        
        Returns:
        ---

        - `TopField`: If only one field matches given tags.
        - `None`: If none of fields matches given tags.
        - `FieldModule`: If more than one fields match given tags.

        Examples:
        ---

        >>> from freerec.data.tags import USER
        >>> fields[0];
        >>> user = fields[USER]
        >>> isinstance(user, TopField)
        True
        >>> isinstance(fields[ID], FieldModule)
        True
        """
        if isinstance(index, int):
            fields = [self.fields[index]]
        elif isinstance(index, FieldTags):
            fields =  self.groupby(index)
        else:
            fields = self.groupby(*index)
        if len(fields) == 1:
            return fields[0]
        elif len(fields) == 0:
            return None # for a safety purpose
        else:
            return FieldModule(fields)
