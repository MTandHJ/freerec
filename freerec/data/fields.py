

from typing import Callable, Iterable, Tuple, Union, Dict, Optional, Any

import torch, abc
import numpy as np
from functools import partial, lru_cache, reduce

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
    """ Fielding data by column.

    Attributes:
        name (None, str): Name of this field.
        tags (set): A set of FieldTags.
        dtype (Any): Data type of the field.
        caster (callable): Function to convert elements to specified dtype (int, float or str) with na_value.
        na_value (Union[None, str, int, float]): The fill value for null/missing values.

    Examples:
        >>> from freerec.data.tags import USER, ITEM, ID, TARGET
        >>> User = SparseField('User', None, int, tags=(USER, ID))
        >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
        >>> Target = SparseField('Label', 0, int, transformer='none', tags=TARGET)
    """

    def __init__(
        self, data: Any = None, 
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple()
    ) -> None:
        self.__tags = set()
        if isinstance(tags, FieldTags):
            self.add_tag(tags)
        else:
            self.add_tag(*tags)
        self.data = data

    def add_tag(self, *tags: FieldTags) -> None:
        """Add some tags.

        Args:
            *tags (FieldTags): The tags to add.

        Examples:
            >>> from freerec.data.tags import ID
            >>> User = SparseField('User', -1, int)
            >>> User.add_tag(ID)
        """
        for tag in tags:
            if not isinstance(tag, FieldTags):
                warnLogger(f"FieldTags is expected but {type(tags)} received ...")
            self.__tags.add(tag)

    @property
    def tags(self) -> set:
        return self.__tags

    def match(self, *tags: FieldTags):
        """If current field matches the given tags, return True.

        Args:
            *tags (FieldTags): The tags to match.

        Returns:
            bool: True if all given tags are matched, False otherwise.

        Examples:
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

    def buffer(
        self, data: Any = None,
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple()
    ) -> 'BufferField':
        """Return a new BufferField with the given or inherited data.

        Notes: The tags will also be inherited.
        """
        buffer_ = BufferField(data, tags)
        buffer_.add_tag(*self.tags)
        return buffer_


class BufferField(Field):
    """For buffering data, which should be re-created once the data changes.
    
    Args:
        data (Any): Any column data.
        tags (Union[FieldTags, Iterable[FieldTags]]): Tags for filtering.
    """

    def __init__(
        self, data: Any, 
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple()
    ) -> None:
        super().__init__(data, tags)
        self.data = data
        if isinstance(tags, FieldTags):
            self.add_tag(tags)
        else:
            self.add_tag(*tags)

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)

    def __iter__(self):
        yield from iter(self.data)

    def to(
        self, device: Optional[Union[int, torch.device]] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        non_blocking: bool = False
    ):
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.to(device, dtype, non_blocking)
        return self

    def to_csr(self) -> torch.Tensor:
        """Convert List to CSR Tensor.

        Notes:
            Each row in self.data should be the col indices !
        """
        crow_indices = np.cumsum([0] + list(map(len, self.data)), dtype=np.int64)
        col_indices = reduce(
            lambda x, y: x + y, self.data
        )
        values = np.ones_like(col_indices, dtype=np.int64)
        return torch.sparse_csr_tensor(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            size=(len(self.data), self.root.count) # B x Num of Items
        )



class FieldModule(Field, torch.nn.Module):
    """ A module that represents a field of data.

    Args:
        name (str): The name of the field.
        tags (set): A set of FieldTags.
        dtype (Callable): A function to convert elements to specified dtype (int, float or str) with na_value.
        na_value (Union[str, int, float]): The fill value for null/missing values.
        transformer (Transformer): A transformer to apply on the column data.

    Attributes:
        name (str): Name of this field.
        tags (set): A set of FieldTags.
        dtype (torch.dtype): Data type of the field.
        caster (callable): Function to convert elements to specified dtype (int, float or str) with na_value.
        na_value (Union[str, int, float]): The fill value for null/missing values.
        dimension (int): The dimension of embeddings
        embeddings (torch.nn.Module): Embedding module.

    Examples:
        >>> from freerec.data.tags import USER, ITEM, ID, TARGET
        >>> User = SparseField('User', None, int, tags=(USER, ID))
        >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
        >>> Target = SparseField('Label', 0, int, transformer='none', tags=TARGET)
    """

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable,
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple(),
        transformer: Union[str, Callable] = 'none'
    ) -> None:
        torch.nn.Module.__init__(self)
        Field.__init__(self, tags=tags)

        self.__name = name
        self.__na_value = na_value
        self.dtype = dtype

        self.transformer = TRANSFORMATIONS[transformer]() if isinstance(transformer, str) else transformer
        self.caster = partial(safe_cast, dest_type=dtype, default=na_value)

    @property
    def name(self) -> str:
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

    def partial_fit(self, col) -> None:
        """
        Updates the transformer with a new batch of data.

        Args:
            col (Iterable): The data to fit the transformer with.

        Returns:
            None
        """
        return self.transformer.partial_fit(col)

    def transform(self, col):
        """
        Applies the transformer to a new batch of data.

        Args:
            col (Iterable): The data to transform.

        Returns:
            The processed data.
        """
        return self.transformer.transform(col)

    def embed(self, dim: int, **kwargs):
        """Embed the field values into a lower dimensional space.

        Args:
            dim (int): The dimension of the lower dimensional space.
            **kwargs: Other arguments for embedding.

        Raises:
            NotImplementedError: if the method is not implemented.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.embed() method should be implemented ...")

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        """Look up embeddings.

        Args:
            x (torch.Tensor): Indices for looking up.

        Raises:
            NotImplementedError: if the method is not implemented.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.look_up() method should be implemented...")

    def __str__(self) -> str:
        tags = ','.join(map(str, self.tags))
        return f"{self.name}: [dtype: {self.dtype}, na_value: {self.na_value}, tags: {tags}]"


class SparseField(FieldModule):
    """ SparseField inherits from FieldModule and represents sparse features. 
    It is used to look up embeddings for categorical features.

    Args:
        name (str): The name of the field.
        na_value (Union[str, int, float]): Fill 'na' with na_value.
        dtype (Callable): Function to cast the data.
        tags (Union[FieldTags, Iterable[FieldTags]]): Optional. For quick retrieve.
        transformer (Union[str, Callable]): Optional. Transform the input data. Default is 'label2index'.

    Attributes:
        name (str): Name of this field.
        tags (set): A set of FieldTags.
        dtype (torch.dtype): Data type of the field.
        caster (callable): Function to convert elements to specified dtype (int, float or str) with na_value.
        na_value (Union[str, int, float]): The fill value for null/missing values.
        dimension (int): The dimension of embeddings
        embeddings (torch.nn.Module): Embedding module.

    Tags:
        SPARSE
    """

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable, 
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple(),
        transformer: Union[str, Callable] = 'label2index'
    ):
        assert transformer == 'label2index', "SparseField supports 'label2index' only !"
        super().__init__(name, na_value, dtype, tags, transformer)
        self.add_tag(SPARSE)

    @property
    def count(self) -> Optional[int]:
        """Return the number of classes."""
        return self.transformer.count

    @property
    def ids(self) -> Optional[Tuple[int]]:
        """Return the tuple of IDs."""
        return self.transformer.ids

    def embed(self, dim: int, **kwargs) -> None:
        """Create an nn.Embedding layer for the sparse field.

        Args:
            dim (int): The embedding dimension.
            **kwargs: Other keyword arguments to be passed to nn.Embedding.

        Returns:
            None

        """

        self.dimension = dim
        self.embeddings = torch.nn.Embedding(self.count, dim, **kwargs)

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for categorical features.

        Args:
            x (torch.Tensor): The input tensor of shape (B, *), where B is the batch size
                and * can be any number of additional dimensions.

        Returns:
            torch.Tensor: The output tensor of shape (B, *, d), where d is the embedding dimension.

        Examples:
            >>> User: SparseField
            >>> ids = torch.arange(3).view(-1, 1)
            >>> User.look_up(ids).ndim
            3
        """
        return self.embeddings(x)


class DenseField(FieldModule):
    """DenseField is used for numerical features.

    Args:
        name (str): Feature name.
        na_value (Union[str, int, float]): Value to represent null value.
        dtype (Callable): Data type converter.
        tags (Union[FieldTags, Iterable[FieldTags]], optional): Field tags. Defaults to tuple().
        transformer (Union[str, Callable], optional): Transformer to convert data. Defaults to 'none'.

    Attributes:
        name (str): Name of this field.
        tags (set): A set of FieldTags.
        dtype (torch.dtype): Data type of the field.
        caster (callable): Function to convert elements to specified dtype (int, float or str) with na_value.
        na_value (Union[str, int, float]): The fill value for null/missing values.
        dimension (int): The dimension of embeddings
        embeddings (torch.nn.Module): Embedding module.

    Tags:
        DENSE

    """

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable, 
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple(),
        transformer: Union[str, Callable] = 'none'
    ):
        super().__init__(name, na_value, dtype, tags, transformer)
        self.add_tag(DENSE)

    def embed(self, dim: int = 1, bias: bool = False, linear: bool = False) -> None:
        """Create Embedding in nn.Linear manner.

        Args:
            dim (int): Dimension.
            bias (bool): Add bias or not.
            linear (bool): 
                - `True`: nn.Linear.
                - `False`: using nn.Identity instead.
            **kwargs: other kwargs for nn.Linear.

        Returns:
            None.
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

        Args:
            x: (B, *), torch.Tensor

        Returns:
            embeddings: (B, *, 1) or (B, *, d)
                - If `linear` is True, it returns (B, *, d).
                - If `linear` is False, it returns (B, *, 1).

        Examples:
            >>> Field: DenseField
            >>> vals = torch.rand(3, 1)
            >>> Field.look_up(vals).ndim
            3
        """
        return self.embeddings(x.unsqueeze(-1))
   

class FieldTuple(tuple):
    """A tuple of fields.

    A tuple of fields, which support attribute access and filtering by tags.

    """

    @lru_cache(maxsize=4)
    def groupby(self, *tags: FieldTags) -> 'FieldTuple':
        """Return those fields matching given tags.

        Args:
            *tags (FieldTags): Variable length argument list of FieldTags to filter the fields by.

        Returns:
            A new FieldTuple that contains only the fields whose tags match the given tags.

        """
        return FieldTuple(field for field in self if field.match(*tags))

    @lru_cache(maxsize=4)
    def groupbynot(self, *tags: FieldTags) -> 'FieldTuple':
        """Return those fields not matching given tags.

        Args:
            *tags (FieldTags): Variable length argument list of FieldTags to filter the fields by.

        Returns:
            A new FieldTuple that contains only the fields whose tags do not match the given tags.

        """
        return FieldTuple(field for field in self if not field.match(*tags))

    def state_dict(self) -> Dict:
        """Return state dict of fields.

        Returns:
            A dictionary containing the name and transformer of each field.

        """
        return {field.name: field.transformer for field in self}

    def load_state_dict(self, state_dict: Dict, strict: bool = False):
        """Load state dict of fields.

        Args:
            state_dict (Dict): A dictionary containing the state of the fields.
            strict (bool): Whether to strictly enforce that the keys in the state dict match the names of the fields.

        """
        for field in self:
            field.transformer = state_dict.get(field.name, field.transformer)

    def copy(self) -> 'FieldTuple':
        """Return a copy of the FieldTuple.

        Returns:
            A new FieldTuple with the same fields as this one.

        """
        return FieldTuple(self)

    def __getitem__(self, index: Union[int, slice, FieldTags, Iterable[FieldTags]]) -> Union[Field, 'FieldTuple', None]:
        """Get fields by index.

        Args:
            index (Union[int, slice, FieldTags, Iterable[FieldTags]]):
                - int: Return the field at position `int`.
                - slice: Return the fields at positions of `slice`.
                - FieldTags: Return the fields matching `FieldTags`.
                - Iterable[FieldTags]: Return the fields matching `Iterable[FieldTags]`.

        Returns:
            - Field: If only one field matches given tags.
            - None: If none of fields matches given tags.
            - FieldTuple: If more than one field match given tags.

        Examples:
            >>> from freerec.data.tags import USER, ITEM, ID
            >>> User = SparseField('User', None, int, tags=(USER, ID))
            >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
            >>> fields = FieldModuleTuple([User, Item])
            >>> fields[USER, ID] is User
            True
            >>> fields[0] is User
            True
            >>> fields[0:1] is User
            True
            >>> fields[Item, ID] is Item
            True
            >>> fields [1] is Item
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
    """A list of fields.

    A list of fields, which support attribute access and filtering by tags.

    """

    def groupby(self, *tags: FieldTags) -> 'FieldList':
        """Return those fields matching given tags.

        Args:
            *tags (FieldTags): Variable length argument list of FieldTags to filter the fields by.

        Returns:
            A new FieldList that contains only the fields whose tags match the given tags.

        """
        return FieldList(field for field in self if field.match(*tags))

    def groupbynot(self, *tags: FieldTags) -> 'FieldList':
        """Return those fields not matching given tags.

        Args:
            *tags (FieldTags): Variable length argument list of FieldTags to filter the fields by.

        Returns:
            A new FieldList that contains only the fields whose tags do not match the given tags.

        """
        return FieldList(field for field in self if not field.match(*tags))

    def copy(self) -> 'FieldList':
        """Return a copy of the FieldList.

        Returns:
            A new FieldList with the same fields as this one.

        """
        return FieldList(self)

    def __getitem__(self, index: Union[int, slice, FieldTags, Iterable[FieldTags]]) -> Union[Field, 'FieldList', None]:
        """Get fields by index.

        Args:
            index (Union[int, FieldTags, Iterable[FieldTags]]):
                - int: Return the field at position `int`.
                - slice: Return the fields at positions of `slice`.
                - FieldTags: Return the fields matching `FieldTags`.
                - Iterable[FieldTags]: Return the fields matching `Iterable[FieldTags]`.

        Returns:
            - Field: If only one field matches given tags.
            - None: If none of fields matches given tags.
            - FieldList: If more than one field match given tags.

        Examples:
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
            >>> fields [1] is Item
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
            fields = FieldList(
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


class FieldModuleList(torch.nn.Module):
    """A collection of fields.
    
    Attributes:
        fields (nn.ModuleList): A list of fields.

    Examples:
        >>> from freerec.data.datasets import Gowalla_m1
        >>> basepipe = Gowalla_m1("../data")
        >>> fields = basepipe.fields
        >>> fields = FieldModule(fields)
    """

    def __init__(self, fields: Iterable[FieldModule]) -> None:
        super().__init__()

        self.fields = torch.nn.ModuleList(fields)

    @lru_cache(maxsize=4)
    def groupby(self, *tags: FieldTags) -> FieldList[FieldModule]:
        """Return those fields matching given tags.

        Args:
            *tags (FieldTags): Variable length argument list of FieldTags to filter the fields by.

        Returns:
            A new FieldList that contains only the fields whose tags match the given tags.

        Examples:
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
        return FieldList(field for field in self.fields if field.match(*tags))

    @lru_cache(maxsize=4)
    def groupbynot(self, *tags: FieldTags) -> 'FieldList':
        """Return those fields not matching given tags.

        Args:
            *tags (FieldTags): Variable length argument list of FieldTags to filter the fields by.

        Returns:
            A new FieldList that contains only the fields whose tags do not match the given tags.

        """
        return FieldList(field for field in self.fields if not field.match(*tags))

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

    def __getitem__(self, index: Union[int, FieldTags, Iterable[FieldTags]]) -> Union[FieldModule, 'FieldList', None]:
        """Get fields by index.

        Args:
            index (Union[int, FieldTags, Iterable[FieldTags]]):
                - int: Return the field at position `int`.
                - slice: Return the fields at positions of `slice`.
                - FieldTags: Return the fields matching `FieldTags`.
                - Iterable[FieldTags]: Return the fields matching `Iterable[FieldTags]`.

        Returns:
            - Field: If only one field matches given tags.
            - None: If none of fields matches given tags.
            - FieldList: If more than one field match given tags.

        Examples:
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
            >>> fields [1] is Item
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
            fields = FieldList(
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