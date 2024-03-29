

from typing import Callable, Iterable, Tuple, Union, Dict, Optional, Any

import torch, abc
import numpy as np
import pandas as pd
from functools import partial, lru_cache, reduce

from .utils import safe_cast
from .transformation import Identifier, Indexer, StandardScaler, MinMaxScaler
from .tags import FieldTags, SPARSE, DENSE, AFFILIATE, ID
from ..utils import warnLogger


__all__ = [
    'Field', 
    'BufferField', 'SparseField', 'DenseField', 'AffiliateField', 
    'FieldTuple', 'FieldList', 'FieldModule'
]


TRANSFORMATIONS = {
    "none": Identifier,
    'label2index': Indexer,
    'minmax': MinMaxScaler,
    'standard': StandardScaler,
}


class Field(metaclass=abc.ABCMeta):
    r"""
    Fielding data by column.

    Parametesr:
    -----------
    name: str, optional 
        Name of this field.
    tags: set 
        A set of FieldTags.
    dtype: Any 
        Data type of the field.
    caster: Callable 
        Function to convert elements to specified dtype (int, float or str) with na_value.
    na_value: Union[None, str, int, float] 
        The fill value for null/missing values.

    Examples:
    ---------
    >>> from freerec.data.tags import USER, ITEM, ID, TARGET
    >>> User = SparseField('User', None, int, tags=(USER, ID))
    >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
    >>> Target = SparseField('Label', 0, int, transformer='none', tags=TARGET)
    """

    _exclusive_tags = set()

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
        r"""
        Add some tags.

        Parameters:
        -----------
        *tags: FieldTags 
            The tags to add.

        Examples:
        ---------
        >>> from freerec.data.tags import ID
        >>> User = SparseField('User', -1, int)
        >>> User.add_tag(ID)
        """
        for tag in tags:
            if not isinstance(tag, FieldTags):
                warnLogger(f"FieldTags is expected but {type(tags)} received ...")
            if tag not in self._exclusive_tags:
                self.__tags.add(tag)
        assert not self.match(SPARSE, DENSE), "A field should not be `SPARSE` and `DENSE` at the same time ..."

    @property
    def tags(self) -> set:
        return self.__tags

    def match(self, *tags: FieldTags):
        r"""
        If current field matches the given tags, return True.

        Parametesr:
        -----------
        *tags (FieldTags): The tags to match.

        Returns:
        --------
        bool: True if all given tags are matched, False otherwise.

        Examples:
        ---------
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
        r"""
        Return a new BufferField with the given or inherited data.

        Parameters:
        -----------
        data: Any
            Any column data.
        tags: Union[FieldTags, Iterable[FieldTags]] 
            Tags for filtering.

        Notes:
        ------
        The tags will also be inherited.
        """
        buffer_ = BufferField(data, tags, root=self)
        return buffer_

    def __repr__(self) -> str:
        tags = ','.join(map(lambda tag: str(tag).split('.')[-1], self.tags))
        return f"{self.__class__.__name__}({self.name}||{tags})"


class BufferField(Field):
    r"""
    For buffering data, which should be re-created once the data changes.
    
    Parameters:
    -----------
    data: Any
        Any column data.
    tags: Union[FieldTags, Iterable[FieldTags]] 
        Tags for filtering.
    root: Field, optional
        Inherit some attribuites from the root.
    """

    def __init__(
        self, data: Any, 
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple(),
        *, root: Optional[Field] = None
    ) -> None:
        super().__init__(data, tags)
        self.data = data
        if isinstance(tags, FieldTags):
            self.add_tag(tags)
        else:
            self.add_tag(*tags)
        self.inherit(root)

    def inherit(self, root: Union[None, 'BufferField', 'FieldModule']):
        if root is None:
            pass
        elif root.match(SPARSE):
            self.count = root.count
            self.add_tag(*root.tags)
        elif root.match(DENSE):
            self.add_tag(*root.tags)
        elif isinstance(root, Field):
            self.add_tag(*root.tags)
        else:
            raise ValueError(
                f"root should be `None|BufferField|FieldMoudle' but {type(root)} received ..."
            )

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

    def to_csr(self, length: Optional[int] = None) -> torch.Tensor:
        r"""
        Convert List to CSR Tensor.

        Notes:
        ------
        Each row in self.data should be the col indices !
        """
        data = self.data
        if isinstance(data, torch.Tensor):
            data = data.tolist()
        elif isinstance(data, np.ndarray):
            data = data.tolist()
        assert isinstance(data[0], (list, tuple)), f"Each row of data should be `list'|`tuple' but `{type(data[0])}' received ..."

        length = self.count if length is None else length
        crow_indices = np.cumsum([0] + list(map(len, self.data)), dtype=np.int64)
        col_indices = reduce(
            lambda x, y: x + y, data
        )
        values = np.ones_like(col_indices, dtype=np.int64)
        return torch.sparse_csr_tensor(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            size=(len(data), length) # B x Num of Items
        )


class FieldModule(Field, torch.nn.Module):
    r"""
    A module that represents a field of data.

    Attributes:
    -----------
    name: str 
        Name of this field.
    tags: set 
        A set of FieldTags.
    dtype: torch.dtype 
        Data type of the field.
    caster: Callable 
        Function to convert elements to specified dtype (int, float or str) with na_value.
    na_value: Union[str, int, float] 
        The fill value for null/missing values.
    dimension: int 
        The dimension of embeddings
    embeddings: torch.nn.Module
        Embedding module.

    Examples:
    ---------
    >>> from freerec.data.tags import USER, ITEM, ID, TARGET
    >>> User = SparseField('User', None, int, tags=(USER, ID))
    >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
    >>> Target = SparseField('Label', 0, int, transformer='none', tags=TARGET)
    """

    def __init__(
        self, name: str, 
        na_value: Optional[Union[str, int, float]], dtype: Callable,
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

        self.dimension: int

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

    def sparse_embed(
        self, 
        dim: int, 
        num_embeddings: Optional[int] = None,
        padding_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        self.dimension = dim
        self.embeddings = torch.nn.Embedding(
            num_embeddings, dim, padding_idx=padding_idx, **kwargs
        )

    def dense_embed(self, dim: int = 1, bias: bool = False, linear: bool = False) -> None:
        if linear:
            self.dimension = dim
            self.embeddings = torch.nn.Linear(1, dim, bias=bias)
        else:
            self.dimension = 1
            self.embeddings = torch.nn.Identity()

    def embed(
        self, dim: int,
        num_embeddings: Optional[int] = None, padding_idx: Optional[int] = None,
        bias: bool = False, linear: bool = False,
        **kwargs
    ):
        r"""
        Embed the field values into a lower dimensional space.

        Parameters:
        -----------
        dim: int 
            The dimension of the lower dimensional space.
        SPARSE:
            num_embeddings: int, optional
                The number of embeddings.
                - `None`: Set the number of embeddings to `count` or `count + 1` (if padding_idx is not None).
                - `int`: Set the number of embeddings to `int`.
            padding_idx: int, optional
                padding index
        DENSE:
            bias: bool 
                Add bias or not.
            linear: bool: 
                - `True`: nn.Linear.
                - `False`: using nn.Identity instead.
        **kwargs: 
            Other arguments for (SPARSE) embedding.

        Raises:
        -------
        NotImplementedError: if the method is not implemented.
        """
        if self.match(SPARSE):
            return self.sparse_embed(
                dim, num_embeddings, padding_idx,
                **kwargs
            )
        elif self.match(DENSE):
            return self.dense_embed(
                dim, bias, linear
            )
        else:
            raise NotImplementedError(f"{self.__class__.__name__}.embed() method should be implemented ...")

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Look up embeddings.

        Parameters:
        -----------
        x: torch.Tensor 
            The input tensor of shape (B, *), where B is the batch size
            and * can be any number of additional dimensions.

        Returns:
        --------
        embeddings: (B, *, 1) or (B, *, d)
            - `SPARSE`: The output tensor of shape (B, *, d), where d is the embedding dimension.
            - `DENSE`:
                - If `linear` is True, it returns (B, *, d).
                - If `linear` is False, it returns (B, *, 1).
        
        Examples:
        ---------
        >>> User: SparseField
        >>> ids = torch.arange(3).view(-1, 1)
        >>> User.look_up(ids).ndim
        3
        >>> Field: DenseField
        >>> vals = torch.rand(3, 1)
        >>> Field.look_up(vals).ndim
        3

        Raises:
        -------
        NotImplementedError: if the method (for other TYPEs) is not implemented.
        """
        if self.match(SPARSE):
            return self.embeddings(x)
        elif self.match(DENSE):
            return self.embeddings(x.unsqueeze(-1))
        else:
            raise NotImplementedError(f"{self.__class__.__name__}.look_up() method should be implemented...")

    def __str__(self) -> str:
        tags = ','.join(map(lambda tag: str(tag).split('.')[-1], self.tags))
        return f"{self.name}: {self.__class__.__name__}[dtype: {self.dtype}, na_value: {self.na_value}, tags: {tags}]"


class SparseField(FieldModule):
    r""" 
    SparseField inherits from FieldModule and represents sparse features. 
    It is used to look up embeddings for categorical features.

    Attributes:
    -----------
    name: str 
        Name of this field.
    tags: set 
        A set of FieldTags.
    dtype: torch.dtype 
        Data type of the field.
    caster: Callable 
        Function to convert elements to specified dtype (int, float or str) with na_value.
    na_value: Union[str, int, float] 
        The fill value for null/missing values.
    dimension: int 
        The dimension of embeddings
    embeddings: torch.nn.Module
        Embedding module.

    Tags:
    -----
    SPARSE
    """

    _exclusive_tags = {DENSE, AFFILIATE}

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable, 
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple(),
        transformer: Union[str, Callable] = 'label2index'
    ):
        super().__init__(name, na_value, dtype, tags, transformer)
        assert isinstance(self.transformer, Indexer), "SparseField supports 'Indexer' only !"
        self.add_tag(SPARSE)

    @property
    def count(self) -> Optional[int]:
        """Return the number of classes."""
        return self.transformer.count

    @property
    def enums(self) -> Optional[Tuple[int]]:
        """Return the tuple of IDs|..."""
        return self.transformer.enums

    def embed(
        self, 
        dim: int,
        num_embeddings: Optional[int] = None,
        padding_idx: Optional[int] = None,
        **kwargs
    ) -> None:
        if num_embeddings is None:
            nums = self.count if padding_idx is None else self.count + 1
        else:
            nums = num_embeddings
        return super().embed(
            dim, 
            num_embeddings=nums, padding_idx=padding_idx,
            **kwargs
        )

    def bind(
        self,
        meta_df: pd.DataFrame,
        rootCol: Union[int, str] = 0,
        mapper: Optional[dict] = None
    ):
        r"""
        Bind attributes for the current field.

        Parameters:
        -----------
        meta_df: DataFrame
            The attribute dataframe like:
                ------------------------------
                Item    Title   Brand   Price
                ------------------------------
                0       X       X       1.2
                1       Y       Y       1.3
                2       Z       Z       1.3
                ------------------------------
        rootCol: int, str
            - `int`: column index
            - `str`: column name
        mapper: dict: (key: (int, str), val (bool))
            Determining SPARSE/DENSE field 
        """
        columns = list(meta_df.columns)
        def index2column(index: Union[int, str]):
            if isinstance(index, int):
                return columns[index]
            elif isinstance(index, str):
                return index
            else:
                raise NotImplementedError(f"index should be `int` or `str`, but `{index}` received ...")

        if mapper is None:
            mapper = dict()
        else:
            mapper = {index2column(key): val for key, val in mapper.items()}
        
        rootCol = index2column(rootCol)
        meta_df[rootCol] = meta_df[rootCol].map(self.caster)
        meta_df = meta_df.sort_values(by=rootCol).reset_index(drop=True)

        affiliates = []
        for feat in columns:
            if feat == rootCol:
                continue
            affiliates.append(
                AffiliateField(
                    name=feat, root=self,
                    data=meta_df[feat].to_list(), 
                    is_dense=mapper.get(feat, False)
                )
            )
        return FieldTuple(affiliates)


class DenseField(FieldModule):
    r"""
    DenseField is used for numerical features.

    Attributes:
    -----------
    name: str 
        Name of this field.
    tags: set 
        A set of FieldTags.
    dtype: torch.dtype 
        Data type of the field.
    caster: Callable 
        Function to convert elements to specified dtype (int, float or str) with na_value.
    na_value: Union[str, int, float] 
        The fill value for null/missing values.
    dimension: int 
        The dimension of embeddings
    embeddings: torch.nn.Module
        Embedding module.

    Tags:
    -----
    DENSE
    """

    _exclusive_tags = {SPARSE, AFFILIATE}

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable, 
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple(),
        transformer: Union[str, Callable] = 'none'
    ):
        super().__init__(name, na_value, dtype, tags, transformer)
        self.add_tag(DENSE)


class AffiliateField(FieldModule):
    r"""
    Affiliate field for a root field (SparseField).

    Parameters:
    -----------
    name: str 
        Name of this field.
    root: SparseField
        root field
    data: Iterable 
        data

    Examples:
    ---------
    >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
    >>> ItemTitles = AffiliateField('Titles', data, Item)
    """

    _exclusive_tags = {ID}

    def __init__(
        self, name: str, 
        root: SparseField,
        data: Optional[Iterable] = None, 
        is_dense: bool = False
    ) -> None:
        super().__init__(
            name=name, na_value=None, dtype=None,
            tags=root.tags - {SPARSE, DENSE}, 
            transformer='none'
        )
        self.add_tag(AFFILIATE)
        if is_dense:
            self.add_tag(DENSE)
        else:
            self.add_tag(SPARSE)

        if data is not None:
            assert len(data) == root.count, f"The number of data ({len(data)}) is not equal to the number of Entities ({root.count})."

        self.data = data
        self.root = root

    def embed(
        self, dim: int,
        num_embeddings: Optional[int] = None, padding_idx: Optional[int] = None,
        bias: bool = False, linear: bool = False,
        **kwargs
    ):
        if num_embeddings is None:
            nums = self.root.count if padding_idx is None else self.root.count + 1
        else:
            nums = num_embeddings
        return super().embed(
            dim, 
            num_embeddings=nums, padding_idx=padding_idx,
            bias=bias, linear=linear,
            **kwargs
        )


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


class FieldList(list):
    """A list of fields, which support attribute access and filtering by tags."""

    def groupby(self, *tags: FieldTags) -> 'FieldList':
        r"""
        Return those fields matching given tags.

        Parameters:
        -----------
        *tags: FieldTags 
            Variable length argument list of FieldTags to filter the fields by.

        Returns:
        --------
        A new FieldList that contains only the fields whose tags match the given tags.
        """
        return FieldList(field for field in self if field.match(*tags))

    def groupbynot(self, *tags: FieldTags) -> 'FieldList':
        r"""
        Return those fields not matching given tags.

        Parameters:
        -----------
        *tags: FieldTags 
            Variable length argument list of FieldTags to filter the fields by.

        Returns:
        --------
            A new FieldList that contains only the fields whose tags do not match the given tags.
        """
        return FieldList(field for field in self if not field.match(*tags))

    def copy(self) -> 'FieldList':
        r"""
        Return a copy of the FieldList.

        Returns:
        --------
        A new FieldList with the same fields as this one.
        """
        return FieldList(self)

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
        >>> fields = FieldList([User, Item])
        >>> fields.index(USER, ID)
        0
        >>> fields.index(ITEM, ID)
        1
        """
        return super().index(self[tags])

    def __getitem__(self, index: Union[int, str, slice, FieldTags, Iterable[FieldTags]]) -> Union[Field, 'FieldList', None]:
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
        Fields: Union[Field, FieldList, None]
            - Field: If only one field matches given tags.
            - None: If none of fields matches given tags.
            - FieldList: If more than one field match given tags.

        Examples:
        ---------
        >>> from freerec.data.tags import USER, ITEM, ID
        >>> User = SparseField('User', None, int, tags=(USER, ID))
        >>> Item = SparseField('Item', None, int, tags=(ITEM, ID))
        >>> fields = FieldList([User, Item])
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
            fields = FieldList(
                super().__getitem__(index)
            )
        elif isinstance(index, str):
            fields = FieldList(field for field in self if field.name == index)
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
        super().__init__()

        self.fields = torch.nn.ModuleList(fields)

    @lru_cache(maxsize=4)
    def groupby(self, *tags: FieldTags) -> FieldList[FieldModule]:
        r"""
        Return those fields matching given tags.

        Parameters:
        -----------
        *tags: FieldTags 
            Variable length argument list of FieldTags to filter the fields by.

        Returns:
        --------
        A new FieldList that contains only the fields whose tags match the given tags.

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
        return FieldList(field for field in self.fields if field.match(*tags))

    @lru_cache(maxsize=4)
    def groupbynot(self, *tags: FieldTags) -> 'FieldList':
        r"""
        Return those fields not matching given tags.

        Parameters:
        -----------
        *tags: FieldTags 
            Variable length argument list of FieldTags to filter the fields by.

        Returns:
        --------
            A new FieldList that contains only the fields whose tags do not match the given tags.
        """
        return FieldList(field for field in self.fields if not field.match(*tags))

    def embed(
        self, dim: int, *tags: FieldTags,
        num_embeddings: Optional[int] = None, padding_idx: Optional[int] = None,
        bias: bool = False, linear: bool = False,
        **kwargs
    ):
        r"""
        Create embeddings.

        Parameters:
        -----------
        dim: int 
            The dimension of the lower dimensional space.
        *tags: FieldTags
        SPARSE:
            num_embeddings: int, optional
                The number of embeddings.
                - `None`: Set the number of embeddings to `count` or `count + 1` (if padding_idx is not None).
                - `int`: Set the number of embeddings to `int`.
            padding_idx: int, optional
                padding index
        DENSE:
            bias: bool 
                Add bias or not.
            linear: bool: 
                - `True`: nn.Linear.
                - `False`: using nn.Identity instead.
        **kwargs: 
            Other arguments for (SPARSE) embedding.



        Examples:
        ---------
        >>> from freerec.data.tags import DENSE, SPARSE
        >>> fields.embed(8, SPARSE) # nn.Embedding(count, 8)
        >>> fields.embed(8, DENSE, bias=False) # nn.Linear(1, 8, bias=False)
        """
        for field in self.groupby(*tags):
            field.embed(
                dim=dim, 
                num_embeddings=num_embeddings, padding_idx=padding_idx, # SPARSE
                bias=bias, linear=linear, # DENSE
                **kwargs
            )

    def calculate_dimension(self, *tags: FieldTags):
        """Return the total dimension of fields matching tiven tags."""
        return sum(field.dimension for field in self.groupby(*tags))

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, index: Union[int, str, FieldTags, Iterable[FieldTags]]) -> Union[FieldModule, 'FieldList', None]:
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
            - FieldList: If more than one field match given tags.

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
            fields = FieldList(
                super().__getitem__(index)
            )
        elif isinstance(index, str):
            fields = FieldList(field for field in self if field.name == index)
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