

from typing import Callable, Iterable, Tuple, Union, Dict, List
from pyparsing import Optional

import torch
import numpy as np
from functools import partial, lru_cache

from .utils import safe_cast
from .preprocessing import X2X, Label2Index, Binarizer, MinMaxScaler, StandardScaler
from .tags import FieldTags, SPARSE, DENSE
from ..utils import errorLogger, warnLogger


__all__ = ['Field', 'DenseField', 'SparseField', 'Token', 'SparseToken', 'DenseToken', 'Tokenizer']

TRANSFORM = {
    "none": X2X,
    'label2index': Label2Index,
    'minmax': MinMaxScaler,
    'standard': StandardScaler,
}


class Field:
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
    transformer: Callable
        Transformation.
    """
    
    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable,
        transformer: Union[str, Callable] = 'none', tags: Union[FieldTags, Iterable[FieldTags]] = tuple()
    ):
        """
        Args:
            name: the name of the field
            na_value: fill 'na' with na_value
            dtype: str|int|float for safe_cast
        Kwargs:
            transformer: 'none'|'label2index'|'binary'|'minmax'|'standard'
            tags: tags for quick search
        """

        self.__name = name
        self.__na_value = na_value
        self.__tags = set()
        self.dtype = dtype
        self.caster = partial(safe_cast, dest_type=dtype, default=na_value)
        self.transformer = TRANSFORM[transformer]() if isinstance(transformer, str) else transformer
        if isinstance(tags, FieldTags):
            self.add_tag(tags)
        else:
            self.add_tag(*tags)

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
                warnLogger(f"[Warning] FieldTags is expected but {type(tags)} received ...")
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

    def partial_fit(self, x: np.array) -> np.array:
        """Partially fit some data."""
        self.transformer.partial_fit(x)

    def transform(self, x: np.array) -> np.array:
        """Transform data."""
        return self.transformer.transform(x)

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
        elif val == float:
            self.__dtype = torch.float32
        else:
            self.__dtype = val

    def __str__(self) -> str:
        tags = ','.join(map(str, self.tags))
        return f"{self.name}: [dtype: {self.dtype}, na_value: {self.na_value}, tags: {tags}]"


class SparseField(Field):

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable, 
        transformer: Union[str, Callable] = 'label2index', 
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple()
    ):
        super().__init__(name, na_value, dtype, transformer, tags)
        self.add_tag(SPARSE)

    @property
    def count(self):
        """Return the number of classes."""
        if isinstance(self.transformer, Label2Index):
            return len(self.transformer.classes_)
        else:
            return None


class DenseField(Field):

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable, 
        transformer: Union[str, Callable] = 'minmax', 
        tags: Union[FieldTags, Iterable[FieldTags]] = tuple()
    ):
        super().__init__(name, na_value, dtype, transformer, tags)
        self.add_tag(DENSE)
   

class Fielder(tuple):

    @lru_cache(maxsize=4)
    def whichis(self, *tags: FieldTags) -> Union[FieldTags, None, 'Fielder']:
        """Return those fields matching given tags.
        
        Returns:
        ---

        - `Field`: If only one field matches given tags.
        - `None`: If none of fields matches given tags.
        - `Fielder`: If more than one fields match given tags.
        """
        fields = Fielder(field for field in self if field.match(*tags))
        if len(fields) == 1:
            return fields[0]
        elif len(fields) == 0:
            return None
        else:
            return fields

    @lru_cache(maxsize=4)
    def whichisnot(self, *tags: FieldTags) -> Union[FieldTags, None, 'Fielder']:
        """Return those fields not matching given tags.

        Returns:
        ---

        - `Field`: If only one field does not match given tags.
        - `None`: If all fields match given tags.
        - `Fielder`: If more than one fields does not match given tags.
        """
        fields = Fielder(field for field in self if not field.match(*tags))
        if len(fields) == 1:
            return fields[0]
        elif len(fields) == 0:
            return None
        else:
            return fields

    def state_dict(self) -> Dict:
        """Return state dict of fields."""
        return {field.name: field.transformer for field in self}

    def load_state_dict(self, state_dict: Dict, strict: bool = False):
        for field in self:
            field.transformer = state_dict.get(field.name, field.transformer)

    def copy(self) -> 'Fielder':
        return Fielder(self)


class Token(torch.nn.Module):

    def __init__(self, field: Field) -> None:
        super().__init__()
        self.__name = field.name
        self.__tags = field.tags.copy()

        self.dimension: int
        self.embeddings: Union[torch.nn.Embedding, torch.nn.Linear, torch.nn.Identity]

    @property
    def name(self):
        return self.__name

    @property
    def tags(self):
        return self.__tags

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
                warnLogger(f"[Warning] FieldTags is expected but {type(tags)} received ...")
            self.__tags.add(tag)

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

    def embed(self, dim: int, **kwargs):
        errorLogger("embed method should be specified ...", NotImplementedError)


class SparseToken(Token):

    def __init__(self, field: Field) -> None:
        super().__init__(field)

        self.__count = field.count

    @property
    def count(self):
        """Return the number of classes."""
        return self.__count

    def embed(self, dim: int, **kwargs):
        """Create nn.Embedding.

        Parameters:
        ---

        dim: int
            Dimension.
        **kwargs: other kwargs for nn.Embedding

        Examples:
        ---

        >>> token = SparseToken(field)
        >>> token.embed(8)
        """

        self.dimension = dim
        self.embeddings = torch.nn.Embedding(self.count, dim, **kwargs)

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        """(B, *,) -> (B, *, d)"""
        return self.embeddings(x)


class DenseToken(Token):

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

        Examples:
        ---

        >>> token = DenseToken(field)
        >>> token.embed(8, bias=True, linear=False)
        """
        if linear:
            self.dimension = dim
            self.embeddings = torch.nn.Linear(1, dim, bias=bias)
        else:
            self.dimension = 1
            self.embeddings = torch.nn.Identity()

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        """(B, *,) -> (B, *) | (B, *, d)"""
        return self.embeddings(x)


class Tokenizer(torch.nn.Module):
    """A collection of tokens.
    
    Attributes:
    ---

    tokens: nn.ModuleList

    Examples:
    ---

    >>> from freerec.data import MovieLens1M
    >>> basepipe = MovieLens1M("../data/MovieLens1M")
    >>> fields = basepipe.fields
    >>> tokenizer = Tokenizer(fields)

    """

    def __init__(self, fields: Iterable[Union[Field, Token]]) -> None:
        super().__init__()

        fields = [self.totoken(field) for field in fields]
        self.tokens = torch.nn.ModuleList(fields)
    
    def totoken(self, field: Field):
        """Field to Token."""
        if isinstance(field, Token):
            return field
        elif isinstance(field, SparseField):
            return SparseToken(field)
        elif isinstance(field, DenseField):
            return DenseToken(field)
        else:
            errorLogger("Only Sparse|DenseField supported !")

    @lru_cache(maxsize=4)
    def groupby(self, *tags: FieldTags) -> List[Token]:
        """Group by given tags and return list of tokens matched.
        
        Examples:
        ---
        >>> from freerec.data.tags import USER, ID
        >>> User = tokenizer.groupby(USER, ID)
        >>> isinstance(User, List)
        True
        >>> Item = tokenizer.groupby(ITEM, ID)[0]
        >>> Item.match(ITEM)
        True
        >>> Item.match(ID)
        True
        >>> Item.match(User)
        False

        """
        return [token for token in self.tokens if token.match(*tags)]

    def embed(self, dim: int, *tags: FieldTags, **kwargs):
        """Create embeddings.

        Parameters:
        ---
        dim: int
            Dimension.
        *tags: FieldTags
        **kwargs: kwargs for nn.Linear

        Examples:
        ---

        >>> from freerec.data.tags import DENSE, SPARSE
        >>> tokenizer.embed(8, SPARSE) # nn.Embedding(count, 8)
        >>> tokenizer.embed(8, DENSE, bias=False) # nn.Linear(1, 8, bias=False)
        """
        for feature in self.groupby(*tags):
            feature.embed(dim, **kwargs)

    def look_up(self, inputs: Dict[str, torch.Tensor], *tags: FieldTags) -> List[torch.Tensor]:
        """Dict[torch.Tensor: B x 1] -> List[torch.Tensor: B x 1 x d]
        
        Parameters:
        ---

        inputs: Dict[str, torch.Tensor]
            A dict of inputs.

        Returns:
        ---

        A list of torch.Tensor.

        Examples:
        ---

        >>> from freerec.data.tags import User, ID
        >>> userEmbds = tokenizer.look_up(users, USER, ID)
        >>> isinstance(userEmbds, List)
        True

        """
        return [token.look_up(inputs[token.name]) for token in self.groupby(*tags)]

    def calculate_dimension(self, *tags: FieldTags):
        """Return the total dimension of tokens matching tiven tags."""
        return sum(feature.dimension for feature in self.groupby(*tags))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index: Union[int, FieldTags, Iterable[FieldTags]]) -> Union[Token, None, 'Tokenizer']:
        """Get tokens by index.

        Parameters:
        ---

        index: Union[int, FieldTags, Iterable[FieldTags]]
            - `int`: Return the token at position `int`.
            - `FieldTags`: Return the tokens matching `FieldTags`.
            - `Iterable[FieldTags]`: Return the tokens matching `Iterable[FieldTags]`.
        
        Returns:
        ---

        - `Token`: If only one token matches given tags.
        - `None`: If none of tokens matches given tags.
        - `Tokenizer`: If more than one tokens match given tags.

        Examples:
        ---

        >>> from freerec.data.tags import USER
        >>> tokenizer[0];
        >>> user = tokenizer[USER]
        >>> isinstance(user, Token)
        True
        >>> isinstance(tokenizer[ID], Tokenizer)
        True
        """
        if isinstance(index, int):
            tokens = [self.tokens[index]]
        elif isinstance(index, FieldTags):
            tokens =  self.groupby(index)
        else:
            tokens = self.groupby(*index)
        if len(tokens) == 1:
            return tokens[0]
        elif len(tokens) == 0:
            return None # for a safety purpose
        else:
            return Tokenizer(tokens)
