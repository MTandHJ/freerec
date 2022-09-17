

from typing import Callable, Iterable, Tuple, Union, Dict, List

import torch
import numpy as np
from functools import partial, lru_cache

from .utils import safe_cast
from .preprocessing import X2X, Label2Index, Binarizer, MinMaxScaler, StandardScaler
from .tags import Tag, SPARSE, DENSE
from ..utils import errorLogger


__all__ = ['Field', 'DenseField', 'SparseField', 'Token', 'SparseToken', 'DenseToken', 'Tokenizer']

TRANSFORM = {
    "none": X2X,
    'label2index': Label2Index,
    'minmax': MinMaxScaler,
    'standard': StandardScaler,
}


class Field:
    
    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable,
        transformer: Union[str, Callable] = 'none', tags: Union[Tag, Iterable[Tag]] = tuple()
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
        self.add_tag(tags)

    def add_tag(self, tags: Union[Tag, Iterable[Tag]]):
        if isinstance(tags, Tag):
            self.__tags.add(tags.name)
        else:
            for tag in tags:
                self.add_tag(tag) 

    @property
    def tags(self):
        return self.__tags

    def match(self, tags: Union[Tag, Iterable[Tag]]):
        """If current field matches the given tags, return True."""
        if isinstance(tags, Iterable):
            return all(map(self.match, tags))
        return tags.name in self.tags # using name for mathcing to avoid deepcopy pitfall !

    def partial_fit(self, x: np.array) -> np.array:
        self.transformer.partial_fit(x)

    def transform(self, x: np.array) -> np.array:
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
        tags = ','.join(self.tags)
        return f"{self.name}: [dtype: {self.dtype}, na_value: {self.na_value}, tags: {tags}]"


class SparseField(Field):

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable, 
        transformer: Union[str, Callable] = 'label2index', 
        tags: Union[Tag, Iterable[Tag]] = tuple()
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
        tags: Union[Tag, Iterable[Tag]] = tuple()
    ):
        super().__init__(name, na_value, dtype, transformer, tags)
        self.add_tag(DENSE)
   

class Fielder(tuple):

    @lru_cache(maxsize=4)
    def whichis(self, *tags: Tag):
        """Return those fields matching given tags."""
        fields = Fielder(field for field in self if field.match(tags))
        if len(fields) == 1:
            return fields[0]
        elif len(fields) == 0:
            return None
        else:
            return fields

    @lru_cache(maxsize=4)
    def whichisnot(self, *tags: Tag):
        """Return those fields not matching given tags."""
        fields = Fielder(field for field in self if not field.match(tags))
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

    @property
    def name(self):
        return self.__name

    @property
    def tags(self):
        return self.__tags

    def add_tag(self, tags: Union[Tag, Iterable[Tag]]):
        if isinstance(tags, Tag):
            self.__tags.add(tags.name)
        else:
            for tag in tags:
                self.add_tag(tag) 

    def match(self, tags: Union[Tag, Iterable[Tag]]):
        """If current token matches the given tags, return True."""
        if isinstance(tags, Iterable):
            return all(map(self.match, tags))
        return tags.name in self.tags

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
        self.dimension = dim
        self.embeddings = torch.nn.Embedding(self.count, dim, **kwargs)

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        """(B, *,) -> (B, *, d)"""
        return self.embeddings(x)


class DenseToken(Token):

    def embed(self, dim: int = 1, bias: bool = False, linear: bool = False):
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
    """A collection of tokens."""

    def __init__(self, fields: Iterable[Field]) -> None:
        super().__init__()

        fields = [self.totoken(field) for field in fields]
        self.tokens = torch.nn.ModuleList(fields)
    
    def totoken(self, field: Field):
        """Field to Token."""
        if isinstance(field, SparseField):
            return SparseToken(field)
        elif isinstance(field, DenseField):
            return DenseToken(field)
        else:
            errorLogger("only Sparse|DenseField supported !", ValueError)

    @lru_cache(maxsize=4)
    def groupby(self, *tags: Union[Tag, Tuple[Tag]]) -> List[Token]:
        """Group by given tags and return list of tokens matched."""
        if len(tags) == 0:
            return self.tokens
        return [token for token in self.tokens if token.match(tags)]

    def look_up(self, inputs: Dict[str, torch.Tensor], *tags: Union[Tag, Tuple[Tag]]) -> List[torch.Tensor]:
        """Dict[torch.Tensor: B x 1] -> List[torch.Tensor: B x 1 x d]"""
        return [token.look_up(inputs[token.name]) for token in self.groupby(*tags)]

    def embed(self, dim: int, *tags: Union[Tag, Tuple[Tag]], **kwargs):
        """
        Args:
            dim: int
            tags: Tag|Tuple[Tag]
                'sparse': nn.Embedding
                    padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                                therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                                i.e. it remains as a fixed "pad". For a newly constructed Embedding,
                                                the embedding vector at :attr:`padding_idx` will default to all zeros,
                                                but can be updated to another value to be used as the padding vector.
                    max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                                is renormalized to have norm :attr:`max_norm`.
                    norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
                    scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                            the words in the mini-batch. Default ``False``.
                    sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.
                                            See Notes for more details regarding sparse gradients.
                'dense': nn.Linear
                    linear: bool = False
                    bias: bool = False
        Kwargs: kwargs for nn.Linear
        """
        for feature in self.groupby(*tags):
            feature.embed(dim, **kwargs)

    def calculate_dimension(self, *tags: Union[Tag, Tuple[Tag]]):
        """Return the total dimension of tokens matching tiven tags."""
        return sum(feature.dimension for feature in self.groupby(*tags))
