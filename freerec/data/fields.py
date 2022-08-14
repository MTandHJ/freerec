


from typing import Callable, Iterable, Tuple, Union, Dict, List

import torch
import numpy as np
from functools import partial, lru_cache

from .utils import safe_cast
from .preprocessing import X2X, Label2Index, Binarizer, MinMaxScaler, StandardScaler
from .tags import Tag, SPARSE, DENSE, FEATURE, NEGATIVE



TRANSFORM = {
    "none": X2X,
    'label2index': Label2Index,
    'minmax': MinMaxScaler,
    'standard': StandardScaler,
}


class Field(torch.nn.Module):
    
    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable,
        transformer: Union[str, Callable] = 'none', tags: Union[Tag, Iterable[Tag]] = tuple()
    ):
        """
        name: field
        na_value: fill na with na_value
        dtype: str|int|float for safe_cast
        transformer: 'none'|'label2index'|'binary'|'minmax'|'standard'
        """
        super().__init__()

        self.__name = name
        self.__na_value = na_value
        self.__tags = set()
        self.dtype = dtype
        self.caster = partial(safe_cast, dest_type=dtype, default=na_value)
        self.dimension: int = 1
        self.transformer = TRANSFORM[transformer]() if isinstance(transformer, str) else transformer
        self.add_tag(tags)

    def add_tag(self, tags: Union[Tag, Iterable[Tag]]):
        if isinstance(tags, Tag):
            self.__tags.add(tags)
        else:
            self.__tags = self.__tags | set(tags)

    def remove_tag(self, tags: Union[Tag, Iterable[Tag]]):
        if isinstance(tags, Tag):
            self.__tags.remove(tags)
        else:
            self.__tags = self.__tags - set(tags)

    @property
    def tags(self):
        return self.__tags

    def match(self, tags: Union[Tag, Iterable[Tag]]):
        if isinstance(tags, Iterable):
            return all(map(self.match, tags))
        return tags in self.tags

    def partial_fit(self, x: np.array) -> np.array:
        self.transformer.partial_fit(x)

    def transform(self, x: np.array) -> np.array:
        return self.transformer.transform(x)

    def embed(self, dim: int, **kwargs):
        raise NotImplementedError()

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
        tags: Union[Tag, Iterable[Tag]] = tuple()
    ):
        super().__init__(name, na_value, dtype, transformer, tags)
        self.add_tag(SPARSE)

    @property
    def count(self):
        if isinstance(self.transformer, Label2Index):
            return len(self.transformer.classes_)
        else:
            raise NotImplementedError()

    def embed(self, dim: int, **kwargs):
        self.dimension = dim
        self.embeddings = torch.nn.Embedding(self.count, dim, **kwargs)

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        """ (B, *,) -> (B, *, d) """
        return self.embeddings(x)


class DenseField(Field):

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable, 
        transformer: Union[str, Callable] = 'minmax', 
        tags: Union[Tag, Iterable[Tag]] = tuple()
    ):
        super().__init__(name, na_value, dtype, transformer, tags)
        self.add_tag(DENSE)
   
    def embed(self, dim: int = 1, bias: bool = False, linear: bool = False):
        if linear:
            self.dimension = dim
            self.embeddings = torch.nn.Linear(1, dim, bias=bias)
        else:
            self.dimension = 1
            self.embeddings = torch.nn.Identity()

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        """ (B, *,) -> (B, *) | (B, *, d) """
        return self.embeddings(x)


class Tokenizer(torch.nn.Module):

    def __init__(self, fields: Iterable[Field]) -> None:
        super().__init__()

        fields = [field for field in fields if not field.match(NEGATIVE)]
        self.fields = torch.nn.ModuleList(fields)

    @lru_cache(maxsize=4)
    def groupby(self, tags: Union[str, Tag, Field, Tuple] = 'all') -> List[Field]:
        """
        tags:
            1. str: 'sparse'|'dense'|'all'(default)
            2. Tag:
            3. Tuple[Tag]
            4. Tuple[Field]
        """
        if tags == 'all':
            return self.fields
        if not isinstance(tags, Iterable):
            tags = (tags,)
        if isinstance(tags[0], Field):
            return tags
        return [field for field in self.fields if field.match(tags)]

    def look_up(self, inputs: Dict[str, torch.Tensor], tags: Union[str, Tag, Field, Tuple]) -> List[torch.Tensor]:
        """ Dict[torch.Tensor: B x 1] -> List[torch.Tensor: B x 1 x d]
        tags:
            1. str: 'sparse'|'dense'|'all'(default)
            2. Tag:
            3. Tuple[Tag]
            4. Tuple[Field]
        """
        return [field.look_up(inputs[field.name]) for field in self.groupby(tags)]

    def flatten_cat(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """ List[torch.Tensor: B x k x d] -> torch.Tensor: B x (k1 x d1 + k2 x d2 + ...)"""
        return torch.cat([input_.flatten(1) for input_ in inputs], dim=1)

    def embed(self, dim: int, tags: Union[str, Tag, Field, Tuple] = SPARSE, **kwargs):
        """
        dim: int
        tags:
            1. str: 'sparse'|'dense'|'all'(default)
            2. Tag:
            3. Tuple[Tag]
            4. Tuple[Field]
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
        """
        for feature in self.groupby(tags):
            feature.embed(dim, **kwargs)

    def calculate_dimension(self, tags: Union[str, Tag, Field, Tuple] = 'all'):
        """
        tags:
            1. str: 'sparse'|'dense'|'all'(default)
            2. Tag:
            3. Tuple[Tag]
            4. Tuple[Field]
        """
        return sum(feature.dimension for feature in self.groupby(tags))
