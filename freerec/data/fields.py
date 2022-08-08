


from typing import Callable, Iterable, Union, Dict, List

import torch
from functools import partial

from .utils import safe_cast
from .preprocessing import X2X, Label2Index, Binarizer, MinMaxScaler, StandardScaler



TRANSFORM = {
    "none": X2X,
    'label2index': Label2Index,
    'minmax': MinMaxScaler,
    'standard': StandardScaler,
}


class Field(torch.nn.Module):
    
    is_feature: bool

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable,
        transformer: Union[str, Callable] = 'none'
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
        self.dtype = dtype
        self.caster = partial(safe_cast, dest_type=dtype, default=na_value)
        self.dimension: int = 1
        self.transformer = TRANSFORM[transformer]() if isinstance(transformer, str) else transformer

    @classmethod
    def filter(cls, fields: Iterable, strict: bool = False):
        """
        strict == True:
            return fields whose class is exactly 'cls', i.e., calling 'type(field) == cls'
        strict == False:
            return fields inherited from 'cls', i.e., calling 'isinstance(field, cls)'
        """
        def _check_instance(instance):
            return isinstance(instance, cls)
        def _check_type(instance):
            return type(instance) == cls
        func = _check_type if strict else _check_instance
        return filter(func, fields)

    def partial_fit(self, x: Union[List[str], torch.Tensor]) -> torch.Tensor:
        if isinstance(x, list):
            self.transformer.partial_fit(x)
        elif isinstance(x, torch.Tensor):
            self.transformer.partial_fit(x.view(-1, 1))
        else:
            raise ValueError("x should be Union[List[str], torch.Tensor] ...")

    def transform(self, x: Union[List[str], torch.Tensor]) -> torch.Tensor:
        if isinstance(x, list):
            return torch.from_numpy(self.transformer.transform(x)).to(self.dtype)
        elif isinstance(x, torch.Tensor):
            return torch.from_numpy(self.transformer.transform(x.view(-1, 1))).to(self.dtype)
        else:
            raise ValueError("x should be Union[List[str], torch.Tensor] ...")

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

class SparseField(Field):

    is_feature: bool = True

    def __init__(
        self, name: str, na_value: Union[str, int, float], dtype: Callable, 
        transformer: str = 'label2index'
    ):
        super().__init__(name, na_value, dtype, transformer)

    def embed(self, dim: int, **kwargs):
        self.dimension = dim
        self.embeddings = torch.nn.Embedding(self.count, dim, **kwargs)

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        """ (B, *,) -> (B, *, d) """
        return self.embeddings(x)


class DenseField(Field):

    is_feature: bool = True

    def __init__(
        self, name: str, na_value: Union[str, int, float], 
        dtype: Callable, transformer: str = 'minmax'
    ):
        super().__init__(name, na_value, dtype, transformer)
    
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


class LabelField(SparseField): 
    is_feature: bool = False
    ...

class TagetField(DenseField):

    is_feature: bool = False

    def __init__(
        self, name: str, na_value: Union[str, int, float], 
        dtype: Callable, transformer: str = 'none'
    ):
        super().__init__(name, na_value, dtype, transformer)


class Tokenizer(torch.nn.Module):

    def __init__(self, fields: Iterable[Field]) -> None:
        super().__init__()

        self.features: List[Field] = [field for field in fields if field.is_feature]
        self.sparse = torch.nn.ModuleList(SparseField.filter(fields, strict=True))
        self.dense = torch.nn.ModuleList(DenseField.filter(fields, strict=True))

    def look_up(self, inputs: Dict[str, torch.Tensor], features: Union[str, List]) -> List[torch.Tensor]:
        """ Dict[torch.Tensor: B x 1] -> List[torch.Tensor: B x 1 x d]
        feautures:
            1. str: 'sparse'|'dense'|'all'(default)
            2. List[str]
            3. List[Field]
        """
        return [field.look_up(inputs[field.name]) for field in self.filter_features(features)]

    def flatten_cat(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """ List[torch.Tensor: B x k x d] -> torch.Tensor: B x (k1 x d1 + k2 x d2 + ...)"""
        return torch.cat([input_.flatten(1) for input_ in inputs], dim=1)

    def filter_features(self, features: Union[str, List] = 'all') -> List[Field]:
        """
        feautures:
            1. str: 'sparse'|'dense'|'all'(default)
            2. List[str]
            3. List[Field]
        """
        if features == 'all':
            features = self.features
        elif features == 'sparse':
            features = self.sparse
        elif features == 'dense':
            features = self.dense
        elif isinstance(features[0], Field):
            pass
        elif isinstance(features[0], str):
            features = [feature for feature in self.features if feature.name in features]
        else:
            raise ValueError(f"features should be: \n {self.filter_features.__doc__}")
        return features

    def embed(self, dim: int, features: Union[str, List] = 'sparse', **kwargs):
        """
        dim: int
        feautures:
            1. str: 'sparse'(default)|'dense'|'all'
            2. List[str]
            3. List[Field]
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
        for feature in self.filter_features(features):
            feature.embed(dim, **kwargs)

    def calculate_dimension(self, features: Union[str, List] = 'all'):
        """
        feautures:
            1. str: 'sparse'|'dense'|'all'(default)
            2. List[str]
            3. List[Field]
        """
        return sum(feature.dimension for feature in self.filter_features(features))
