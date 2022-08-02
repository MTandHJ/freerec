


from typing import Callable, Iterable, Tuple, Union, Any, Dict, List
from pyparsing import Optional

import torch
from sklearn.preprocessing import LabelEncoder

from .utils import safe_cast



class Field(torch.nn.Module):
    
    is_feature: bool

    def __init__(self, name: str, na_value: Union[str, int, float], dtype: Callable):
        """
        name: field
        na_value: fill na with na_value
        dtype: str, int ... for safe_cast
        """
        super().__init__()

        self.__name = name
        self.__na_value = na_value
        self.__dtype = dtype
        self.dimension: int = 1

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

    @property
    def name(self):
        return self.__name

    @property
    def na_value(self):
        return self.__na_value
    
    @property
    def dtype(self):
        return self.__dtype

    def __call__(self, val: Any) -> Any:
        return safe_cast(val, self.dtype, self.na_value)


class SparseField(Field):

    is_feature: bool = True

    def fit(self, enum: Iterable):
        """
        enum: all possible values (unique)
        """
        self.enum = enum
        self.count = len(enum)
        self.transformer = LabelEncoder()
        self.transformer.fit(enum)

    def transform(self, x: Iterable) -> torch.Tensor:
        return torch.from_numpy(self.transformer.transform(x)).long().view(-1, 1)

    def embed(self, dim: int, **kwargs):
        self.dimension = dim
        self.embeddings = torch.nn.Embedding(self.count, dim, **kwargs)

    def look_up(self, x: torch.Tensor) -> torch.Tensor:
        """
            (B, *,) -> (B, *, d)
        """
        self.embeddings: torch.nn.Embedding
        return self.embeddings(x)



class DenseField(Field):

    is_feature: bool = True
    
    def fit(self, lower: float, upper: float, bound: Tuple = (0., 1.)):
        """
        lower: minimum
        upper: maximum
        bound: the bound after scaling
        """
        data_range = upper - lower
        feat_range = bound[1] - bound[0]
        self.scale_ = feat_range / data_range
        self.min_ = bound[0] - lower * self.scale_


    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.scale_ + self.min_).float().view(-1, 1)


class LabelField(SparseField): 
    is_feature: bool = False
    ...

class TagetField(DenseField):

    is_feature: bool = False

    def fit(self, lower: float, upper: float, bound: Tuple = (0, 1)):
        ...

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().view(-1, 1)


class Tokenizer(torch.nn.Module):

    def __init__(self, fields: Iterable[Field]) -> None:
        super().__init__()

        self.features = [field for field in fields if field.is_feature]
        self.sparse = torch.nn.ModuleList(SparseField.filter(fields, strict=True))
        self.dense = torch.nn.ModuleList(DenseField.filter(fields, strict=True))

    def look_up(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Dict[torch.Tensor: B x 1] -> List[torch.Tensor: B x 1 x d]
        """
        return [field.look_up(inputs[field.name]) for field in self.sparse]

    def collect_dense(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Dict[torch.Tensor: B x 1] -> torch.Tensor: B x len(self.dense)
        """
        return torch.cat([inputs[field.name] for field in self.dense], dim=1)

    def flatten_cat(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        List[torch.Tensor: B x k x d] -> torch.Tensor: B x (k1 x d1 + k2 x d2 + ...)
        """
        return torch.cat([input_.flatten(1) for input_ in inputs], dim=1)

    def dimension(self, features: Union[str, List] = 'all'):
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
            raise ValueError(f"features should be: \n {self.dimension.__doc__}")
        return sum(feature.dimension for feature in features)
