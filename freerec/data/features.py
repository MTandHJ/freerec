


from typing import Callable, Iterable, Tuple, Union, Any, Dict, List

import torch
from sklearn.preprocessing import LabelEncoder

from .utils import safe_cast


class Field(torch.nn.Module):

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

    def fit(self, enum: Iterable):
        """
        enum: all possible values (unique)
        """
        self.enum = enum
        self.count = len(enum)
        self.transformer = LabelEncoder()
        self.transformer.fit(enum)

    def transform(self, x: Iterable) -> torch.Tensor:
        return torch.from_numpy(self.transformer.transform(x)).long()

    def embed(self, dim: int, **kwargs):
        self.embeddings = torch.nn.Embedding(self.count, dim, **kwargs)

    def look_up(self, x: torch.Tensor) -> torch.Tensor: # B x d
        self.embeddings: torch.nn.Embedding
        return self.embeddings(x)



class DenseField(Field):

    
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
        return (x * self.scale_ + self.min_).float()


class LabelField(SparseField): ...
class TagetField(DenseField):

    def fit(self, lower: float, upper: float, bound: Tuple = (0, 1)):
        ...

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Tokenizer(torch.nn.Module):

    def __init__(self, fields: Iterable[Field]) -> None:
        super().__init__()

        self.sparse = torch.nn.ModuleList(SparseField.filter(fields))
        self.dense = torch.nn.ModuleList(DenseField.filter(fields))

    def look_up(self, inputs: Dict[torch.Tensor]) -> List[torch.Tensor]:
        """
        Dict[torch.Tensor: (B, )] -> List[torch.Tensor: B x d]
        """
        return [field.look_up(inputs[field.name]) for field in self.sparse]

    def collect_dense(self, inputs: Dict[torch.Tensor]) -> torch.Tensor:
        """
        Dict[torch.Tensor: (B, )] -> torch.Tensor: B x len(self.dense)
        """
        return torch.stack([inputs[field.name] for field in self.dense], dim=1)

    def flatten_cat(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        List[torch.Tensor: B x k x d] -> torch.Tensor: B x (k1 x d1 + k2 x d2 + ...)
        """
        return torch.cat([input_.flatten(1) for input_ in inputs])

