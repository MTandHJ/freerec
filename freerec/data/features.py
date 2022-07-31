


from typing import Callable, Iterable, Tuple, Union, Any

import torch
from sklearn.preprocessing import LabelEncoder

from .utils import safe_cast
from ..dict2obj import Config



class Field(Config):

    def __init__(self, name: str, na_value: Union[str, int, float], dtype: Callable):
        """
        name: field
        na_value: fill na with na_value
        dtype: str, int ... for safe_cast
        """
        super().__init__(name=name, na_value=na_value, dtype=dtype)


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



