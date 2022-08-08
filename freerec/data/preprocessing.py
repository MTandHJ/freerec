


from typing import Any, Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, Binarizer, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.utils._encode import _unique
from sklearn.utils.validation import column_or_1d

__all__ = ['X2X', 'Label2Index', 'Binarizer', 'StandardScaler', 'MinMaxScaler']

class X2X(FunctionTransformer):

    def partial_fit(self, x) -> None:
        ...

    def transform(self, x):
        return x.numpy()

class Label2Index(LabelEncoder):

    def _flatten(self, y):
        return np.ravel(np.asarray(y))

    def partial_fit(self, y):
        y = self._flatten(y)
        if not hasattr(self, 'classes_'):
            self.fit(y)
        else:
            self.classes_ = _unique(np.concatenate([self.classes_, y]))

    def transform(self, y):
        y = self._flatten(y)
        return super().transform(y).reshape(-1, 1)

class Binarizer(Binarizer):

    def partial_fit(self, x): ...



if __name__ == "__main__":

    transformer = Label2Index()
    transformer.partial_fit(torch.tensor([[1], [2], [3]]))
    print(transformer.transform(torch.tensor([[1], [2], [3]])).reshape(-1, 1))