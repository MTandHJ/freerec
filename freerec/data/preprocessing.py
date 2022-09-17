

import numpy as np
import torch
import sklearn.preprocessing as preprocessing # LabelEncoder, Binarizer, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.utils._encode import _unique


__all__ = ['X2X', 'Label2Index', 'Binarizer', 'StandardScaler', 'MinMaxScaler']


class X2X(preprocessing.FunctionTransformer):
    """Identity transformation."""

    def partial_fit(self, x) -> None:
        ...

    def transform(self, x):
        return x


class Label2Index(preprocessing.LabelEncoder):
    """Convert labels to indices."""

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
        return super().transform(y)


class Binarizer(preprocessing.Binarizer):

    def partial_fit(self, x): ...


class StandardScaler(preprocessing.StandardScaler): ...
class MinMaxScaler(preprocessing.MinMaxScaler): ...


if __name__ == "__main__":

    transformer = Label2Index()
    transformer.partial_fit(torch.tensor([[1], [2], [3]]))
    print(transformer.transform(torch.tensor([[1], [2], [3]])).reshape(-1, 1))