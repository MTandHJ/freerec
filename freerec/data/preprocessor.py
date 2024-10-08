

from typing import List, Union

import math
import polars as pl


__all__ = ['Identifier', 'Indexer', 'StandardScaler', 'MinMaxScaler']


class preprocessorror(Exception): ...


class Preprocessor:
    r"""
    Transform X into X.

    Parameters:
    -----------
    X : Any
        The input data to be transformed.

    Returns:
    --------
    Any
        The transformed data.

    Examples:
    ---------
    >>> col = [3, 2, 1]
    >>> preprocessor = Preprocessor()
    >>> preprocessor.transform(col)
    [3, 2, 1]
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.fitting = True

    @property
    def count(self):
        return len(self.olds)

    def partial_fit(self, data: pl.Series):
        self.fitting = True

    def transform(self, data: pl.Series) -> pl.Series:
        self.fitting = False
        return data


class Counter(Preprocessor):
    """Counting uniques."""

    def reset(self):
        super().reset()
        self.olds = []

    @property
    def count(self):
        return len(self.olds)

    def partial_fit(self, data: pl.Series):
        self.fitting = True
        if not data.dtype.is_float():
            self.olds = set(self.olds) | set(data.unique())

    def transform(self, data: pl.Series) -> pl.Series:
        self.fitting = False
        return data


class ReIndexer(Counter):
    r"""
    Transform categorical features into tokens.

    classes: set
        Set of unique classes seen during `fit`.
    mapper: dict
        Mapping of input values to output indices.

    Examples:
    ---------
    >>> col = [3, 2, 1]
    >>> col2 = [4, 5, 6]
    >>> preprocessor = Tokenizer()
    >>> preprocessor.partial_fit(col)
    >>> preprocessor.classes
    {1, 2, 3}
    >>> preprocessor.mapper
    {1: 0, 2: 1, 3: 2}
    >>> preprocessor.partial_fit(col2)
    >>> preprocessor.classes
    {1, 2, 3, 4, 5, 6}
    >>> preprocessor.mapper
    {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    >>> preprocessor.transform(col2)
    [3, 4, 5]
    """

    def transform(self, data: pl.Series) -> pl.Series:
        if self.fitting:
            self.olds = sorted(self.olds)
            self.news = list(range(self.count))
            self.fitting = False
        return data.replace_strict(old=self.olds, new=self.news)


class StandardScaler(Preprocessor):
    r"""
    Normalize numerical features using the standard scaler.

    Attributes:
    -----------
    nums: int
        The number of fitted features.
    sum: float
        The summation of the fitted features.
    ssum: float 
        The summation of the squared fitted items.

    Examples:
    ---------
    >>> col = [3., 2., 1.]
    >>> preprocessor = StandardScaler()
    >>> preprocessor.partial_fit(col)
    >>> preprocessor.transform(col)
    [1.2247448713915887, 0.0, -1.2247448713915887]
    """

    def reset(self):
        super().reset()
        self.nums = 0
        self.sum = 0
        self.ssum = 0 # sum of squared
        self.eps = 1.e-8

    @property
    def mean(self):
        return self.sum / self.nums

    @property
    def std(self):
        return math.sqrt(
            (self.ssum - self.sum ** 2 / self.nums) / (self.nums - 1)
        )

    def partial_fit(self, data: pl.Series):
        self.fitting = True
        self.nums += data.len()
        self.sum += data.sum()
        self.ssum += data.pow(2).sum()

    def transform(self, data: pl.Series) -> pl.Series:
        self.fitting = False
        return (data - self.mean) / (self.std + self.eps)


class MinMaxScaler(Preprocessor):
    r"""
    Scale data to the range [0, 1].

    Attributes:
    -----------
    min: float
        The minimum value in the data.
    max: float
        The maximum value in the data.

    Examples:
    ---------
    >>> col = [3., 2., 1.]
    >>> preprocessor = MinMaxScaler()
    >>> preprocessor.partial_fit(col)
    >>> preprocessor.transform(col)
    [1.0, 0.5, 0.0]
    """

    def reset(self):
        super().reset()
        self.min = float('inf')
        self.max = float('-inf')
        self.eps = 1.e-8

    def partial_fit(self, data: pl.Series):
        self.fitting = True
        self.min = min(data.min(), self.min)
        self.max = max(data.max(), self.max)

    def transform(self, data: pl.Series) -> pl.Series:
        self.fitting = False
        return (data - self.min) / (self.max - self.min + self.eps)


if __name__ == "__main__":
    import doctest
    doctest.testmod()