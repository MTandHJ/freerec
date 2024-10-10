

import math
import polars as pl


__all__ = ['Normalizer', 'Counter', 'ReIndexer', 'StandardScaler', 'MinMaxScaler']


class Normalizer:
    r"""
    Normalize X into X.

    Parameters:
    -----------
    X : Any
        The input data to be normalized.

    Returns:
    --------
    Any
        The normalized data.

    Examples:
    ---------
    >>> col = pl.Series([3, 2, 1])
    >>> normalizer = Normalizer()
    >>> normalizer.normalize(col).to_list()
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

    def normalize(self, data: pl.Series) -> pl.Series:
        self.fitting = False
        return data

    def __call__(self, data: pl.Series) -> pl.Series:
        return self.normalize(data)


class Counter(Normalizer):
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

    def normalize(self, data: pl.Series) -> pl.Series:
        self.fitting = False
        return data


class ReIndexer(Counter):
    r"""
    Normalize categorical features into tokens.

    classes: set
        Set of unique classes seen during `fit`.
    mapper: dict
        Mapping of input values to output indices.

    Examples:
    ---------
    >>> col = pl.Series([3, 2, 1])
    >>> col2 = pl.Series([4, 5, 6])
    >>> normalizer = ReIndexer()
    >>> normalizer.partial_fit(col)
    >>> normalizer.olds
    {1, 2, 3}
    >>> normalizer.partial_fit(col2)
    >>> normalizer.olds
    {1, 2, 3, 4, 5, 6}
    >>> normalizer.normalize(col2).to_list()
    [3, 4, 5]
    >>> normalizer.olds
    [1, 2, 3, 4, 5, 6]
    """

    def normalize(self, data: pl.Series) -> pl.Series:
        if self.fitting:
            self.olds = sorted(self.olds)
            self.news = list(range(self.count))
            self.fitting = False
        return data.replace_strict(old=self.olds, new=self.news)


class StandardScaler(Normalizer):
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
    >>> col = pl.Series([3., 2., 1.])
    >>> normalizer = StandardScaler()
    >>> normalizer.partial_fit(col)
    >>> normalizer.normalize(col).to_list()
    [0.9999999900000002, 0.0, -0.9999999900000002]
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

    def normalize(self, data: pl.Series) -> pl.Series:
        self.fitting = False
        return (data - self.mean) / (self.std + self.eps)


class MinMaxScaler(Normalizer):
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
    >>> col = pl.Series([3., 2., 1.])
    >>> normalizer = MinMaxScaler()
    >>> normalizer.partial_fit(col)
    >>> normalizer.normalize(col).to_list()
    [0.999999995, 0.4999999975, 0.0]
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

    def normalize(self, data: pl.Series) -> pl.Series:
        self.fitting = False
        return (data - self.min) / (self.max - self.min + self.eps)


if __name__ == "__main__":
    import doctest
    doctest.testmod()