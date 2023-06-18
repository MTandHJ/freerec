

from typing import List

import math
import numpy as np


__all__ = ['Identifier', 'Indexer', 'StandardScaler', 'MinMaxScaler']


class TransformError(Exception): ...


class Identifier:
    r"""
    Transform X into X identically.

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
    >>> transformer = Identifier()
    >>> transformer.transform(col)
    [3, 2, 1]
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self): ...

    def partial_fit(self, col: List): ...

    def transform(self, col: List) -> List:
        return col


class Indexer(Identifier):
    r"""
    Transform sparse items into indices.

    classes: set
        Set of unique classes seen during `fit`.
    mapper: dict
        Mapping of input values to output indices.

    Examples:
    ---------
    >>> col = [3, 2, 1]
    >>> col2 = [4, 5, 6]
    >>> transformer = Indexer()
    >>> transformer.partial_fit(col)
    >>> transformer.classes
    {1, 2, 3}
    >>> transformer.mapper
    {1: 0, 2: 1, 3: 2}
    >>> transformer.partial_fit(col2)
    >>> transformer.classes
    {1, 2, 3, 4, 5, 6}
    >>> transformer.mapper
    {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    >>> transformer.transform(col2)
    [3, 4, 5]
    """

    def reset(self):
        self.classes = set()
        self.mapper = None
        self.enums = None

    def partial_fit(self, col: List):
        self.classes |= set(col)
        ordered = sorted(self.classes)
        self.count = len(ordered)
        self.enums = tuple(range(self.count))
        self.mapper = dict(zip(ordered, self.enums))

    def _map(self, x):
        return self.mapper[x]

    def transform(self, col: List) -> List:
        if self.mapper:
            return list(map(self._map, col))
        else:
            raise TransformError("Indexer should be (partially) fitted before using ...")


class UpIndexer(Indexer):
    r"""
    Transform sparse items into indices from zero to maximum.

    Examples:
    ---------
    >>> col = [3, 2, 1]
    >>> col2 = [4, 5, 6]
    >>> transformer = UpIndexer()
    >>> transformer.partial_fit(col)
    >>> transformer.count
    4
    >>> transformer.enums
    (0, 1, 2, 3)
    >>> transformer.partial_fit(col2)
    >>> transformer.count
    7
    >>> transformer.enums
    (0, 1, 2, 3, 4, 5, 6)
    >>> transformer.transform(col2)
    [4, 5, 6]
    """

    def reset(self):
        self.enums = None
        self.count = float("-inf")

    def partial_fit(self, col: List):
        self.count = max(np.max(col).item() + 1, self.count)
        self.enums = tuple(range(self.count))

    def transform(self, col: List) -> List:
        if self.enums:
            return col
        else:
            raise TransformError("Indexer should be (partially) fitted before using ...")

class NumIndexer(Indexer):
    r"""
    Transform sparse items into indices from zero to maximum.

    Parameters:
    -----------
    nums: int
        the maximum index.

    Examples:
    ---------
    >>> transformer = NumIndexer(6)
    >>> transformer.count
    7
    >>> transformer.enums
    (0, 1, 2, 3, 4, 5, 6)
    """

    def __init__(self, nums: int) -> None:
        self.nums = nums
        super().__init__()

    def reset(self):
        self.count = self.nums + 1
        self.enums = tuple(range(self.count))

    def partial_fit(self, col: List): ...

    def transform(self, col: List) -> List:
        return col


class StandardScaler(Identifier):
    r"""
    Normalize dense items using the standard scaler.

    Attributes:
    -----------
    nums: int
        The number of fitted items.
    sum: float
        The summation of the fitted items.
    ssum: float 
        The summation of the squared fitted items.

    Examples:
    ---------
    >>> col = [3., 2., 1.]
    >>> transformer = StandardScaler()
    >>> transformer.partial_fit(col)
    >>> transformer.transform(col)
    [1.2247448713915887, 0.0, -1.2247448713915887]
    """

    def reset(self):
        self.nums = 0
        self.sum = 0
        self.ssum = 0 # sum of squared

    @property
    def mean(self):
        return self.sum / self.nums

    @property
    def std(self):
        return math.sqrt(self.ssum / self.nums - self.mean ** 2)

    def partial_fit(self, col: List):
        col = np.array(col)
        self.nums += len(col)
        self.sum += col.sum().item()
        self.ssum += (col ** 2).sum().item()

    def transform(self, col: List) -> List:
        if self.nums:
            return ((np.array(col) - self.mean) / self.std).tolist()
        else:
            raise TransformError("StandardScaler should be (partially) fitted before using ...")


class MinMaxScaler(Identifier):
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
    >>> transformer = MinMaxScaler()
    >>> transformer.partial_fit(col)
    >>> transformer.transform(col)
    [1.0, 0.5, 0.0]
    """

    def reset(self):
        self.min = float('inf')
        self.max = float('-inf')

    def partial_fit(self, col: List):
        self.min = min(np.min(col).item(), self.min)
        self.max = max(np.max(col).item(), self.max)

    def transform(self, col: List) -> List:
        if self.min < self.max:
            return ((np.array(col) - self.min) / (self.max - self.min)).tolist()
        else: # Same
            return col


if __name__ == "__main__":
    import doctest
    doctest.testmod()