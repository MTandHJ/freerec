

from typing import List

import math
import numpy as np

__all__ = ['Identifier', 'Indexer', 'StandardScaler', 'MinMaxScaler']


class TransformError(Exception): ...

class Identifier:
    """Transform X into X identically.

    Examples:
    ---

    >>> import torcharrow.dtypes as dt
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
    """Transform sparse items into indices.

    Attributes:
    ---

    classes: set
        the classes therein
    maper: dict
        map X to Y

    Examples:
    ---

    >>> col = [3, 2, 1]
    >>> col2 = [4, 5, 6]
    >>> transformer = Indexer()
    >>> transformer.partial_fit(col)
    >>> transformer.classes
    {1, 2, 3}
    >>> transformer.maper
    {1: 0, 2: 1, 3: 2}
    >>> transformer.partial_fit(col2)
    >>> transformer.classes
    {1, 2, 3, 4, 5, 6}
    >>> transformer.maper
    {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    >>> transformer.transform(col2)
    [3, 4, 5]
    """

    def reset(self):
        self.classes = set()
        self.maper = None
        self.ids = None

    def partial_fit(self, col: List):
        self.classes |= set(col)
        ordered = sorted(self.classes)
        self.count = len(ordered)
        self.ids = tuple(range(self.count))
        self.maper = dict(zip(ordered, self.ids))

    def _map(self, x):
        return self.maper[x]

    def transform(self, col: List) -> List:
        if self.maper:
            return list(map(self._map, col))
        else:
            raise TransformError("Indexer should be (partially) fitted before using ...")


class StandardScaler(Identifier):
    """Normalize dense items.

    Attributes:
    ---

    field: `dt.Field` or the `Field` defined in fields
    nums: int
        the number of fitted items
    sum: float
        the summation of the fitted items
    ssum: float 
        the summation of the suqared items

    Examples:
    ---

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
    """Scale to [0, 1].

    Attributes:
    ---

    min: float
        the minimum
    max: float
        the maximum

    Examples:
    ---

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
   