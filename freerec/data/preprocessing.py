

from typing import List

import math
import numpy as np

__all__ = ['Identifier', 'Indexer', 'StandardScaler', 'MinMaxScaler']


class TransformError(Exception): ...

class Identifier:
    """
    Transform X into X identically.

    Parameters:
    ----------
    X : Any
        The input data to be transformed.

    Returns:
    -------
    Any
        The transformed data.

    Examples:
    --------
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
    """
    Transform sparse items into indices.

    classes: set
        Set of unique classes seen during `fit`.
    maper: dict
        Mapping of input values to output indices.

    Examples:
    ---------
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
    """
    Normalize dense items using the standard scaler.

    Attributes:
    -----------
    field: dt.Field or the Field defined in torcharrow.fields
        The field type of the dense items.
    n_samples: int
        The number of fitted items.
    sum_: float
        The summation of the fitted items.
    sum_of_squares: float 
        The summation of the squared fitted items.

    Examples:
    ---------
    >>> import torcharrow.dtypes as dt
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
    """
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