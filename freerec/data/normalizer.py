


from typing import Optional

import math
import polars as pl


__all__ = ['Normalizer', 'Counter', 'ReIndexer', 'StandardScaler', 'MinMaxScaler']


NORMALIZERS = dict()

def register_normalizer(normalizer: 'Normalizer', name: Optional[str] = None):
    r"""Register a normalizer class in the global registry.

    Parameters
    ----------
    normalizer : :class:`Normalizer`
        The normalizer class to register. Must implement ``partial_fit``
        and ``normalize`` methods.
    name : str, optional
        The name to register the normalizer under. If ``None``, the class
        name is used.

    Returns
    -------
    :class:`Normalizer`
        The registered normalizer class, unchanged.

    Raises
    ------
    AssertionError
        If the normalizer does not have ``partial_fit`` or ``normalize``
        methods.
    """
    name = normalizer.__name__ if name is None else name
    assert hasattr(normalizer, 'partial_fit'), f"`partial_fit` method is not in `{name}`"
    assert hasattr(normalizer, 'normalize'), f"`normalize` method is not in `{name}`"
    NORMALIZERS[name.upper()] = normalizer
    return normalizer


class Normalizer:
    r"""Identity normalizer that returns data unchanged.

    This is the base class for all normalizers. Subclasses should override
    ``partial_fit`` and ``normalize`` to implement specific normalization
    logic.

    Examples
    --------
    >>> col = pl.Series([3, 2, 1])
    >>> normalizer = Normalizer()
    >>> normalizer.normalize(col).to_list()
    [3, 2, 1]
    """

    def __init__(self) -> None:
        r"""Initialize and reset the normalizer."""
        self.reset()

    def reset(self):
        r"""Reset the normalizer to its initial state."""
        self.fitting = True

    @property
    def count(self):
        r"""Return the number of unique old values."""
        return len(self.olds)

    def partial_fit(self, data: pl.Series):
        r"""Incrementally fit the normalizer on a batch of data.

        Parameters
        ----------
        data : :class:`polars.Series`
            A data batch to fit on.
        """
        self.fitting = True

    def normalize(self, data: pl.Series) -> pl.Series:
        r"""Normalize the given data.

        Parameters
        ----------
        data : :class:`polars.Series`
            The data to normalize.

        Returns
        -------
        :class:`polars.Series`
            The normalized data.
        """
        self.fitting = False
        return data

    def __call__(self, data: pl.Series) -> pl.Series:
        r"""Call ``normalize`` on the given data."""
        return self.normalize(data)


@register_normalizer
class Counter(Normalizer):
    r"""Normalizer that counts unique non-float values.

    Collects unique values seen during ``partial_fit`` but returns
    the data unchanged from ``normalize``.
    """

    def reset(self):
        r"""Reset the counter to an empty state."""
        super().reset()
        self.olds = []

    @property
    def count(self):
        r"""Return the number of unique values seen so far."""
        return len(self.olds)

    def partial_fit(self, data: pl.Series):
        r"""Accumulate unique values from a data batch.

        Parameters
        ----------
        data : :class:`polars.Series`
            A data batch. Only non-float series are counted.
        """
        self.fitting = True
        if not data.dtype.is_float():
            self.olds = set(self.olds) | set(data.unique())

    def normalize(self, data: pl.Series) -> pl.Series:
        r"""Return the data unchanged.

        Parameters
        ----------
        data : :class:`polars.Series`
            The data to normalize.

        Returns
        -------
        :class:`polars.Series`
            The original data, unmodified.
        """
        self.fitting = False
        return data


@register_normalizer
class ReIndexer(Counter):
    r"""Re-index categorical features into contiguous integer tokens.

    Unique values accumulated via ``partial_fit`` are sorted and mapped
    to ``0, 1, ..., count - 1`` upon the first call to ``normalize``.

    Attributes
    ----------
    olds : list or set
        Unique classes seen during fitting.
    news : list
        Contiguous integer indices assigned after normalization.

    Examples
    --------
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
        r"""Map categorical values to contiguous integer indices.

        Parameters
        ----------
        data : :class:`polars.Series`
            The categorical data to re-index.

        Returns
        -------
        :class:`polars.Series`
            The re-indexed data with integer tokens.
        """
        if self.fitting:
            self.olds = sorted(self.olds)
            self.news = list(range(self.count))
            self.fitting = False
        return data.replace_strict(old=self.olds, new=self.news)


@register_normalizer
class StandardScaler(Normalizer):
    r"""Normalize numerical features to zero mean and unit variance.

    Incrementally accumulates count, sum, and squared sum during
    ``partial_fit``, then applies the standard scaling formula
    ``(x - mean) / (std + eps)`` in ``normalize``.

    Attributes
    ----------
    nums : int
        The number of fitted values.
    sum : float
        The running sum of fitted values.
    ssum : float
        The running sum of squared fitted values.
    eps : float
        Small constant to avoid division by zero.

    Examples
    --------
    >>> col = pl.Series([3., 2., 1.])
    >>> normalizer = StandardScaler()
    >>> normalizer.partial_fit(col)
    >>> normalizer.normalize(col).to_list()
    [0.9999999900000002, 0.0, -0.9999999900000002]
    """

    def reset(self):
        r"""Reset all accumulators to zero."""
        super().reset()
        self.nums = 0
        self.sum = 0
        self.ssum = 0 # sum of squared
        self.eps = 1.e-8

    @property
    def mean(self):
        r"""Return the running mean of fitted values."""
        return self.sum / self.nums

    @property
    def std(self):
        r"""Return the sample standard deviation of fitted values."""
        return math.sqrt(
            (self.ssum - self.sum ** 2 / self.nums) / (self.nums - 1)
        )

    def partial_fit(self, data: pl.Series):
        r"""Accumulate statistics from a data batch.

        Parameters
        ----------
        data : :class:`polars.Series`
            A numerical data batch.
        """
        self.fitting = True
        self.nums += data.len()
        self.sum += data.sum()
        self.ssum += data.pow(2).sum()

    def normalize(self, data: pl.Series) -> pl.Series:
        r"""Apply standard scaling to the data.

        Parameters
        ----------
        data : :class:`polars.Series`
            The numerical data to normalize.

        Returns
        -------
        :class:`polars.Series`
            The standard-scaled data.
        """
        self.fitting = False
        return (data - self.mean) / (self.std + self.eps)


@register_normalizer
class MinMaxScaler(Normalizer):
    r"""Scale numerical features to the range [0, 1].

    Tracks the running minimum and maximum during ``partial_fit``,
    then applies ``(x - min) / (max - min + eps)`` in ``normalize``.

    Attributes
    ----------
    min : float
        The minimum value seen during fitting.
    max : float
        The maximum value seen during fitting.
    eps : float
        Small constant to avoid division by zero.

    Examples
    --------
    >>> col = pl.Series([3., 2., 1.])
    >>> normalizer = MinMaxScaler()
    >>> normalizer.partial_fit(col)
    >>> normalizer.normalize(col).to_list()
    [0.999999995, 0.4999999975, 0.0]
    """

    def reset(self):
        r"""Reset min and max to initial sentinel values."""
        super().reset()
        self.min = float('inf')
        self.max = float('-inf')
        self.eps = 1.e-8

    def partial_fit(self, data: pl.Series):
        r"""Update running min and max from a data batch.

        Parameters
        ----------
        data : :class:`polars.Series`
            A numerical data batch.
        """
        self.fitting = True
        self.min = min(data.min(), self.min)
        self.max = max(data.max(), self.max)

    def normalize(self, data: pl.Series) -> pl.Series:
        r"""Apply min-max scaling to the data.

        Parameters
        ----------
        data : :class:`polars.Series`
            The numerical data to normalize.

        Returns
        -------
        :class:`polars.Series`
            The scaled data in range [0, 1].
        """
        self.fitting = False
        return (data - self.min) / (self.max - self.min + self.eps)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
