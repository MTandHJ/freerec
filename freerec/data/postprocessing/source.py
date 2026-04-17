import random
from typing import Any, Dict, Iterable

import torchdata.datapipes as dp

from freerec.data.datasets.base import RecDataSet
from freerec.data.fields import Field
from freerec.data.postprocessing.base import Source

__all__ = [
    "RandomChoicedSource",
    "RandomShuffledSource",
    "OrderedSource",
    "PipedSource",
]


@dp.functional_datapipe("choiced_source_")
class RandomChoicedSource(Source):
    r"""Datapipe that generates random items from a given source with replacement.

    Each iteration samples ``dataset.datasize`` items uniformly at random
    (with replacement) from the stored source rows.

    Parameters
    ----------
    dataset : :class:`~RecDataSet`
        The recommendation dataset.
    source : iterable of dict
        The source data rows to sample from.
    """

    def __init__(self, dataset: RecDataSet, source: Iterable[Dict[Field, Any]]) -> None:
        r"""Initialize the RandomChoicedSource."""
        super().__init__(dataset, source, dataset.datasize, shuffle=False)

        self._rng = random.Random()
        self.set_seed(0)

    def set_seed(self, seed: int):
        r"""Set the random seed for sampling.

        Parameters
        ----------
        seed : int
            Random seed value.
        """
        self._rng.seed(seed)

    def __iter__(self):
        r"""Yield randomly chosen rows from the source."""
        self.guard_mode()
        for _ in self.launcher:
            yield self._rng.choice(self.source).copy()


@dp.functional_datapipe("shuffled_source_")
class RandomShuffledSource(Source):
    r"""Datapipe that yields every source row exactly once per epoch in shuffled order.

    Parameters
    ----------
    dataset : :class:`~RecDataSet`
        The recommendation dataset.
    source : iterable of dict
        The source data rows.
    """

    def __init__(self, dataset: RecDataSet, source: Iterable[Dict[Field, Any]]) -> None:
        r"""Initialize the RandomShuffledSource."""
        super().__init__(dataset, source, shuffle=True)

    def __iter__(self):
        r"""Yield source rows in shuffled order."""
        self.guard_mode()
        for i in self.launcher:
            yield self.source[i].copy()


@dp.functional_datapipe("ordered_source_")
class OrderedSource(Source):
    r"""Datapipe that yields source rows in their original order.

    Parameters
    ----------
    dataset : :class:`~RecDataSet`
        The recommendation dataset.
    source : iterable of dict
        The source data rows.
    """

    def __init__(self, dataset: RecDataSet, source: Iterable[Dict[Field, Any]]) -> None:
        r"""Initialize the OrderedSource."""
        super().__init__(dataset, source, shuffle=False)

    def __iter__(self):
        r"""Yield source rows in sequential order."""
        self.guard_mode()
        for i in self.launcher:
            yield self.source[i].copy()


@dp.functional_datapipe("piped_source_")
class PipedSource(Source):
    r"""Datapipe that yields rows directly from an upstream :class:`~IterDataPipe`.

    Parameters
    ----------
    dataset : :class:`~RecDataSet`
        The recommendation dataset.
    source : :class:`~IterDataPipe`
        An upstream iterable datapipe.
    """

    def __init__(self, dataset: RecDataSet, source: dp.iter.IterDataPipe) -> None:
        r"""Initialize the PipedSource."""
        super().__init__(dataset, source)
        assert isinstance(source, dp.iter.IterDataPipe), (
            f"PipedSource needs `IterDataPipe` but {type(source)} received ..."
        )

    def __iter__(self):
        r"""Yield rows from the upstream datapipe."""
        self.guard_mode()
        for row in self.launcher:
            yield row
