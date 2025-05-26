

from typing import TypeVar, Any, Iterator, Iterable, Callable, Dict, List, Optional, Sized

import torch, random
import torchdata.datapipes as dp
from torch.utils.data.graph_settings import get_all_graph_pipes
from torch.utils.data.datapipes.datapipe import IterDataPipe

from ..datasets.base import RecDataSet
from ..fields import Field, FieldTuple


__all__ = ['BaseProcessor', 'Source', 'PostProcessor', 'SampleMultiplexer']


T = TypeVar('T')


class Launcher(dp.iter.IterDataPipe):

    def __init__(self, datasize: int, shuffle: bool = True):
        super().__init__()

        self.source = list(range(datasize))
        self.shuffle = shuffle

        self._rng = random.Random()
        self.set_seed(0)

    def set_seed(self, seed: int):
        self._rng.seed(seed)

    def __iter__(self):
        if self.shuffle:
            self._rng.shuffle(self.source)
        yield from iter(self.source)


class BaseProcessor(dp.iter.IterDataPipe):
    r"""
    A base processor that defines the property of fields.

    Parameters:
    -----------
    fields: Field or Iterable, optional
        - `None': Pass.
        - `Field`: FieldTuple with one Field.
        - `Iterable`: FieldTuple with multi Fields
    
    Raises:
    -------
    AttributeError: 
        If `fields' are not given or `None` before using.
    """

    def __init__(self, dataset: RecDataSet) -> None:
        super().__init__()
        self.__dataset = dataset
        self.fields = dataset.fields

    @property
    def dataset(self):
        return self.__dataset

    @property
    def fields(self) -> FieldTuple:
        return self.__fields

    @fields.setter
    def fields(self, fields: Iterable[Field]):
        self.__fields = FieldTuple(fields)
 
    @staticmethod
    def listmap(func: Callable, *iterables):
        r"""
        Apply a function to multiple iterables and return a list.

        Parameters:
        -----------
        func (Callable): The function to be applied.
        *iterables: Multiple iterables to be processed.

        Returns:
        --------
        List: The results after applying the function to the iterables.
        """
        return list(map(func, *iterables))

    @classmethod
    def to_rows(cls, field_dict: Dict[Field, Iterable[T]]) -> List[Dict[Field, T]]:
        fields = field_dict.keys()
        return cls.listmap(
            lambda values: dict(zip(fields, values)),
            zip(*field_dict.values())
        )
    

class Source(BaseProcessor):
    """Source datapipe. The start point of Train/valid/test datapipe"""

    def __init__(
        self, 
        dataset: RecDataSet, source: Iterable[Dict[Field, Any]],
        datasize: Optional[int] = None,
        shuffle: bool = True
    ) -> None:
        super().__init__(dataset)
        self.mode = dataset.mode
        if isinstance(source, dp.iter.IterDataPipe):
            self.source = source
            self.launcher = source.sharding_filter()
        else:
            self.source = tuple(source)
            self.datasize = len(self.source) if datasize is None else datasize
            self.launcher = Launcher(self.datasize, shuffle=shuffle).sharding_filter()
    
    def guard_mode(self):
        r"""
        Make sure the dataset is at a required mode.
        This is especially necessary for datapipe source.
        """
        getattr(self.dataset, self.mode)()

    def __getstate__(self):
        # `traverse_dps' will be particularly time-consuming
        # if a lot of data is buffered.
        # Hence, we directly return the connected datapipes.
        state = self.__dict__
        if IterDataPipe.getstate_hook is not None:
            return self.launcher
        return state


class PostProcessor(BaseProcessor):
    r"""
    A post-processor that wraps another IterDataPipe object.

    Parameters:
    -----------
    source: BaseProcessor
        The data pipeline to be wrapped.
    """

    def __init__(self, source: BaseProcessor) -> None:
        graph = torch.utils.data.graph.traverse_dps(source)
        dataset = None
        for pipe in get_all_graph_pipes(graph):
            if isinstance(pipe, BaseProcessor):
                dataset = pipe.dataset
                break
        assert dataset is not None, "Make sure datapipe starts from a BaseProcessor ..."
        super().__init__(dataset)
        self.source: Iterator[Dict[Field, Any]] = source

    def __getstate__(self):
        # `traverse_dps' will be particularly time-consuming
        # if a lot of data is buffered.
        # Hence, we directly return the connected datapipes.
        state = self.__dict__
        if IterDataPipe.getstate_hook is not None:
            return self.source
        return state


class SampleMultiplexer(IterDataPipe):
    r"""
    Takes a `Dict` of (IterDataPipe, Weight), and yields items by sampling from these
    DataPipes with respect to their weights. When individual DataPipes are exhausted, continues to sample from
    the remaining DataPipes according to their relative weights.
    If you wish to maintain the same ratio of weights indefinitely, you need to ensure that the
    inputs are never exhausted, by, for instance, applying ``cycle`` to them.

    Parameters:
    -----------
    pipes_to_weights_dict: a `Dict` of IterDataPipes and Weights. The total weight of
        unexhausted DataPipes will be normalized to 1 for the purpose of sampling.

    Examples:
    ---------
    >>> source_dp1 = IterableWrapper([0] * 10)
    >>> source_dp2 = IterableWrapper([1] * 10)
    >>> d = {source_dp1: 99999999, source_dp2: 0.0000001}
    >>> sample_mul_dp = SampleMultiplexer(pipes_to_weights_dict=d)
    >>> list(sample_mul_dp)
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    """

    def __init__(
        self,
        pipes_to_weights_dict: Dict[IterDataPipe, float]
    ):
        if not pipes_to_weights_dict:
            raise ValueError("Empty dictionary passed to SampleMultiplexerDataPipe")
        total_weight: float = 0
        for v in pipes_to_weights_dict.values():
            if v <= 0:
                raise ValueError(f"Expecting a positive and non-zero weight, got {v}")
            total_weight += v

        self.pipes_and_weights = tuple([(k, v / total_weight) for k, v in pipes_to_weights_dict.items()])

        self._rng = random.Random()
        self.set_seed(0)

    def set_seed(self, seed: int):
        self._rng.seed(seed)

    def __iter__(self) -> Iterator:
        pipes_and_weights = [(iter(k), v) for k, v in self.pipes_and_weights]
        while len(pipes_and_weights) > 1:
            r = self._rng.random()
            s: float = 0
            for it, weight in pipes_and_weights:
                s += weight
                if r < s:
                    try:
                        item = next(it)
                        yield item
                    except StopIteration:
                        # remove the current stream
                        new_total = 1 - weight
                        assert new_total > 0
                        pipes_and_weights = [(k, v / new_total) for k, v in pipes_and_weights if k != it]
                    break

        # only one stream left
        for item in pipes_and_weights[0][0]:
            yield item

    def __len__(self) -> int:
        if all(isinstance(dp, Sized) for dp, _ in self.pipes_and_weights):
            return sum(len(dp) for dp, _ in self.pipes_and_weights)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")