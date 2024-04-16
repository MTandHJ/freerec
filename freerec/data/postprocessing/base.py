

from typing import TypeVar, Any, Iterator, Iterable, Callable, Dict, List

import torch, random
import torchdata.datapipes as dp
from torch.utils.data.graph_settings import get_all_graph_pipes

from ..datasets.base import RecDataSet
from ..fields import Field, FieldTuple


__all__ = ['BaseProcessor', 'Postprocessor']


T = TypeVar('T')


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

    def __init__(self, dataset: RecDataSet) -> None:
        super().__init__(dataset)
        self._rng = random.Random()
        self.set_seed(0)

    def set_seed(self, seed: int):
        self._rng.seed(seed)


class Postprocessor(BaseProcessor):
    r"""
    A post-processor that wraps another IterDataPipe object.

    Parameters:
    -----------
    source: BaseProcessor
        The data pipeline to be wrapped.
    """

    def __init__(self, source: BaseProcessor) -> None:
        graph = torch.utils.data.graph.traverse_dps(source)
        for pipe in get_all_graph_pipes(graph):
            if isinstance(pipe, RecDataSet):
                dataset = pipe
                break
        super().__init__(dataset)
        self.source: Iterator[Dict[Field, Any]] = source

    def sure_input_fields(self) -> List[Field]:
        return list(next(iter(self.source)).keys())


