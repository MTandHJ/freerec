

from typing import TypeVar, Literal, Optional, Union, Any, Iterator, Iterable, Callable, Dict, List

import torchdata.datapipes as dp
from torch.utils.data import DataChunk, default_collate

from ..datasets.base import RecDataSet
from ..fields import Field, FieldTuple
from .sampler import NUM_NEGS_FOR_SAMPLE_BASED_RANKING


__all__ = ['BaseProcessor', 'Postprocessor']


T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)


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

    @property
    def dataset(self) -> RecDataSet: ...

    @property
    def fields(self) -> FieldTuple[Field]: ...
 
    @staticmethod
    def listmap(func: Callable, *iterables) -> List[Any]:
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
        ...

    @classmethod
    def to_rows(cls, field_dict: Dict[Field, Iterable[T]]) -> List[Dict[Field, T]]:
        ...

    # Functional form of 'GenTrainPositiveSampler'
    def gen_train_sampling_pos_(self) -> BaseProcessor: ...

    # Functional form of 'GenTrainNegativeSampler'
    def gen_train_sampling_neg_(self, unseen_only: bool = True, num_negatives: int = 1) -> BaseProcessor: ...

    # Functional form of 'SeqTrainPositiveSampler'
    def seq_train_yielding_pos_(self, start_idx_for_target: Optional[int] = 1, end_idx_for_input: Optional[int] = -1) -> BaseProcessor: ...
        
    # Functional form of 'SeqTrainNegativeSampler'
    def seq_train_sampling_neg_(self, unseen_only: bool = True, num_negatives: int = 1) -> BaseProcessor: ...

    # Functional form of 'ValidSampler'
    def valid_sampling_(self, ranking: Literal['full', 'pool'] = 'full', num_negatives: int = NUM_NEGS_FOR_SAMPLE_BASED_RANKING) -> BaseProcessor: ...

    # Functional form of 'TestSampler'
    def test_sampling_(self, ranking: Literal['full', 'pool'] = 'full', num_negatives: int = NUM_NEGS_FOR_SAMPLE_BASED_RANKING) -> BaseProcessor: ...

    # Functional form of 'LeftPruningRow'
    def lprune_(self, maxlen: int, *, modified_fields: Iterable[Field]) -> BaseProcessor: ...

    # Functional form of 'RightPruningRow'
    def rprune_(self, maxlen: int, *, modified_fields: Iterable[Field]) -> BaseProcessor: ...

    # Functional form of 'AddingRow'
    def add_(self, offset: int, *, modified_fields: Iterable[Field]) -> BaseProcessor: ...

    # Functional form of 'LeftPaddingRow'
    def lpad_(self, maxlen: int, *, modified_fields: Iterable[Field], padding_value: int = 0) -> BaseProcessor: ...

    # Functional form of 'RightPaddingRow'
    def rpad_(self, maxlen: int, *, modified_fields: Iterable[Field], padding_value: int = 0) -> BaseProcessor: ...

    #========================================Functional forms from IterDataPipe========================================

    # Functional form of 'Batcher'
    def batch(self, batch_size: int, drop_last: bool = False, wrapper_class=DataChunk) -> BaseProcessor:
        r"""
        Creates mini-batches of data (functional name: ``batch``). An outer dimension will be added as
        ``batch_size`` if ``drop_last`` is set to ``True``, or ``length % batch_size`` for the
        last batch if ``drop_last`` is set to ``False``.
    
        Args:
            datapipe: Iterable DataPipe being batched
            batch_size: The size of each batch
            drop_last: Option to drop the last batch if it's not full
            wrapper_class: wrapper to apply onto each batch (type ``List``) before yielding,
                defaults to ``DataChunk``
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> dp = IterableWrapper(range(10))
            >>> dp = dp.batch(batch_size=3, drop_last=True)
            >>> list(dp)
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        """
    
    # Functional form of 'CollatorIterDataPipe'
    def collate(self, conversion: Optional[Union[Callable[..., Any],Dict[Union[str, Any], Union[Callable, Any]],]] = default_collate, collate_fn: Optional[Callable] = None) -> BaseProcessor:
        r"""
        Collates samples from DataPipe to Tensor(s) by a custom collate function (functional name: ``collate``).
        By default, it uses :func:`torch.utils.data.default_collate`.
    
        .. note::
            While writing a custom collate function, you can import :func:`torch.utils.data.default_collate` for the
            default behavior and `functools.partial` to specify any additional arguments.
    
        Args:
            datapipe: Iterable DataPipe being collated
            collate_fn: Customized collate function to collect and combine data or a batch of data.
                Default function collates to Tensor(s) based on data type.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> # Convert integer data to float Tensor
            >>> class MyIterDataPipe(torch.utils.data.IterDataPipe):
            ...     def __init__(self, start, end):
            ...         super(MyIterDataPipe).__init__()
            ...         assert end > start, "this example code only works with end >= start"
            ...         self.start = start
            ...         self.end = end
            ...
            ...     def __iter__(self):
            ...         return iter(range(self.start, self.end))
            ...
            ...     def __len__(self):
            ...         return self.end - self.start
            ...
            >>> ds = MyIterDataPipe(start=3, end=7)
            >>> print(list(ds))
            [3, 4, 5, 6]
            >>> def collate_fn(batch):
            ...     return torch.tensor(batch, dtype=torch.float)
            ...
            >>> collated_ds = CollateIterDataPipe(ds, collate_fn=collate_fn)
            >>> print(list(collated_ds))
            [tensor(3.), tensor(4.), tensor(5.), tensor(6.)]
        """

    # Functional form of 'MapperIterDataPipe'
    def map(self, fn: Callable, input_col=None, output_col=None) -> BaseProcessor:
        r"""
        Applies a function over each item from the source DataPipe (functional name: ``map``).
        The function can be any regular Python function or partial object. Lambda
        function is not recommended as it is not supported by pickle.
    
        Args:
            datapipe: Source Iterable DataPipe
            fn: Function being applied over each item
            input_col: Index or indices of data which ``fn`` is applied, such as:
    
                - ``None`` as default to apply ``fn`` to the data directly.
                - Integer(s) is used for list/tuple.
                - Key(s) is used for dict.
    
            output_col: Index of data where result of ``fn`` is placed. ``output_col`` can be specified
                only when ``input_col`` is not ``None``
    
                - ``None`` as default to replace the index that ``input_col`` specified; For ``input_col`` with
                  multiple indices, the left-most one is used, and other indices will be removed.
                - Integer is used for list/tuple. ``-1`` represents to append result at the end.
                - Key is used for dict. New key is acceptable.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
            >>> def add_one(x):
            ...     return x + 1
            >>> dp = IterableWrapper(range(10))
            >>> map_dp_1 = dp.map(add_one)  # Invocation via functional form is preferred
            >>> list(map_dp_1)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> # We discourage the usage of `lambda` functions as they are not serializable with `pickle`
            >>> # Use `functools.partial` or explicitly define the function instead
            >>> map_dp_2 = Mapper(dp, lambda x: x + 1)
            >>> list(map_dp_2)
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        """

    # Functional form of 'ShardingFilterIterDataPipe'
    def sharding_filter(self, sharding_group_filter=None) -> BaseProcessor:
        r"""
        Wrapper that allows DataPipe to be sharded (functional name: ``sharding_filter``). After ``apply_sharding`` is
        called, each instance of the DataPipe (on different workers) will have every `n`-th element of the
        original DataPipe, where `n` equals to the number of instances.
    
        Args:
            source_datapipe: Iterable DataPipe that will be sharded
        """

    # Functional form of 'UnBatcherIterDataPipe'
    def unbatch(self, unbatch_level: int = 1) -> BaseProcessor:
        r"""
        Undoes batching of data (functional name: ``unbatch``). In other words, it flattens the data up to the specified level
        within a batched DataPipe.
    
        Args:
            datapipe: Iterable DataPipe being un-batched
            unbatch_level: Defaults to ``1`` (only flattening the top level). If set to ``2``,
                it will flatten the top two levels, and ``-1`` will flatten the entire DataPipe.
    
        Example:
            >>> # xdoctest: +SKIP
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> source_dp = IterableWrapper([[[0, 1], [2]], [[3, 4], [5]], [[6]]])
            >>> dp1 = source_dp.unbatch()
            >>> list(dp1)
            [[0, 1], [2], [3, 4], [5], [6]]
            >>> dp2 = source_dp.unbatch(unbatch_level=2)
            >>> list(dp2)
            [0, 1, 2, 3, 4, 5, 6]
        """

    # Functional form of 'BatchMapperIterDataPipe'
    def map_batches(self, fn: Callable, batch_size: int, input_col=None) -> BaseProcessor:
        r"""
        Combines elements from the source DataPipe to batches and applies a function
        over each batch, then flattens the outputs to a single, unnested IterDataPipe
        (functional name: ``map_batches``).
    
        Args:
            datapipe: Source IterDataPipe
            fn: The function to be applied to each batch of data
            batch_size: The size of batch to be aggregated from ``datapipe``
            input_col: Index or indices of data which ``fn`` is applied, such as:
    
                - ``None`` as default to apply ``fn`` to the data directly.
                - Integer(s) is used for list/tuple.
                - Key(s) is used for dict.
    
        Example:
            >>> from torchdata.datapipes.iter import IterableWrapper
            >>> def fn(batch):
            >>>     return [d + 1 for d in batch]
            >>> source_dp = IterableWrapper(list(range(5)))
            >>> mapped_dp = source_dp.map_batches(fn, batch_size=3)
            >>> list(mapped_dp)
            [1, 2, 3, 4, 5]
    
        Notes:
            Compared with ``map``, the reason that ``map_batches`` doesn't take
            ``output_col`` argument is the size of ``fn`` output is not guaranteed
            to be the same as input batch. With different size, this operation cannot
            assign data back to original data structure.
    
            And, this operation is introduced based on the use case from `TorchText`.
            A pybinded C++ vectorized function can be applied for efficiency.
        """

    def __iter__(self) -> Iterator[Dict[Field, Any]]: ...

   
class Source(BaseProcessor):
    """Source datapipe. The start point of Train/valid/test datapipe"""

    dataset: RecDataSet

    def __init__(self, dataset: RecDataSet) -> None: ...

    def set_seed(self, seed: int) -> None: ...


class Postprocessor(BaseProcessor):
    r"""
    A post-processor that wraps another IterDataPipe object.

    Parameters:
    -----------
    source: BaseProcessor
        The data pipeline to be wrapped.
    """

    source: BaseProcessor
    
    def __init__(self, source: BaseProcessor) -> None: ...

    def sure_input_fields(self) -> List[Field]: ...