

from typing import TypeVar, Any, Literal, Optional, Union, Iterator, Iterable, Callable, Dict, List

import torchdata.datapipes as dp
from torch.utils.data import DataChunk, default_collate

from .sampler import NUM_NEGS_FOR_SAMPLE_BASED_RANKING
from ..datasets.base import RecDataSet
from ..fields import Field, FieldTuple


__all__ = ['BaseProcessor', 'Postprocessor']


T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)


class Launcher(dp.iter.IterDataPipe):

    def __init__(self, datasize: int, shuffle: bool = True): ...

    def set_seed(self, seed: int) -> None: ...


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

    def __init__(self, dataset: RecDataSet) -> None: ...

    @property
    def dataset(self) -> RecDataSet:
        return self.__dataset

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
    def gen_train_sampling_pos_(self: T) -> T:
        r"""
        Sampling a positive item for each user.

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.train().choiced_user_ids_source().gen_train_sampling_pos_()
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 12623, Field(ITEM:ID,ITEM,POSITIVE): 6467}
        """

    # Functional form of 'GenTrainNegativeSampler'
    def gen_train_sampling_neg_(self: T, num_negatives: int = 1, unseen_only: bool = True) -> T:
        r"""
        Sampling negatives for each user.

        Parameters:
        -----------
        num_negatives: int, default to 1
            The number of negatives for each row.
        unseen_only: bool, default to `True`
            `True`: sampling negatives from the unseen.
            `False`: sampling negatives from all items.
        nums_need_vectorized_bsearch: int, default to 10
            The number negatives suitable for using vectorized bsearch.

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.train().choiced_user_ids_source(
        ).gen_train_sampling_pos_(
        ).gen_train_sampling_neg(
            num_negatives=2
        )
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 12623,
        Field(ITEM:ID,ITEM,POSITIVE): 6471,
        Field(ITEM:ID,ITEM,NEGATIVE): [7415, 2353]}
        """

    # Functional form of 'SeqTrainPositiveSampler'
    def seq_train_yielding_pos_(self: T, start_idx_for_target: Optional[int] = 1, end_idx_for_input: Optional[int] = -1) -> T:
        r"""
        Yielding positive sequence for each user sequence.

        Parameters:
        -----------
        start_idx_for_target: int, optional
            Target sequence as seq[start_idx_for_target:]
            `None`: seq
        end_idx_for_input: int, optional
            Input sequence as seq[:end_idx_for_input]
            `None`: seq

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.train().shuffled_seqs_source(
            maxlen=10
        ).seq_train_yielding_pos_(
            1, -1
        )
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 21853,
        Field(ITEM:ID,ITEM,SEQUENCE): (3562, 9621, 9989),
        Field(ITEM:ID,ITEM,POSITIVE): (9621, 9989, 10579)}
        >>> datapipe = dataset.train().shuffled_seqs_source(
            maxlen=10
        ).seq_train_yielding_pos_(
            None, None
        )
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 21853,
        Field(ITEM:ID,ITEM,SEQUENCE): (3562, 9621, 9989, 10579),
        Field(ITEM:ID,ITEM,POSITIVE): (3562, 9621, 9989, 10579)}
        """
        
    # Functional form of 'SeqTrainNegativeSampler'
    def seq_train_sampling_neg_(self: T, num_negatives: int = 1, unseen_only: bool = True) -> T:
        r"""
        Sampling negatives for each positive.

        Parameters:
        -----------
        num_negatives: int, default to 1
            The number of negatives for each row.
        unseen_only: bool, default to `True`
            `True`: sampling negatives from the unseen.
            `False`: sampling negatives from all items.
        nums_need_vectorized_bsearch: int, default to 10
            The number negatives suitable for using vectorized bsearch.

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.train().shuffled_seqs_source(
            maxlen=10
        ).seq_train_yielding_pos_(
        ).seq_train_sampling_neg_(
            num_negatives=2
        )
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 21853,
        Field(ITEM:ID,ITEM,SEQUENCE): (3562, 9621, 9989),
        Field(ITEM:ID,ITEM,POSITIVE): (9621, 9989, 10579),
        Field(ITEM:ID,ITEM,NEGATIVE): [[4263, 5582], [1439, 1800], [7969, 9149]]}
        """

    # Functional form of 'ValidSampler'
    def valid_sampling_(self: T, ranking: Literal['full', 'pool'] = 'full', num_negatives: int = NUM_NEGS_FOR_SAMPLE_BASED_RANKING) -> T:
        r"""
        Sampler for validation.

        Parameters:
        -----------
        ranking: 'full' or 'pool', default to 'full'
            'full': full ranking
            'pool': sampled-based ranking
        num_negatives: int, default to 100
            The number of negatives for 'pool'.
        
        Yields:
        -------
        Field(USER:ID,USER): user id
        Field(ITEM:ID,ITEM,SEQUENCE): user sequence
        Field(ITEM:ID,ITEM,UNSEEN):
            'full': target items
            'pool': target items + negatives items
        Field(ITEM:ID,ITEM,SEEN): seen items
        
        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.valid().ordered_user_ids_source(
        ).valid_sampling_(ranking='full')
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 0,
        Field(ITEM:ID,ITEM,SEQUENCE): (9449, 9839, 10076, 11155),
        Field(ITEM:ID,ITEM,UNSEEN): (11752,),
        Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
        >>> datapipe = dataset.valid().ordered_user_ids_source(
        ).valid_sampling_(ranking='pool', num_negatives=5)
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 0,
        Field(ITEM:ID,ITEM,SEQUENCE): (9449, 9839, 10076, 11155),
        Field(ITEM:ID,ITEM,UNSEEN): (11752, 7021, 11954, 1052, 11116, 10916),
        Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
        """

    # Functional form of 'TestSampler'
    def test_sampling_(self: T, ranking: Literal['full', 'pool'] = 'full', num_negatives: int = NUM_NEGS_FOR_SAMPLE_BASED_RANKING) -> T:
        r"""
        Sampler for test.

        Parameters:
        -----------
        ranking: 'full' or 'pool', default to 'full'
            'full': full ranking
            'pool': sampled-based ranking
        num_negatives: int, default to 100
            The number of negatives for 'pool'.
        
        Yields:
        -------
        Field(USER:ID,USER): user id
        Field(ITEM:ID,ITEM,SEQUENCE): user sequence
        Field(ITEM:ID,ITEM,UNSEEN):
            'full': target items
            'pool': target items + negatives items
        Field(ITEM:ID,ITEM,SEEN): seen items
        
        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.test().ordered_user_ids_source(
        ).valid_sampling_(ranking='full')
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 0,
        Field(ITEM:ID,ITEM,SEQUENCE): (9449, 9839, 10076, 11155),
        Field(ITEM:ID,ITEM,UNSEEN): (11752,),
        Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
        >>> datapipe = dataset.test().ordered_user_ids_source(
        ).valid_sampling_(ranking='pool', num_negatives=5)
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 0,
        Field(ITEM:ID,ITEM,SEQUENCE): (9449, 9839, 10076, 11155),
        Field(ITEM:ID,ITEM,UNSEEN): (11752, 10413, 9774, 487, 4114, 10546),
        Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
        """

    # Functional form of 'LeftPruningRow'
    def lprune_(self: T, maxlen: int, modified_fields: Iterable[Field]) -> T:
        r"""
        A functional datapipe that prunes the left side of a given datapipe to a specified maximum length.

        Parameters:
        -----------
        maxlen: int 
            The maximum length to prune the input data to.
        modifields_fields: Iterable[Field]
            The fields to be modified.

        Flows:
        ------
        [1, 2, 3, 4] --(maxlen=3)--> [2, 3, 4]
        [3, 4] --(maxlen=3)--> [3, 4]
        
        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> ISeq = dataset[ITEM, ID].fork(SEQUENCE)
        >>> datapipe = dataset.valid().ordered_user_ids_source(
        ).valid_sampling_().lprune_(
            3, modified_fields=(ISeq,)
        )
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 0,
        Field(ITEM:ID,ITEM,SEQUENCE): (9839, 10076, 11155),
        Field(ITEM:ID,ITEM,UNSEEN): (11752,),
        Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
        """

    # Functional form of 'RightPruningRow'
    def rprune_(self: T, maxlen: int, modified_fields: Iterable[Field]) -> T:
        r"""
        A functional datapipe that prunes the right side of a given datapipe to a specified maximum length.

        Parameters:
        -----------
        maxlen: int 
            The maximum length to prune the input data to.
        modifields_fields: Iterable[Field]
            The fields to be modified.

        Flows:
        ------
        [1, 2, 3, 4] --(maxlen=3)--> [1, 2, 3]
        [3, 4] --(maxlen=3)--> [3, 4]

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> ISeq = dataset[ITEM, ID].fork(SEQUENCE)
        >>> datapipe = dataset.valid().ordered_user_ids_source(
        ).valid_sampling_().rprune_(
            3, modified_fields=(ISeq,)
        )
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 0,
        Field(ITEM:ID,ITEM,SEQUENCE): (9449, 9839, 10076),
        Field(ITEM:ID,ITEM,UNSEEN): (11752,),
        Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
        """

    # Functional form of 'AddingRow'
    def add_(self: T, offset: int, modified_fields: Iterable[Field]) -> T:
        r"""
        Mapper that adds the input data by a specified offset.

        Parameters:
        -----------
        offset: int
            Amount to add the input data by.   
        modifields_fields: Iterable[Field]
            The fields to be modified.

        Flows:
        ------
        [1, 2, 3, 4] --(offset=1)--> [2, 3, 4, 5]

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> ISeq = dataset[ITEM, ID].fork(SEQUENCE)
        >>> datapipe = dataset.valid().ordered_user_ids_source(
        ).valid_sampling_().lprune_(
            3, modified_fields=(ISeq,)
        ).add_(
            1, modified_fields=(ISeq,)
        )
        """

    # Functional form of 'LeftPaddingRow'
    def lpad_(self: T, maxlen: int, modified_fields: Iterable[Field], padding_value: int = 0) -> T:
        r"""
        A functional data pipeline component that left pads sequences to a maximum length.

        Parameters:
        -----------
        maxlen : int
            The maximum length to pad the sequences to.
        modifields_fields: Iterable[Field]
            The fields to be modified.
        padding_value : int, optional (default=0)
            The value to use for padding.

        Flows:
        ------
        [1, 2, 3, 4] --(maxlen=7, padding_value=0)--> [0, 0, 0, 1, 2, 3, 4]
        [1, 2, 3, 4] --(maxlen=4, padding_value=0)--> [1, 2, 3, 4]

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> ISeq = dataset[ITEM, ID].fork(SEQUENCE)
        >>> datapipe = dataset.valid().ordered_user_ids_source(
        ).valid_sampling_().lprune_(
            3, modified_fields=(ISeq,)
        ).add_(
            1, modified_fields=(ISeq,)
        ).lpad_(
            5, modified_fields=(ISeq,)
        )
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 0,
        Field(ITEM:ID,ITEM,SEQUENCE): [0, 0, 9840, 10077, 11156],
        Field(ITEM:ID,ITEM,UNSEEN): (11752,),
        Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
        """

    # Functional form of 'RightPaddingRow'
    def rpad_(self: T, maxlen: int, modified_fields: Iterable[Field], padding_value: int = 0) -> T:
        r"""
        A functional data pipeline component that right pads sequences to a maximum length.

        Parameters:
        -----------
        maxlen : int
            The maximum length to pad the sequences to.
        modifields_fields: Iterable[Field]
            The fields to be modified.
        padding_value : int, optional (default=0)
            The value to use for padding.

        Flows:
        ------
        [1, 2, 3, 4] --(maxlen=7, padding_value=0)--> [1, 2, 3, 4, 0, 0, 0]
        [1, 2, 3, 4] --(maxlen=4, padding_value=0)--> [1, 2, 3, 4]

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> ISeq = dataset[ITEM, ID].fork(SEQUENCE)
        >>> datapipe = dataset.valid().ordered_user_ids_source(
        ).valid_sampling_().rprune_(
            3, modified_fields=(ISeq,)
        ).add_(
            1, modified_fields=(ISeq,)
        ).rpad_(
            5, modified_fields=(ISeq,)
        )
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 0,
        Field(ITEM:ID,ITEM,SEQUENCE): [9450, 9840, 10077, 0, 0],
        Field(ITEM:ID,ITEM,UNSEEN): (11752,),
        Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
        """

    # Functional form of 'Batcher_'
    def batch_(self: T, batch_size: int, drop_last: bool = False) -> T:
        r"""
        A postprocessor that converts a batch of rows into:
            Dict[Field, List[Any]]

        Parameters:
        -----------
        source: dp.IterDataPipe 
            A datapipe that yields a batch samples.
        batch_size: int
        drop_last: bool, default False
        """

    # Functional form of 'ToTensor'
    def tensor_(self: T) -> T:
        r"""
        A datapipe that converts lists into torch Tensors.
        This class converts a List into a torch.Tensor.

        Notes:
        ------
        The returned tensor is at least 2d. 
        """

    #========================================Functional forms from IterDataPipe========================================

    # Functional form of 'Batcher'
    def batch(self: T, batch_size: int, drop_last: bool = False, wrapper_class=DataChunk) -> T:
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
    def collate(self: T, conversion: Optional[Union[Callable[..., Any],Dict[Union[str, Any], Union[Callable, Any]],]] = default_collate, collate_fn: Optional[Callable] = None) -> T:
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
    def map(self: T, fn: Callable, input_col=None, output_col=None) -> T:
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
    def sharding_filter(self: T, sharding_group_filter=None) -> T:
        r"""
        Wrapper that allows DataPipe to be sharded (functional name: ``sharding_filter``). After ``apply_sharding`` is
        called, each instance of the DataPipe (on different workers) will have every `n`-th element of the
        original DataPipe, where `n` equals to the number of instances.
    
        Args:
            source_datapipe: Iterable DataPipe that will be sharded
        """

    # Functional form of 'UnBatcherIterDataPipe'
    def unbatch(self: T, unbatch_level: int = 1) -> T:
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
    def map_batches(self: T, fn: Callable, batch_size: int, input_col=None) -> T:
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
    source: Iterable

    def __init__(
        self, dataset: RecDataSet, source: Iterable, 
        datasize: Optional[int] = None, shuffle: bool = True
    ) -> None:
        self.source: Iterable[Dict[Field, Any]]
        self.datasize: int
        self.lanucher: Launcher


class Postprocessor(BaseProcessor):
    r"""
    A post-processor that wraps another IterDataPipe object.

    Parameters:
    -----------
    source: BaseProcessor
        The data pipeline to be wrapped.
    """

    source: BaseProcessor
    
    def __init__(self, source: BaseProcessor) -> None:
        self.source: Iterator[Dict[Field, Any]]

    def sure_input_fields(self) -> List[Field]: ...