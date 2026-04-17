from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import torchdata.datapipes as dp
from torch.utils.data import DataChunk, default_collate

from ..datasets.base import RecDataSet
from ..fields import Field, FieldTuple
from .sampler import NUM_NEGS_FOR_SAMPLE_BASED_RANKING

__all__ = ["BaseProcessor", "PostProcessor"]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

class Launcher(dp.iter.IterDataPipe):
    r"""An internal datapipe that yields indices in ``[0, datasize)`` each epoch.

    Parameters
    ----------
    datasize : int
        Number of indices to generate.
    shuffle : bool, optional
        Whether to shuffle the indices before each iteration.
        Default is ``True``.
    """

    def __init__(self, datasize: int, shuffle: bool = True): ...
    def set_seed(self, seed: int) -> None:
        r"""Set the random seed for shuffling.

        Parameters
        ----------
        seed : int
            Random seed value.
        """
        ...

class BaseProcessor(dp.iter.IterDataPipe):
    r"""A base processor that defines the property of fields.

    Parameters
    ----------
    dataset : :class:`~RecDataSet`
        The recommendation dataset providing fields and metadata.

    Raises
    ------
    AttributeError
        If ``fields`` are not given or ``None`` before using.
    """

    def __init__(self, dataset: RecDataSet) -> None: ...
    @property
    def dataset(self) -> RecDataSet:
        r"""Return the underlying :class:`~RecDataSet`."""
        ...

    @property
    def fields(self) -> FieldTuple[Field]:
        r"""Return the :class:`~FieldTuple` of fields."""
        ...

    @staticmethod
    def listmap(func: Callable, *iterables) -> List[Any]:
        r"""Apply a function to multiple iterables and return a list.

        Parameters
        ----------
        func : callable
            The function to be applied.
        *iterables
            Multiple iterables to be processed.

        Returns
        -------
        list
            The results after applying the function to the iterables.
        """
        ...

    @classmethod
    def to_rows(cls, field_dict: Dict[Field, Iterable[T]]) -> List[Dict[Field, T]]:
        r"""Convert a column-oriented dict to a list of row dicts.

        Parameters
        ----------
        field_dict : dict
            Mapping from :class:`~Field` to an iterable of values.

        Returns
        -------
        list of dict
            Each dict maps :class:`~Field` to a single value.
        """
        ...

    # Functional form of 'GenTrainPositiveSampler'
    def gen_train_sampling_pos_(self: T) -> T:
        r"""Sampling a positive item for each user.

        Examples
        --------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.train().choiced_user_ids_source().gen_train_sampling_pos_()
        >>> next(iter(datapipe))
        {Field(USER:ID,USER): 12623, Field(ITEM:ID,ITEM,POSITIVE): 6467}
        """

    # Functional form of 'GenTrainNegativeSampler'
    def gen_train_sampling_neg_(
        self: T, num_negatives: int = 1, unseen_only: bool = True
    ) -> T:
        r"""Sampling negatives for each user.

        Parameters
        ----------
        num_negatives : int, optional
            The number of negatives for each row. Default is ``1``.
        unseen_only : bool, optional
            If ``True``, sample negatives from unseen items only.
            If ``False``, sample from all items. Default is ``True``.

        Examples
        --------
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
    def seq_train_yielding_pos_(
        self: T,
        start_idx_for_target: Optional[int] = 1,
        end_idx_for_input: Optional[int] = -1,
    ) -> T:
        r"""Yielding positive sequence for each user sequence.

        Parameters
        ----------
        start_idx_for_target : int or None, optional
            Target sequence as ``seq[start_idx_for_target:]``.
            ``None`` means the full sequence. Default is ``1``.
        end_idx_for_input : int or None, optional
            Input sequence as ``seq[:end_idx_for_input]``.
            ``None`` means the full sequence. Default is ``-1``.

        Examples
        --------
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
    def seq_train_sampling_neg_(
        self: T, num_negatives: int = 1, unseen_only: bool = True
    ) -> T:
        r"""Sampling negatives for each positive.

        Parameters
        ----------
        num_negatives : int, optional
            The number of negatives for each row. Default is ``1``.
        unseen_only : bool, optional
            If ``True``, sample negatives from unseen items only.
            If ``False``, sample from all items. Default is ``True``.

        Examples
        --------
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
    def valid_sampling_(
        self: T,
        ranking: Literal["full", "pool"] = "full",
        num_negatives: int = NUM_NEGS_FOR_SAMPLE_BASED_RANKING,
    ) -> T:
        r"""Sampler for validation.

        Parameters
        ----------
        ranking : ``'full'`` or ``'pool'``, optional
            ``'full'`` for full ranking, ``'pool'`` for sample-based ranking.
            Default is ``'full'``.
        num_negatives : int, optional
            The number of negatives for ``'pool'`` ranking.
            Default is ``100``.

        Yields
        ------
        dict
            A dict containing the following :class:`~Field` keys:

            - ``Field(USER:ID,USER)``: user id
            - ``Field(ITEM:ID,ITEM,SEQUENCE)``: user sequence
            - ``Field(ITEM:ID,ITEM,UNSEEN)``: target items (``'full'``) or
              target items + negative items (``'pool'``)
            - ``Field(ITEM:ID,ITEM,SEEN)``: seen items

        Examples
        --------
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
    def test_sampling_(
        self: T,
        ranking: Literal["full", "pool"] = "full",
        num_negatives: int = NUM_NEGS_FOR_SAMPLE_BASED_RANKING,
    ) -> T:
        r"""Sampler for test.

        Parameters
        ----------
        ranking : ``'full'`` or ``'pool'``, optional
            ``'full'`` for full ranking, ``'pool'`` for sample-based ranking.
            Default is ``'full'``.
        num_negatives : int, optional
            The number of negatives for ``'pool'`` ranking.
            Default is ``100``.

        Yields
        ------
        dict
            A dict containing the following :class:`~Field` keys:

            - ``Field(USER:ID,USER)``: user id
            - ``Field(ITEM:ID,ITEM,SEQUENCE)``: user sequence
            - ``Field(ITEM:ID,ITEM,UNSEEN)``: target items (``'full'``) or
              target items + negative items (``'pool'``)
            - ``Field(ITEM:ID,ITEM,SEEN)``: seen items

        Examples
        --------
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
        r"""Prune the left side of sequences to a specified maximum length.

        Parameters
        ----------
        maxlen : int
            The maximum length to prune the input data to.
        modified_fields : Iterable[:class:`~Field`]
            The fields to be modified.

        Notes
        -----
        ``[1, 2, 3, 4] --(maxlen=3)--> [2, 3, 4]``

        ``[3, 4] --(maxlen=3)--> [3, 4]``

        Examples
        --------
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
        r"""Prune the right side of sequences to a specified maximum length.

        Parameters
        ----------
        maxlen : int
            The maximum length to prune the input data to.
        modified_fields : Iterable[:class:`~Field`]
            The fields to be modified.

        Notes
        -----
        ``[1, 2, 3, 4] --(maxlen=3)--> [1, 2, 3]``

        ``[3, 4] --(maxlen=3)--> [3, 4]``

        Examples
        --------
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
        r"""Add a specified offset to sequence elements.

        Parameters
        ----------
        offset : int
            Amount to add to each element.
        modified_fields : Iterable[:class:`~Field`]
            The fields to be modified.

        Notes
        -----
        ``[1, 2, 3, 4] --(offset=1)--> [2, 3, 4, 5]``

        Examples
        --------
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
    def lpad_(
        self: T, maxlen: int, modified_fields: Iterable[Field], padding_value: int = 0
    ) -> T:
        r"""Left-pad sequences to a maximum length.

        Parameters
        ----------
        maxlen : int
            The maximum length to pad the sequences to.
        modified_fields : Iterable[:class:`~Field`]
            The fields to be modified.
        padding_value : int, optional
            The value to use for padding. Default is ``0``.

        Notes
        -----
        ``[1, 2, 3, 4] --(maxlen=7, padding_value=0)--> [0, 0, 0, 1, 2, 3, 4]``

        ``[1, 2, 3, 4] --(maxlen=4, padding_value=0)--> [1, 2, 3, 4]``

        Examples
        --------
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
    def rpad_(
        self: T, maxlen: int, modified_fields: Iterable[Field], padding_value: int = 0
    ) -> T:
        r"""Right-pad sequences to a maximum length.

        Parameters
        ----------
        maxlen : int
            The maximum length to pad the sequences to.
        modified_fields : Iterable[:class:`~Field`]
            The fields to be modified.
        padding_value : int, optional
            The value to use for padding. Default is ``0``.

        Notes
        -----
        ``[1, 2, 3, 4] --(maxlen=7, padding_value=0)--> [1, 2, 3, 4, 0, 0, 0]``

        ``[1, 2, 3, 4] --(maxlen=4, padding_value=0)--> [1, 2, 3, 4]``

        Examples
        --------
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
        r"""Batch rows and convert to ``Dict[Field, List[Any]]``.

        Parameters
        ----------
        batch_size : int
            The size of each batch.
        drop_last : bool, optional
            Whether to drop the last incomplete batch. Default is ``False``.
        """

    # Functional form of 'Marker'
    def mark_(self: T, **markers) -> T:
        r"""Mark each yielded dict with additional key-value pairs.

        Parameters
        ----------
        **markers
            Arbitrary keyword arguments to insert into each row dict.

        Examples
        --------
        >>> source_dp1 = IterableWrapper([{'i': i} for i in range(3)]).mark_(dataset='A')
        >>> source_dp2 = IterableWrapper([{'i': i} for i in range(3)]).mark_(dataset='B')
        >>> d = {source_dp1: 1, source_dp2: 1}
        >>> sample_mul_dp = SampleMultiplexer(pipes_to_weights_dict=d)
        >>> list(sample_mul_dp)
        [{'i': 0, 'dataset': 'B'},
        {'i': 1, 'dataset': 'B'},
        {'i': 0, 'dataset': 'A'},
        {'i': 1, 'dataset': 'A'},
        {'i': 2, 'dataset': 'B'},
        {'i': 2, 'dataset': 'A'}]
        """

    # Functional form of 'ToTensor'
    def tensor_(self: T) -> T:
        r"""Convert lists into :class:`torch.Tensor` objects.

        Notes
        -----
        The returned tensor is at least 2-d.
        """

    # ========================================Functional forms from IterDataPipe========================================

    # Functional form of 'Batcher'
    def batch(
        self: T, batch_size: int, drop_last: bool = False, wrapper_class=DataChunk
    ) -> T:
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
    def collate(
        self: T,
        conversion: Optional[
            Union[
                Callable[..., Any],
                Dict[Union[str, Any], Union[Callable, Any]],
            ]
        ] = default_collate,
        collate_fn: Optional[Callable] = None,
    ) -> T:
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
    r"""Source datapipe that serves as the starting point of train/valid/test pipelines.

    Parameters
    ----------
    dataset : :class:`~RecDataSet`
        The recommendation dataset.
    source : iterable
        Source data, either an :class:`~IterDataPipe` or a finite iterable of row dicts.
    datasize : int, optional
        Override for the source length. If ``None``, inferred from ``source``.
    shuffle : bool, optional
        Whether to shuffle indices each epoch. Default is ``True``.
    """

    def __init__(
        self,
        dataset: RecDataSet,
        source: Iterable[Dict[Field, Any]],
        datasize: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        self.source: Iterable[Dict[Field, Any]]
        self.datasize: int
        self.launcher: Union[Launcher, Iterable[Dict[Field, Any]]]

    def guard_mode(self):
        r"""Ensure the dataset is set to the required mode.

        This is especially necessary for datapipe sources where the mode
        may have been changed externally.
        """

class PostProcessor(BaseProcessor):
    r"""A post-processor that wraps another :class:`~IterDataPipe` object.

    Parameters
    ----------
    source : :class:`~BaseProcessor`
        The data pipeline to be wrapped.
    """

class SampleMultiplexer(dp.iter.IterDataPipe):
    r"""Yield items by sampling from weighted :class:`~IterDataPipe` instances.

    Takes a dict of ``(IterDataPipe, weight)`` pairs and yields items by
    sampling from these datapipes with respect to their weights. When
    individual datapipes are exhausted, sampling continues from the remaining
    datapipes according to their relative weights.

    If you wish to maintain the same ratio of weights indefinitely, ensure
    that the inputs are never exhausted, e.g. by applying ``cycle``.

    Parameters
    ----------
    pipes_to_weights_dict : dict
        Mapping from :class:`~IterDataPipe` to a positive float weight.
        The total weight of unexhausted datapipes is normalized to 1.

    Raises
    ------
    ValueError
        If ``pipes_to_weights_dict`` is empty or contains non-positive weights.

    Examples
    --------
    >>> source_dp1 = IterableWrapper([0] * 10)
    >>> source_dp2 = IterableWrapper([1] * 10)
    >>> d = {source_dp1: 99999999, source_dp2: 0.0000001}
    >>> sample_mul_dp = SampleMultiplexer(pipes_to_weights_dict=d)
    >>> list(sample_mul_dp)
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    """
    def __init__(self, pipes_to_weights_dict: Dict[dp.iter.IterDataPipe, float]): ...
