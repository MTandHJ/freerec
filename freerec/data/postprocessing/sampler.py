import random
from typing import Iterable, List, Literal, Optional, Tuple

import torchdata.datapipes as dp

from freerec.data.fields import Field
from freerec.data.postprocessing.base import BaseProcessor, PostProcessor
from freerec.data.tags import (
    ID,
    ITEM,
    MATCHING,
    NEGATIVE,
    NEXTITEM,
    POSITIVE,
    SEEN,
    SEQUENCE,
    UNSEEN,
    USER,
)
from freerec.data.utils import negsamp_vectorized_bsearch
from freerec.utils import timemeter

__all__ = [
    "GenTrainPositiveSampler",
    "GenTrainNegativeSampler",
    "SeqTrainPositiveYielder",
    "SeqTrainNegativeSampler",
    "ValidSampler",
    "TestSampler",
]


NUM_NEGS_FOR_SAMPLE_BASED_RANKING = 100


class BaseSampler(PostProcessor):
    r"""Base sampler for training pipelines.

    Sets up user/item :class:`~Field` references and delegates to
    :meth:`prepare` for subclass-specific initialization.

    Parameters
    ----------
    source : :class:`~BaseProcessor`
        Source datapipe defined in ``source.py``.
    """

    def __init__(self, source: BaseProcessor) -> None:
        r"""Initialize the BaseSampler."""
        super().__init__(source)
        self.User: Field = self.fields[USER, ID]
        self.Item: Field = self.fields[ITEM, ID]
        if self.Item is not None:
            self.ISeq: Field = self.Item.fork(SEQUENCE)
            self.IPos: Field = self.Item.fork(POSITIVE)
            self.INeg: Field = self.Item.fork(NEGATIVE)
            self.IUnseen: Field = self.Item.fork(UNSEEN)
            self.ISeen: Field = self.Item.fork(SEEN)
        self.prepare()

    def prepare(self):
        r"""Prepare subclass-specific data structures (no-op by default)."""
        pass

    @property
    def seenItems(self) -> Tuple:
        r"""Return the tuple of seen-item sequences per user."""
        return self.__seenItems

    @seenItems.setter
    def seenItems(self, seenItems: Iterable):
        r"""Set the seen-item sequences per user."""
        self.__seenItems = tuple(tuple(items) for items in seenItems)

    @property
    def unseenItems(self) -> Tuple:
        r"""Return the tuple of unseen-item sequences per user."""
        return self.__unseenItems

    @unseenItems.setter
    def unseenItems(self, unseenItems: Iterable):
        r"""Set the unseen-item sequences per user."""
        self.__unseenItems = tuple(tuple(items) for items in unseenItems)


# ===============================For Training ===============================


@dp.functional_datapipe("gen_train_sampling_pos_")
class GenTrainPositiveSampler(BaseSampler):
    r"""Sample a positive item for each user.

    For every incoming row, a random positive item from the user's
    interaction history is sampled and stored under :pycode:`IPos`.

    Examples
    --------
    >>> dataset: RecDataSet
    >>> datapipe = dataset.train().choiced_user_ids_source().gen_train_sampling_pos_()
    >>> next(iter(datapipe))
    {Field(USER:ID,USER): 12623, Field(ITEM:ID,ITEM,POSITIVE): 6467}
    """

    @timemeter
    def prepare(self):
        r"""Build per-user sorted lists of seen items from training sequences."""
        seenItems = [set() for _ in range(self.User.count)]

        self.listmap(
            lambda row: seenItems[row[self.User]].update(row[self.ISeq]),
            self.dataset.train().to_seqs(),
        )

        # sorting for ordered positives
        self.seenItems = [sorted(items) for items in seenItems]

    def _sample_pos(self, user: int) -> int:
        r"""Randomly sample a positive item for a user.

        Parameters
        ----------
        user : int
            A user index.

        Returns
        -------
        int
            A positive item that the user has interacted with.
        """
        return random.choice(self.seenItems[user])

    def _check(self, user: int) -> bool:
        r"""Return whether the user has at least one seen item."""
        return len(self.seenItems[user]) > 0

    def __iter__(self):
        r"""Yield rows augmented with a sampled positive item."""
        for row in self.source:
            user = row[self.User]
            if self._check(user):
                row[self.IPos] = self._sample_pos(user)
                yield row


@dp.functional_datapipe("gen_train_sampling_neg_")
class GenTrainNegativeSampler(GenTrainPositiveSampler):
    r"""Sample negative items for each user.

    Extends :class:`~GenTrainPositiveSampler` by adding negative sampling.
    Negative items are drawn either from unseen items only or from all items.

    Parameters
    ----------
    source : :class:`~BaseProcessor`
        Source datapipe.
    num_negatives : int, optional
        The number of negatives for each row. Default is ``1``.
    unseen_only : bool, optional
        If ``True``, sample negatives from unseen items only.
        If ``False``, sample negatives from all items. Default is ``True``.
    nums_need_vectorized_bsearch : int, optional
        Threshold above which vectorized binary search is used.
        Default is ``17``.

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

    def __init__(
        self,
        source: BaseProcessor,
        num_negatives: int = 1,
        unseen_only: bool = True,
        nums_need_vectorized_bsearch: int = 17,
    ) -> None:
        r"""Initialize the GenTrainNegativeSampler."""
        self.unseen_only = unseen_only
        super().__init__(source)
        self.num_negatives = num_negatives
        self.need_vectorized_bsearch = (
            self.num_negatives >= nums_need_vectorized_bsearch
        )

    @timemeter
    def prepare(self):
        r"""Build per-user sorted lists of seen items for negative filtering."""
        seenItems = [set() for _ in range(self.User.count)]

        if self.unseen_only:
            self.listmap(
                lambda row: seenItems[row[self.User]].update(row[self.ISeq]),
                self.dataset.train().to_seqs(),
            )

        # sorting for ordered positives
        self.seenItems = [sorted(items) for items in seenItems]

    def _sample_one(self, seen: Iterable[int]) -> int:
        r"""Sample a single negative item not in *seen*."""
        neg = random.randint(0, self.Item.count - 1)
        while neg in seen:
            neg = random.randint(0, self.Item.count - 1)
        return neg

    def _sample_neg(self, user: int) -> List[int]:
        r"""Randomly sample negative items for a user.

        Parameters
        ----------
        user : int
            A user index.

        Returns
        -------
        list of int
            When ``unseen_only`` is ``True``, a list of items the user has
            not interacted with. Otherwise, a list drawn from
            ``[0, self.Item.count - 1]``.
        """
        seen = self.seenItems[user]
        if self.need_vectorized_bsearch:
            # `vectorized_bsearch` would be faster
            # if more negatives are sampled at once
            return negsamp_vectorized_bsearch(seen, self.Item.count, self.num_negatives)
        else:
            return self.listmap(self._sample_one, [seen] * self.num_negatives)

    def __iter__(self):
        r"""Yield rows augmented with sampled negative items."""
        for row in self.source:
            row[self.INeg] = self._sample_neg(row[self.User])
            yield row


@dp.functional_datapipe("seq_train_yielding_pos_")
class SeqTrainPositiveYielder(BaseSampler):
    r"""Yield positive sequences derived from each user's item sequence.

    Splits the item sequence into an input portion and a target portion
    according to the given indices.

    Parameters
    ----------
    source : :class:`~BaseProcessor`
        Source datapipe.
    start_idx_for_target : int or None, optional
        Target sequence is ``seq[start_idx_for_target:]``.
        ``None`` keeps the full sequence. Default is ``1``.
    end_idx_for_input : int or None, optional
        Input sequence is ``seq[:end_idx_for_input]``.
        ``None`` keeps the full sequence. Default is ``-1``.

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

    def __init__(
        self,
        source: BaseProcessor,
        start_idx_for_target: Optional[int] = 1,
        end_idx_for_input: Optional[int] = -1,
    ) -> None:
        r"""Initialize the SeqTrainPositiveYielder."""
        super().__init__(source)
        self.start_idx_for_target = start_idx_for_target
        self.end_idx_for_input = end_idx_for_input

    def _check(self, seq: Iterable) -> bool:
        r"""Return whether the sequence has more than one element."""
        return len(seq) > 1

    def __iter__(self):
        r"""Yield rows with input and target sequences split from the original."""
        for row in self.source:
            seq = row[self.ISeq]
            if self._check(seq):
                row[self.IPos] = seq[self.start_idx_for_target :]
                row[self.ISeq] = seq[: self.end_idx_for_input]
                yield row


@dp.functional_datapipe("seq_train_sampling_neg_")
class SeqTrainNegativeSampler(BaseSampler):
    r"""Sample negative items for each positive in the sequence.

    For every positive item in the target sequence, one or more negative
    items are sampled.

    Parameters
    ----------
    source : :class:`~BaseProcessor`
        Source datapipe.
    num_negatives : int, optional
        The number of negatives for each positive. Default is ``1``.
    unseen_only : bool, optional
        If ``True``, sample negatives from unseen items only.
        If ``False``, sample from all items. Default is ``True``.
    nums_need_vectorized_bsearch : int, optional
        Threshold above which vectorized binary search is used.
        Default is ``17``.

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

    def __init__(
        self,
        source: BaseProcessor,
        num_negatives: int = 1,
        unseen_only: bool = True,
        nums_need_vectorized_bsearch: int = 17,
    ) -> None:
        r"""Initialize the SeqTrainNegativeSampler."""
        self.unseen_only = unseen_only
        super().__init__(source)
        self.num_negatives = num_negatives
        self.nums_need_vectorized_bsearch = nums_need_vectorized_bsearch

    @timemeter
    def prepare(self):
        r"""Build per-user sorted lists of seen items for negative filtering."""
        seenItems = [set() for _ in range(self.User.count)]

        if self.unseen_only:
            self.listmap(
                lambda row: seenItems[row[self.User]].update(row[self.ISeq]),
                self.dataset.train().to_seqs(),
            )

        # sorting for ordered positives
        self.seenItems = [sorted(items) for items in seenItems]

    def _sample_one(self, seen: Iterable[int]) -> int:
        r"""Sample a single negative item not in *seen*."""
        neg = random.randint(0, self.Item.count - 1)
        while neg in seen:
            neg = random.randint(0, self.Item.count - 1)
        return neg

    def _sample_neg(self, user: int, positives: Tuple) -> List[int]:
        r"""Randomly sample negative items for a user.

        Parameters
        ----------
        user : int
            A user index.
        positives : tuple
            A tuple of positive item indices.

        Returns
        -------
        list of int or :class:`numpy.ndarray`
            When ``unseen_only`` is ``True``, items the user has not
            interacted with. Otherwise, items drawn from
            ``[0, self.Item.count - 1]``.
        """
        seen = self.seenItems[user]
        if len(positives) > self.nums_need_vectorized_bsearch or self.num_negatives > 1:
            # `vectorized_bsearch` would be faster
            # if more negatives are sampled at once
            if self.num_negatives > 1:
                return negsamp_vectorized_bsearch(
                    seen, self.Item.count, (len(positives), self.num_negatives)
                )
            else:
                return negsamp_vectorized_bsearch(seen, self.Item.count, len(positives))
        else:
            return self.listmap(self._sample_one, [seen] * len(positives))

    def __iter__(self):
        r"""Yield rows augmented with sampled negatives for each positive."""
        for row in self.source:
            row[self.INeg] = self._sample_neg(row[self.User], row[self.IPos])
            yield row


# ===============================For Evaluation===============================


@dp.functional_datapipe("valid_sampling_")
class ValidSampler(BaseSampler):
    r"""Sampler for validation.

    Produces rows containing the user's input sequence, unseen target items,
    and seen items, optionally with sampled negative items for pool-based
    ranking.

    Parameters
    ----------
    source : :class:`~BaseProcessor`
        Source datapipe.
    ranking : ``'full'`` or ``'pool'``, optional
        ``'full'`` for full ranking; ``'pool'`` for sample-based ranking.
        Default is ``'full'``.
    num_negatives : int, optional
        The number of negatives for ``'pool'`` ranking.
        Default is ``100``.

    Yields
    ------
    dict
        Row dict containing:

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

    def __init__(
        self,
        source: BaseProcessor,
        ranking: Literal["full", "pool"] = "full",
        num_negatives: int = NUM_NEGS_FOR_SAMPLE_BASED_RANKING,
    ) -> None:
        r"""Initialize the ValidSampler."""
        super().__init__(source)
        assert ranking in ("full", "pool"), (
            f"`ranking` should be 'full' or 'pool' but {ranking} received ..."
        )
        self.sampling_neg = True if ranking == "pool" else False
        self.num_negatives = num_negatives

    @timemeter
    def prepare(self):
        r"""Build per-user seen and unseen item lists from train/valid splits."""
        seenItems = [[] for _ in range(self.User.count)]
        unseenItems = [[] for _ in range(self.User.count)]

        self.listmap(
            lambda row: seenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.train().to_seqs(),
        )

        self.listmap(
            lambda row: unseenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.valid().to_seqs(),
        )

        self.seenItems = seenItems
        self.unseenItems = unseenItems
        self.negItems = dict()

    def _sample_neg(self, user: int, k: int, positive: int, seen: Tuple[int]):
        r"""Sample negatives for pool-based ranking."""
        idx = (user, k)
        if self.negItems.get(idx, None) is None:
            seen = sorted(set((positive,) + seen))
            self.negItems[idx] = tuple(
                negsamp_vectorized_bsearch(seen, self.Item.count, self.num_negatives)
            )
        return self.negItems[idx]

    def _check(self, user: int) -> bool:
        r"""Return whether the user has at least one unseen item."""
        return len(self.unseenItems[user]) > 0

    def _matching_from_pool(self):
        r"""Yield matching rows with pool-based negative sampling."""
        for row in self.source:
            user = row[self.User]
            seq = seen = self.seenItems[user]
            for k, positive in enumerate(self.unseenItems[user]):
                unseen = (positive,) + self._sample_neg(user, k, positive, seen)
                yield {
                    self.User: user,
                    self.ISeq: seq,
                    self.IUnseen: unseen,
                    self.ISeen: seen,
                }

    def _matching_from_full(self):
        r"""Yield matching rows for full ranking."""
        for row in self.source:
            user = row[self.User]
            if self._check(user):
                seq = seen = self.seenItems[user]
                unseen = self.unseenItems[user]
                yield {
                    self.User: user,
                    self.ISeq: seq,
                    self.IUnseen: unseen,
                    self.ISeen: seen,
                }

    def _nextitem_from_pool(self):
        r"""Yield next-item rows with pool-based negative sampling."""
        for row in self.source:
            user = row[self.User]
            seen = self.seenItems[user]
            for k, positive in enumerate(self.unseenItems[user]):
                seq = self.seenItems[user] + self.unseenItems[user][:k]
                unseen = (positive,) + self._sample_neg(user, k, positive, seen)
                yield {
                    self.User: user,
                    self.ISeq: seq,
                    self.IUnseen: unseen,
                    self.ISeen: seen,
                }

    def _nextitem_from_full(self):
        r"""Yield next-item rows for full ranking."""
        for row in self.source:
            user = row[self.User]
            seen = self.seenItems[user]
            for k, positive in enumerate(self.unseenItems[user]):
                seq = self.seenItems[user] + self.unseenItems[user][:k]
                unseen = (positive,)
                yield {
                    self.User: user,
                    self.ISeq: seq,
                    self.IUnseen: unseen,
                    self.ISeen: seen,
                }

    def __iter__(self):
        r"""Yield validation rows according to task type and ranking mode."""
        if self.dataset.TASK is MATCHING:
            if self.sampling_neg:
                yield from self._matching_from_pool()
            else:
                yield from self._matching_from_full()
        elif self.dataset.TASK is NEXTITEM:
            if self.sampling_neg:
                yield from self._nextitem_from_pool()
            else:
                yield from self._nextitem_from_full()


@dp.functional_datapipe("test_sampling_")
class TestSampler(ValidSampler):
    r"""Sampler for test.

    Identical to :class:`~ValidSampler` except that seen items include both
    training and validation interactions, and unseen items come from the
    test split.

    Parameters
    ----------
    source : :class:`~BaseProcessor`
        Source datapipe.
    ranking : ``'full'`` or ``'pool'``, optional
        ``'full'`` for full ranking; ``'pool'`` for sample-based ranking.
        Default is ``'full'``.
    num_negatives : int, optional
        The number of negatives for ``'pool'`` ranking.
        Default is ``100``.

    Yields
    ------
    dict
        Row dict containing:

        - ``Field(USER:ID,USER)``: user id
        - ``Field(ITEM:ID,ITEM,SEQUENCE)``: user sequence
        - ``Field(ITEM:ID,ITEM,UNSEEN)``: target items (``'full'``) or
          target items + negative items (``'pool'``)
        - ``Field(ITEM:ID,ITEM,SEEN)``: seen items

    Examples
    --------
    >>> dataset: RecDataSet
    >>> datapipe = dataset.test().ordered_user_ids_source(
    ).test_sampling_(ranking='full')
    >>> next(iter(datapipe))
    {Field(USER:ID,USER): 0,
    Field(ITEM:ID,ITEM,SEQUENCE): (9449, 9839, 10076, 11155),
    Field(ITEM:ID,ITEM,UNSEEN): (11752,),
    Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
    >>> datapipe = dataset.test().ordered_user_ids_source(
    ).test_sampling_(ranking='pool', num_negatives=5)
    >>> next(iter(datapipe))
    {Field(USER:ID,USER): 0,
    Field(ITEM:ID,ITEM,SEQUENCE): (9449, 9839, 10076, 11155),
    Field(ITEM:ID,ITEM,UNSEEN): (11752, 10413, 9774, 487, 4114, 10546),
    Field(ITEM:ID,ITEM,SEEN): (9449, 9839, 10076, 11155)}
    """

    @timemeter
    def prepare(self):
        r"""Build per-user seen and unseen item lists from train/valid/test splits."""
        seenItems = [[] for _ in range(self.User.count)]
        unseenItems = [[] for _ in range(self.User.count)]

        self.listmap(
            lambda row: seenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.train().to_seqs(),
        )

        self.listmap(
            lambda row: seenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.valid().to_seqs(),
        )

        self.listmap(
            lambda row: unseenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.test().to_seqs(),
        )

        self.seenItems = seenItems
        self.unseenItems = unseenItems
        self.negItems = dict()
