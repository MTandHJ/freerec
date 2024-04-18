

from typing import List, Tuple, Optional, Iterable

import random
import torchdata.datapipes as dp

from .base import PostProcessor, BaseProcessor
from ..fields import Field
from ..tags import USER, ITEM, ID, SEQUENCE, UNSEEN, SEEN, POSITIVE, NEGATIVE,  MATCHING, NEXTITEM
from ..utils import negsamp_vectorized_bsearch
from ...utils import timemeter


__all__ = [
    'GenTrainPositiveSampler', 'GenTrainNegativeSampler',
    'SeqTrainPositiveYielder', 'SeqTrainNegativeSampler',
    'ValidSampler', 'TestSampler',
]


NUM_NEGS_FOR_SAMPLE_BASED_RANKING = 100


class BaseSampler(PostProcessor):
    r"""
    Base Sampler for training.

    Parameters:
    -----------
    source: Source datapipe defined in source.py
    dataset: RecDataSet 
        The dataset that provides the data source.
    """

    def __init__(self, source: BaseProcessor) -> None:
        super().__init__(source)
        self.User: Field = self.fields[USER, ID]
        self.Item: Field = self.fields[ITEM, ID]
        self.prepare()

    def prepare(self):
        pass

    @property
    def seenItems(self) -> Tuple:
        return self.__seenItems

    @seenItems.setter
    def seenItems(self, seenItems: Iterable):
        self.__seenItems = tuple(tuple(items) for items in seenItems)

    @property
    def unseenItems(self) -> Tuple:
        return self.__unseenItems
    
    @unseenItems.setter
    def unseenItems(self, unseenItems: Iterable):
        self.__unseenItems = tuple(tuple(items) for items in unseenItems) 


#===============================For Training ===============================


@dp.functional_datapipe("gen_train_sampling_pos_")
class GenTrainPositiveSampler(BaseSampler):

    def __init__(self, source: BaseProcessor) -> None:
        super().__init__(source)
        self.IPos = self.Item.fork(POSITIVE)

    @timemeter
    def prepare(self):
        seenItems = [set() for _ in range(self.User.count)]

        self.listmap(
            lambda user, item: seenItems[user].add(item),
            self.dataset.train().interdata[self.User],
            self.dataset.train().interdata[self.Item]
        )

        # sorting for ordered positives
        self.seenItems = [sorted(items) for items in seenItems]

    def _sample_pos(self, user: int) -> int:
        r"""
        Randomly sample a positive item for a user.

        Parameters:
        -----------
        user: int 
            A user index.

        Returns:
        --------
        positive: int 
            A positive item that the user has interacted with.
        """
        return random.choice(self.seenItems[user])

    def _check(self, user: int) -> bool:
        return len(self.seenItems[user]) > 0

    def __iter__(self):
        for row in self.source:
            user = row[self.User]
            if self._check(user):
                row[self.IPos] = self._sample_pos(user)
                yield row


@dp.functional_datapipe("gen_train_sampling_neg_")
class GenTrainNegativeSampler(GenTrainPositiveSampler):

    def __init__(
        self, 
        source: BaseProcessor,
        num_negatives: int = 1, unseen_only: bool = True
    ) -> None:
        super().__init__(source)
        self.INeg = self.Item.fork(NEGATIVE)
        self.unseen_only = unseen_only
        self.num_negatives = num_negatives

    def _sample_neg(self, user: int) -> List[int]:
        r"""Randomly sample negative items for a user.

        Parameters:
        ----------
        user: int 
            A user index.

        Returns:
        --------
        negatives: List[int] 
            `unseen_only == True`:
                A list of negative items that the user has not interacted with.
            `unseen_only == False`:
                A list of negative items from [0, self.Item.count - 1]
        """
        seen = self.seenItems[user] if self.unseen_only else []
        return negsamp_vectorized_bsearch(
            seen, self.Item.count, self.num_negatives
        )

    def __iter__(self):
        for row in self.source:
            user = row[self.User]
            row[self.INeg] = self._sample_neg(user)
            yield row


@dp.functional_datapipe("seq_train_yielding_pos_")
class SeqTrainPositiveYielder(BaseSampler):
    r"""
    Sequence sampler for training.

    Parameters:
    -----------
    yielding_target_only: bool, default to `False`,
        `False`: Only yielding (user, sequence)
    start_idx_for_target: int, optional
        Target sequence as seq[start_idx_for_target:]
        `None`: seq
    end_idx_for_input: int, optional
        Input sequence as seq[:end_idx_for_input]
        `None`: seq

    Flows:
    ------
    - yielding_target_only == True:
        yielding (user, seq)
    - yielding_target_only == False and sampling_neg == False:
        yielding (user, seq[:end_idx_for_input], seq[start_idx_for_target:])
    - yielding_target_only == False and sampling_neg == True:
        yielding (user, seq[:end_idx_for_input], seq[start_idx_for_target:], negatives)
    where negatives is in the size of  (len(positives), num_negatives)
    """

    def __init__(
        self, 
        source: BaseProcessor,
        start_idx_for_target: Optional[int] = 1, 
        end_idx_for_input: Optional[int] = -1,
    ) -> None:
        super().__init__(source)
        self.ISeq = self.Item.fork(SEQUENCE)
        self.IPos = self.Item.fork(POSITIVE)
        self.start_idx_for_target = start_idx_for_target
        self.end_idx_for_input = end_idx_for_input

    def _check(self, seq: Iterable) -> bool:
        return len(seq) > 1

    def __iter__(self):
        for row in self.source:
            seq = row[self.ISeq]
            if self._check(seq):
                positives = seq[self.start_idx_for_target:]
                seq = seq[:self.end_idx_for_input]
                row.update([
                    (self.ISeq, seq),
                    (self.IPos, positives)
                ])
                yield row


@dp.functional_datapipe("seq_train_sampling_neg_")
class SeqTrainNegativeSampler(BaseSampler):

    def __init__(
        self, 
        source: BaseProcessor,
        num_negatives: int = 1, unseen_only: bool = True
    ) -> None:
        super().__init__(source)
        self.ISeq = self.Item.fork(SEQUENCE)
        self.IPos = self.Item.fork(POSITIVE)
        self.INeg = self.Item.fork(NEGATIVE)
        self.unseen_only = unseen_only
        self.num_negatives = num_negatives

    @timemeter
    def prepare(self):
        seenItems = [set() for _ in range(self.User.count)]

        self.listmap(
            lambda user, item: seenItems[user].add(item),
            self.dataset.train().interdata[self.User],
            self.dataset.train().interdata[self.Item],
        )

        # sorting for ordered positives
        self.seenItems = [sorted(items) for items in seenItems]

    def _sample_neg(self, user: int, positives: Tuple) -> List[int]:
        r"""Randomly sample negative items for a user.

        Parameters:
        ----------
        user: int
        positives: Tuple 
            A tuple of positives.

        Returns:
        --------
        negatives: np.ndarray
            `unseen_only == True`:
                A list of negative items that the user has not interacted with.
            `unseen_only == False`:
                A list of negative items from [0, self.Item.count - 1]
        """
        seen = self.seenItems[user] if self.unseen_only else []
        return negsamp_vectorized_bsearch(
            seen, self.Item.count, (len(positives), self.num_negatives)
        )

    def __iter__(self):
        for row in self.source:
            row[self.INeg] = self._sample_neg(
                row[self.User], row[self.IPos]
            )
            yield row


#===============================For Evaluation===============================

@dp.functional_datapipe("valid_sampling_")
class ValidSampler(BaseSampler):

    def __init__(
        self, source: BaseProcessor,
        ranking: str = 'full', num_negatives: int = NUM_NEGS_FOR_SAMPLE_BASED_RANKING
    ) -> None:
        super().__init__(source)
        self.ISeq = self.Item.fork(SEQUENCE)
        self.IUnseen = self.Item.fork(UNSEEN)
        self.ISeen = self.Item.fork(SEEN)

        assert ranking in ('full', 'pool'), f"`ranking` should be 'full' or 'pool' but {ranking} received ..."
        self.sampling_neg = True if ranking == 'pool' else False
        self.num_negatives = num_negatives

    @timemeter
    def prepare(self):
        seenItems = [[] for _ in range(self.User.count)]
        unseenItems = [[] for _ in range(self.User.count)]

        self.listmap(
            lambda user, item: seenItems[user].append(item),
            self.dataset.train().interdata[self.User],
            self.dataset.train().interdata[self.Item],
        )

        self.listmap(
            lambda user, item: unseenItems[user].append(item),
            self.dataset.valid().interdata[self.User],
            self.dataset.valid().interdata[self.Item],
        )

        self.seenItems = seenItems
        self.unseenItems = unseenItems
        self.negItems = dict()

    def _sample_neg(self, user: int, k: int, positive: int, seen: Tuple[int]):
        """Sampling negatives for ranking_from_pool"""
        idx = (user, k)
        if self.negItems.get(idx, None) is None:
            seen = sorted(set(
                (positive,) + seen
            ))
            self.negItems[idx] = tuple(
                negsamp_vectorized_bsearch(
                    seen, self.Item.count, self.num_negatives
                )
            )
        return self.negItems[idx]

    def _check(self, user: int) -> bool:
        return len(self.unseenItems[user]) > 0

    def _matching_from_pool(self):
        for row in self.source:
            user = row[self.User]
            seq = seen = self.seenItems[user]
            for k, positive in enumerate(self.unseenItems[user]):
                unseen = (positive,) + self._sample_neg(user, k, positive, seen)
                yield {self.User: user, self.ISeq: seq, self.IUnseen: unseen, self.ISeen: seen}

    def _matching_from_full(self):
        for row in self.source:
            user = row[self.User]
            if self._check(user): 
                seq = seen = self.seenItems[user]
                unseen = self.unseenItems[user]
                yield {self.User: user, self.ISeq: seq, self.IUnseen: unseen, self.ISeen: seen}

    def _nextitem_from_pool(self):
        for row in self.source:
            user = row[self.User]
            seen = self.seenItems[user]
            for k, positive in enumerate(self.unseenItems[user]):
                seq = self.seenItems[user] + self.unseenItems[user][:k]
                unseen = (positive,) + self._sample_neg(user, k, positive, seen)
                yield {self.User: user, self.ISeq: seq, self.IUnseen: unseen, self.ISeen: seen}

    def _nextitem_from_full(self):
        for row in self.source:
            user = row[self.User]
            seen = self.seenItems[user]
            for k, positive in enumerate(self.unseenItems[user]):
                seq = self.seenItems[user] + self.unseenItems[user][:k]
                unseen = (positive,)
                yield {self.User: user, self.ISeq: seq, self.IUnseen: unseen, self.ISeen: seen}

    def __iter__(self):
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

    @timemeter
    def prepare(self):
        seenItems = [[] for _ in range(self.User.count)]
        unseenItems = [[] for _ in range(self.User.count)]

        self.listmap(
            lambda user, item: seenItems[user].append(item),
            self.dataset.train().interdata[self.User],
            self.dataset.train().interdata[self.Item],
        )

        self.listmap(
            lambda user, item: seenItems[user].append(item),
            self.dataset.valid().interdata[self.User],
            self.dataset.valid().interdata[self.Item],
        )


        self.listmap(
            lambda user, item: unseenItems[user].append(item),
            self.dataset.test().interdata[self.User],
            self.dataset.test().interdata[self.Item],
        )

        self.seenItems = seenItems
        self.unseenItems = unseenItems
        self.negItems = dict()