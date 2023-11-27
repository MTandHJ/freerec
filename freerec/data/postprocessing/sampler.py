

from typing import List, Tuple, Optional, Iterable, Callable

import random
import torchdata.datapipes as dp
from collections import defaultdict

from .base import Postprocessor
from ..datasets.base import RecDataSet
from ..fields import SparseField
from ..tags import USER, SESSION, ITEM, ID
from ..utils import negsamp_vectorized_bsearch
from ...utils import timemeter


__all__ = [
    'GenTrainYielder', 'GenValidYielder', 'GenTestYielder',
    'GenTrainUniformSampler', 'GenValidYielder', 'GenTestYielder',
    'SeqTrainYielder', 'SeqValidYielder', 'SeqTestYielder', 
    'SeqTrainUniformSampler', 'SeqValidSampler', 'SeqTestSampler',
    'SessTrainYielder', 'SessValidYielder', 'SessTestYielder', 
    'SessTrainUniformSampler', 'SessValidSampler', 'SessTestSampler',
]


NUM_NEGS_FOR_SAMPLE_BASED_RANKING = 100


def _to_tuple(func: Callable):
    def wrapper(*args, **kwargs) -> Tuple:
        return tuple(func(*args, **kwargs))
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

#===============================For General Recommendation===============================


@dp.functional_datapipe("gen_train_yielding_")
class GenTrainYielder(Postprocessor):
    r"""A datapipe that yields (user, item) pairs."""

    def __init__(
        self, source_dp: dp.iter.IterDataPipe,
        dataset: RecDataSet
    ) -> None:
        r"""
        Initializes a new instance of the Tripleter postprocessor.
        
        Parameters:
        -----------
        source_dp: RecDatapipe or Postprocessor 
            A RecDatapipe or another Postprocessor that yields dicts of NumPy arrays.
        dataset: RecDataSet 
            The dataset that provides the data source.
        """
        super().__init__(source_dp)

        self.User: SparseField = dataset.fields[USER, ID]
        self.Item: SparseField = dataset.fields[ITEM, ID]

        self.prepare(dataset)

    @timemeter
    def prepare(self, dataset: RecDataSet):
        r"""
        Prepares the dataset by building sets of seen items for each user.

        Parameters:
        -----------
        dataset: RecDataSet 
            The dataset that provides the data source.
        """
        self.posItems = [set() for _ in range(self.User.count)]

        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.posItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        # sorting for ordered positives
        self.posItems = [tuple(sorted(items)) for items in self.posItems]

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
        return random.choice(self.posItems[user])

    def _check(self, user: int) -> bool:
        return len(self.posItems[user]) > 0

    def __iter__(self):
        for user in self.source:
            if self._check(user):
                yield [user, self._sample_pos(user)]


@dp.functional_datapipe("gen_valid_yielding_")
class GenValidYielder(GenTrainYielder):

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.seenItems = [set() for _ in range(self.User.count)]
        self.unseenItems = [set() for _ in range(self.User.count)]

        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.seenItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        for chunk in dataset.valid():
            self.listmap(
                lambda user, item: self.unseenItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        self.seenItems = [tuple(items) for items in self.seenItems]
        self.unseenItems = [tuple(items) for items in self.unseenItems]

    def _check(self, user: int) -> bool:
        return len(self.unseenItems[user]) > 0

    def __iter__(self):
        for user in self.source:
            if self._check(user):
                yield [user, self.unseenItems[user], self.seenItems[user]]


@dp.functional_datapipe("gen_test_yielding_")
class GenTestYielder(GenValidYielder):

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.seenItems = [set() for _ in range(self.User.count)]
        self.unseenItems = [set() for _ in range(self.User.count)]

        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.seenItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        for chunk in dataset.valid():
            self.listmap(
                lambda user, item: self.seenItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        for chunk in dataset.test():
            self.listmap(
                lambda user, item: self.unseenItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        self.seenItems = [tuple(items) for items in self.seenItems]
        self.unseenItems = [tuple(items) for items in self.unseenItems]


@dp.functional_datapipe("gen_train_uniform_sampling_")
class GenTrainUniformSampler(GenTrainYielder):
    r"""
    A functional datapipe for uniformly sampling users and their negatives.

    Parameters:
    -----------
    source_dp: dp.iter.IterableWrapper 
        A datapipe that yields users.
    dataset: RecDataSet 
        The dataset object that contains field objects.
    num_negatives: int 
        The number of negative samples for each piece of data.  
    """

    def __init__(
        self, source_dp: dp.iter.IterableWrapper,
        dataset: RecDataSet,
        num_negatives: int = 1,
    ) -> None:
        self.num_negatives = num_negatives
        super().__init__(source_dp, dataset)

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.posItems = [set() for _ in range(self.User.count)]
        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.posItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        # sorting for ordered positives
        self.posItems = [tuple(sorted(items)) for items in self.posItems]

    def _sample_neg(self, user: int) -> List[int]:
        r"""Randomly sample negative items for a user.

        Parameters:
        ----------
        user: int 
            A user index.

        Returns:
        --------
        negatives: List[int] 
            A list of negative items that the user has not interacted with.
        """
        seen = self.posItems[user]
        return negsamp_vectorized_bsearch(
            seen, self.Item.count, self.num_negatives
        )

    def __iter__(self):
        for user in self.source:
            if self._check(user):
                yield [user, self._sample_pos(user), self._sample_neg(user)]


@dp.functional_datapipe("gen_valid_sampling_")
class GenValidSampler(GenValidYielder):

    def _sample_negs(self, user: int, posItem: int):
        idx = (user, posItem)
        if self.negItems.get(idx, None) is None:
            seen = self.seenItems[user]
            self.negItems[idx] = tuple(
                negsamp_vectorized_bsearch(
                    seen, self.Item.count, NUM_NEGS_FOR_SAMPLE_BASED_RANKING
                ).tolist()
            )
        return self.negItems[idx]

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.seenItems = [set() for _ in range(self.User.count)]

        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.seenItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        self.negItems = dict()
        # sorting for ordered positives
        self.seenItems = [tuple(sorted(items)) for items in self.seenItems]

    def __iter__(self):
        for user, posItem in self.source:
            yield [user, (posItem,) + self._sample_negs(user, posItem)]


@dp.functional_datapipe("gen_test_sampling_")
class GenTestSampler(GenValidSampler):

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.seenItems = [set() for _ in range(self.User.count)]

        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.seenItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        for chunk in dataset.valid():
            self.listmap(
                lambda user, item: self.seenItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        self.negItems = dict()
        # sorting for ordered positives
        self.seenItems = [tuple(sorted(items)) for items in self.seenItems]


#===============================For Sequential Recommendation===============================


@dp.functional_datapipe("seq_train_yielding_")
class SeqTrainYielder(Postprocessor):
    r"""
    A functional datapipe for yielding (user, positives, targets).

    Parameters:
    -----------
    source_dp: dp.iter.IterableWrapper 
        A datapipe that yields users.
    dataset: RecDataSet 
        The dataset object that contains field objects.
    leave_one_out: bool, default to `True`
        `True`: take the last one as a target
        `False`: take `posItems[1:]` as targets
    """

    def __init__(
        self, 
        source_dp: dp.iter.IterableWrapper,
        dataset: Optional[RecDataSet] = None,
        leave_one_out: bool = True
    ) -> None:
        super().__init__(source_dp)
        self.User = dataset.fields[USER, ID]
        self.Item = dataset.fields[ITEM, ID]
        if leave_one_out:
            self.marker = -1
        else:
            self.marker = 1

        self.prepare(dataset)

    @timemeter
    def prepare(self, dataset: RecDataSet):
        pass

    def _check(self, seq: Tuple) -> bool:
        return len(seq) > 1

    def __iter__(self):
        for user, seq in self.source:
            if self._check(seq):
                yield [user, seq[:-1], seq[self.marker:]]


@dp.functional_datapipe("seq_valid_yielding_")
class SeqValidYielder(SeqTrainYielder):

    def __init__(
        self, 
        source_dp: dp.iter.IterableWrapper, 
        dataset: RecDataSet
    ) -> None:
        super().__init__(source_dp, dataset, True)

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.posItems = [[] for _ in range(self.User.count)]
        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        for chunk in dataset.valid():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        self.posItems = [tuple(items) for items in self.posItems]

    def __iter__(self):
        for user in self.source:
            posItems = self.posItems[user]
            if self._check(posItems):
                # (user, seqs, unseen, seen)
                yield [user, posItems[:-1], posItems[-1:], posItems[:-1]]


@dp.functional_datapipe("seq_test_yielding_")
class SeqTestYielder(SeqValidYielder):

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.posItems = [[] for _ in range(self.User.count)]
        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        for chunk in dataset.valid():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        for chunk in dataset.test():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        self.posItems = [tuple(items) for items in self.posItems]


@dp.functional_datapipe("seq_train_uniform_sampling_")
class SeqTrainUniformSampler(SeqTrainYielder):

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.posItems = [[] for _ in range(self.User.count)]
        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        # sorting for ordered positives
        self.posItems = [tuple(sorted(items)) for items in self.posItems]

    def _sample_neg(self, user: int, positives: Tuple) -> List[int]:
        r"""Randomly sample negative items for a user.

        Parameters:
        ----------
        user: int
        positives: Tuple 
            A tuple of positives.

        Returns:
        --------
        negatives: List[int] 
            A list of negative items that the user has not interacted with.
        """
        seen = self.posItems[user]
        return negsamp_vectorized_bsearch(
            seen, self.Item.count, len(positives)
        )

    def __iter__(self):
        for user, seq in self.source:
            if self._check(seq):
                seen = seq[:-1]
                positives = seq[self.marker:]
                negatives = self._sample_neg(user, positives)
                yield [user, seen, positives, negatives]


@dp.functional_datapipe("seq_valid_sampling_")
class SeqValidSampler(SeqValidYielder):

    @_to_tuple
    def _sample_negs(self, seen: List[int]):
        # sorting for ordered positives
        seen = sorted(seen)
        return negsamp_vectorized_bsearch(
            seen, self.Item.count, NUM_NEGS_FOR_SAMPLE_BASED_RANKING
        ).tolist()

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.posItems = [[] for _ in range(self.User.count)]
        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        for chunk in dataset.valid():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        self.posItems = [tuple(items) for items in self.posItems]

        self.negItems = self.listmap(
            self._sample_negs, self.posItems
        )

    def __iter__(self):
        for user in self.source:
            posItems = self.posItems[user]
            if self._check(posItems):
                # (user, seq, positive || negatives)
                yield [user, posItems[:-1], posItems[-1:] + self.negItems[user]]


@dp.functional_datapipe("seq_test_sampling_")
class SeqTestSampler(SeqValidSampler):

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.posItems = [[] for _ in range(self.User.count)]
        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        for chunk in dataset.valid():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        for chunk in dataset.test():
            self.listmap(
                lambda user, item: self.posItems[user].append(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        self.posItems = [tuple(items) for items in self.posItems]

        self.negItems = self.listmap(
            self._sample_negs, self.posItems
        )


#===============================For Session Recommendation===============================


@dp.functional_datapipe("sess_train_yielding_")
class SessTrainYielder(Postprocessor):
    r"""
    A functional datapipe for yielding (sess, sequences, targets).

    Parameters:
    -----------
    source_dp: dp.iter.IterableWrapper 
        A datapipe that yields users.
    dataset: RecDataSet 
        The dataset object that contains field objects.
    leave_one_out: bool, default to `True`
        `True`: take the last one as a target
        `False`: take `posItems[1:]` as targets
    """

    def __init__(
        self, source_dp: dp.iter.IterableWrapper,
        dataset: Optional[RecDataSet] = None,
        leave_one_out: bool = True
    ) -> None:
        super().__init__(source_dp)
        self.Item = dataset.fields[ITEM, ID]
        if leave_one_out:
            self.marker = -1
        else:
            self.marker = 1

        self.prepare(dataset)

    @timemeter
    def prepare(self, dataset: RecDataSet):
        pass

    def _check(self, seq) -> bool:
        return len(seq) > 1

    def __iter__(self):
        for sess, seq in self.source:
            if self._check(seq):
                yield [sess, seq[:-1], seq[self.marker:]]


@dp.functional_datapipe("sess_valid_yielding_")
class SessValidYielder(SessTrainYielder):

    def __init__(
        self, 
        source_dp: dp.iter.IterableWrapper, 
        dataset: RecDataSet
    ) -> None:
        super().__init__(source_dp, dataset, True)

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.seenItems = set()
        for chunk in dataset.train():
            self.listmap(
                lambda item: self.seenItems.add(item),
                chunk[ITEM, ID]
            )

    def _check(self, seq) -> bool:
        # filter out the sequence with a target not appearing in training set
        return len(seq) > 1 and seq[-1] in self.seenItems

    def __iter__(self):
        for sess, seq in self.source:
            if self._check(seq):
                # (user, seqs, unseen, seen)
                yield [sess, seq[:-1], seq[-1:], seq[:-1]]


@dp.functional_datapipe("sess_test_yielding_")
class SessTestYielder(SessValidYielder): ...


@dp.functional_datapipe("sess_train_uniform_sampling_")
class SessTrainUniformSampler(SessTrainYielder):
    r"""
    A functional datapipe for uniformly sampling negatives for each sequence.

    Parameters:
    -----------
    source_dp: dp.iter.IterableWrapper 
        A datapipe that yields users.
    dataset: RecDataSet 
        The dataset object that contains field objects.
    num_negatives: int 
        The number of negative samples for each piece of data.  
    leave_one_out: bool, default to `True`
        `True`: take the last one as a target
        `False`: take `posItems[1:]` as targets
    """

    @timemeter
    def prepare(self, dataset: RecDataSet):
        r"""
        Prepare the data before sampling.

        Parameters:
        -----------
        dataset: RecDataSet 
            The dataset object that contains field objects.
        """
        self.negative_pool = self._sample_from_all(dataset.train().datasize)

    def _sample_neg(self, seen: Tuple, positives: Tuple) -> List[int]:
        r"""Randomly sample negative items for a user.

        Parameters:
        ----------
        seen: Tuple 
            A sequence of seen items.
        positives: Tuple
            A sequence of positive items.

        Returns:
        --------
        negatives: List[int] 
            A list of negative items that the user has not interacted with.
        """
        # sorting for ordered positives
        seen = sorted(seen)
        return negsamp_vectorized_bsearch(
            seen, self.Item.count, len(positives)
        )

    def __iter__(self):
        for sess, seq in self.source:
            if self._check(seq):
                seen = seq[:-1]
                positives = seq[self.marker:]
                negatives = self._sample_neg(seq, positives)
                yield [sess, seen, positives, negatives]


@dp.functional_datapipe("sess_valid_sampling_")
class SessValidSampler(SessTrainYielder):

    def __init__(
        self, 
        source_dp: dp.iter.IterableWrapper, 
        dataset: Optional[RecDataSet] = None
    ) -> None:
        super().__init__(source_dp, dataset, True)

    def _sample_negs(self, sess: int, seq: Tuple):
        idx = (sess, tuple(seq))
        if self.negItems.get(idx, None) is None:
            seen = sorted(self.seenItems[sess])
            self.negItems[idx] = tuple(
                negsamp_vectorized_bsearch(
                    seen, self.Item.count, NUM_NEGS_FOR_SAMPLE_BASED_RANKING
                ).tolist()
            )
        return self.negItems[idx]

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.seenItems = defaultdict(set)

        for chunk in dataset.valid():
            self.listmap(
                lambda id_, item: self.seenItems[id_].add(item),
                chunk[SESSION, ID], chunk[ITEM, ID]
            )

        self.negItems = dict()
        for key in self.seenItems:
            self.seenItems[key] = tuple(self.seenItems[key])

    def __iter__(self):
        for sess, seq in self.source:
            if self._check(seq):
                seen = seq[:-1]
                posItem = seq[-1]
                yield [sess, seen, (posItem,) + self._sample_negs(sess, seq)]


@dp.functional_datapipe("sess_test_sampling_")
class SessTestSampler(SessValidSampler):

    @timemeter
    def prepare(self, dataset: RecDataSet):
        self.seenItems = defaultdict(set)

        for chunk in dataset.test():
            self.listmap(
                lambda id_, item: self.seenItems[id_].add(item),
                chunk[SESSION, ID], chunk[ITEM, ID]
            )

        self.negItems = dict()
        for key in self.seenItems:
            self.seenItems[key] = tuple(self.seenItems[key])