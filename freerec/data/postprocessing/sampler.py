

from typing import List, Tuple

import random
import torchdata.datapipes as dp

from .base import Postprocessor
from ..datasets.base import RecDataSet
from ..fields import SparseField
from ..tags import USER, ITEM, ID
from ...utils import timemeter


__all__ = ['TrainUniformSampler', 'ValidTripleter', 'TestTripleter']


@dp.functional_datapipe("train_uniform_sampling_")
class TrainUniformSampler(Postprocessor):
    """
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
        super().__init__(source_dp)
        self.num_negatives = num_negatives
        self.User: SparseField = dataset.fields[USER, ID]
        self.Item: SparseField = dataset.fields[ITEM, ID]
        self.prepare(dataset)

    @timemeter("UniformSampler/prepare")
    def prepare(self, dataset: RecDataSet):
        """
        Prepare the data before sampling.

        Parameters:
        -----------
        dataset: RecDataSet 
            The dataset object that contains field objects.
        """
        self.posItems = [set() for _ in range(self.User.count)]
        self.negative_pool = self._sample_from_all(dataset.datasize * self.num_negatives)
        for chunk in dataset.train():
            self.listmap(
                lambda user, item: self.posItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )
        self.posItems = [tuple(items) for items in self.posItems]

    def _sample_from_all(self, pool_size: int = 51200):
        """
        Randomly sample items from all items.

        Parameters:
        -----------
        pool_size: int 
            The number of items to be sampled.

        Returns:
        --------
        Generator: A generator that yields sampled items.
        """
        allItems = self.Item.ids
        while 1:
            for item in random.choices(allItems, k=pool_size):
                yield item

    def _sample_from_pool(self, seen: Tuple):
        """
        Randomly sample a negative item from the pool of all items.

        Parameters:
        -----------
        seen: set 
            A set of seen items.

        Returns:
        --------
        negative: int 
            A negative item that has not been seen.
        """
        negative = next(self.negative_pool)
        while negative in seen:
            negative = next(self.negative_pool)
        return negative

    def _sample_pos(self, user: int) -> int:
        """
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

    def _sample_neg(self, user: int) -> List[int]:
        """Randomly sample negative items for a user.

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
        return self.listmap(self._sample_from_pool, [seen] * self.num_negatives)

    def __iter__(self):
        for user in self.source:
            yield user, self._sample_pos(user), self._sample_neg(user)


@dp.functional_datapipe("valid_triplet_")
class ValidTripleter(Postprocessor):
    """
    A datapipe that yields (user, unseen, seen) triplets.
    The ValidTripleter postprocessor takes a RecDataSet as input,
    and yields (user, unseen, seen) triplets. The unseen and seen sets contain the IDs of
    the items that the user has not seen and seen, respectively. Whether to use validation
    or test set as the source of unseen items depends on the value of `dataset.VALID_IS_TEST`.
    """

    def __init__(
        self, source_dp: dp.iter.IterDataPipe,
        dataset: RecDataSet
    ) -> None:
        """
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

    @timemeter("ValidTripleter/prepare")
    def prepare(self, dataset: RecDataSet):
        """
        Prepares the dataset by building sets of seen items for each user.

        Parameters:
        -----------
        dataset: RecDataSet 
            The dataset that provides the data source.
        """
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

    def __iter__(self):
        """
        Yields:
        -------
        user, unseen, seen: int, List[int], List[int]
            Triplets for each user in the data source.
        """
        for user in self.source:
            yield user, self.unseenItems[user], self.seenItems[user]


@dp.functional_datapipe("test_triplet_")
class TestTripleter(ValidTripleter):
    """
    A datapipe that yields (user, unseen, seen) triplets from the test set.
    The TestTriplet postprocessor takes a RecDataSet as input,
    and yields (user, unseen, seen) triplets from the test set. The unseen and seen sets contain the IDs of
    the items that the user has not seen and seen, respectively.
    """

    @timemeter("TestTripleter/prepare")
    def prepare(self, dataset: RecDataSet):
        """
        Prepares the dataset by building sets of seen items for each user.

        Parameters:
        -----------
        dataset: RecDataSet 
            The dataset that provides the data source.
        """
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
