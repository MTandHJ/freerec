

from typing import Iterator

import random
import torchdata.datapipes as dp

from .base import Postprocessor, ModeError
from ..fields import SparseField
from ..tags import USER, ITEM, ID, POSITIVE, NEGATIVE, SEEN, UNSEEN
from ...utils import timemeter


__all__ = ['NegativesForTrain', 'NegativesForEval', 'UniformSampler', 'Tripleter']


@dp.functional_datapipe("train_user_pos_negs_")
class NegativesForTrain(Postprocessor):
    """Sampling negatives for trainpipe."""

    def __init__(
        self, datapipe: Postprocessor, 
        num_negatives: int = 1,
        unseen_only: bool = True
    ) -> None:
        """
        Parameters:
        ---

        datapipe: RecDataSet or Postprocessor
            Yielding dict of np.array.
        num_negatives: int
            Sampling `num_negatives` for every piece of data.
        unseen_only: bool, default True
            - `True`: Sampling negatives only from unseen items (slow).
            - `False`: Sampling negatives from all items (fast).
        """
        super().__init__(datapipe)
        self.num_negatives = num_negatives
        self.unseen_only = unseen_only
        self.User: SparseField = self.fields[USER, ID]
        self.Item: SparseField = self.fields[ITEM, ID]
        self.Negative = self.Item.buffer(tags=NEGATIVE)
        self.prepare()

    @timemeter("NegativeForTrain/prepare")
    def prepare(self):

        self.posItems = [set() for _ in range(self.User.count)]
        self.negative_pool = self._sample_from_all(self.datasize * self.num_negatives)

        for chunk in self.source.train():
            self.listmap(
                lambda user, item: self.posItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID],
            )

    def _sample_from_all(self, pool_size: int = 51200):
        allItems = self.Item.ids
        while 1:
            for item in random.choices(allItems, k=pool_size):
                yield item

    def _sample_from_pool(self, seen):
        negative = next(self.negative_pool)
        while negative in seen:
            negative = next(self.negative_pool)
        return negative

    def sample_from_unseen(self, x):
        seen = self.posItems[x]
        return self.listmap(self._sample_from_pool, [seen] * self.num_negatives)

    def sample_from_all(self, x):
        return random.choices(self.Item.ids, k=self.num_negatives)

    def forward(self):
        if self.mode == 'train':
            for chunk in self.source:
                if self.unseen_only:
                    negatives = self.listmap(self.sample_from_unseen, chunk[USER, ID])
                else:
                    negatives = self.listmap(self.sample_from_all, chunk[USER, ID])
                yield chunk[USER, ID], chunk[ITEM, ID].buffer(tags=POSITIVE), self.Negative.buffer(negatives)
        else:    
            raise ModeError("for training only ...")


@dp.functional_datapipe("eval_user_pos_negs_")
class NegativesForEval(NegativesForTrain):
    """Sampling negatives for valid|testpipe."""

    def __init__(
        self, datapipe: Postprocessor, num_negatives: int = 100,
    ) -> None:
        """
        Parameters:
        ---

        datapipe: RecDataSet or Postprocessor
            Yielding dict of np.array.
        num_negatives: int
            `num_negatives` negatives will be sampled for every user in advance, 
            and then they will be yieled following their users. 
            Note that all negatives will be frozen once sampled.
        """
        super().__init__(datapipe, num_negatives)
    
    def _prepare_for_valid(self):
        posItems = [set() for _ in range(self.User.count)]
        allItems = set(self.Item.ids)
        self.negItems_for_valid = []

        for chunk in self.source.train():
            self.listmap(
                lambda user, item: posItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        for items in posItems:
            unseen = list(allItems - items)
            self.negItems_for_valid.append(tuple(random.sample(unseen, k=self.num_negatives)))
            # random.sample or random.choices ?

        return posItems, allItems

    def _prepare_for_test(self, posItems: set, allItems: set):
        self.negItems_for_test = []

        for chunk in self.source.valid():
            self.listmap(
                lambda user, item: posItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        for items in posItems:
            unseen = list(allItems - items)
            self.negItems_for_test.append(tuple(random.sample(unseen, k=self.num_negatives)))

    @timemeter("NegativeForEval/prepare")
    def prepare(self):
        if self.VALID_IS_TEST:
            self._prepare_for_valid()
            self.negItems_for_test = self.negItems_for_valid
        else: # validation set is not test set
            self._prepare_for_test(
                *self._prepare_for_valid()
            )

    def sample_for_valid(self, user: int):
        return self.negItems_for_valid[user]

    def sample_for_test(self, user: int):
        return self.negItems_for_test[user]

    def forward(self) -> Iterator:
        if self.mode == 'valid':
            for chunk in self.source:
                negatives = self.listmap(self.sample_for_valid, chunk[USER, ID])
                yield chunk[USER, ID], chunk[ITEM, ID].buffer(tags=POSITIVE), self.Negative.buffer(negatives)
        elif self.mode =='test':
            for chunk in self.source:
                negatives = self.listmap(self.sample_for_test, chunk[USER, ID])
                yield chunk[USER, ID], chunk[ITEM, ID].buffer(tags=POSITIVE), self.Negative.buffer(negatives)
        else:
            raise ModeError("for evaluation only ...")


@dp.functional_datapipe("train_uniform_user_")
class UniformSampler(Postprocessor):
    """Uniformly sampling users and their negatives."""

    def __init__(
        self, datapipe: Postprocessor, 
        num_negatives: int = 1,
        unseen_only: bool = True
    ) -> None:
        """
        Parameters:
        ---

        datapipe: RecDataSet or Postprocessor
            Yielding dict of np.array.
        num_negatives: int
            Sampling `num_negatives` for every piece of data.
        unseen_only: bool
            - `True`: Sampling negatives only from unseen items (slow).
            - `False`: Sampling negatives from all items (fast).
        """
        super().__init__(datapipe)
        self.num_negatives = num_negatives
        self.unseen_only = unseen_only
        self.User: SparseField = self.fields[USER, ID]
        self.Item: SparseField = self.fields[ITEM, ID]
        self.prepare()

    @timemeter("UniformSampler/prepare")
    def prepare(self):
        self.posItems = [set() for _ in range(self.User.count)]
        self.negative_pool = self._sample_from_all(self.datasize * self.num_negatives)

        for chunk in self.source.train():
            self.listmap(
                lambda user, item: self.posItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        self.posItems = [tuple(items) for items in self.posItems]

    def _sample_from_all(self, pool_size: int = 51200):
        allItems = self.Item.ids
        while 1:
            for item in random.choices(allItems, k=pool_size):
                yield item

    def _sample_from_pool(self, seen):
        negative = next(self.negative_pool)
        while negative in seen:
            negative = next(self.negative_pool)
        return negative

    def _sample_pos(self, user):
        return random.choice(self.posItems[user])

    def _sample_neg(self, user):
        seen = self.posItems[user]
        return self.listmap(self._sample_from_pool, [seen] * self.num_negatives)

    def sample_from_unseen(self, user):
        return self._sample_neg(user)

    def sample_from_all(self, user):
        return random.choices(self.Item.ids, k=self.num_negatives)

    def forward(self):
        users = random.choices(self.User.ids, k=self.datasize)
        chunk_size = self.DEFAULT_CHUNK_SIZE
        if self.mode == 'train':
            for start, end in zip(
                range(0, self.datasize, chunk_size), 
                range(chunk_size, self.datasize + chunk_size, chunk_size)
            ):
                chunk_users = users[start:end]
                chunk_pos_items = self.listmap(self._sample_pos, chunk_users)
                if self.unseen_only:
                    chunk_neg_items = self.listmap(self.sample_from_unseen, chunk_users)
                else:
                    chunk_neg_items = self.listmap(self.sample_from_all, chunk_users)
                yield self.User.buffer(chunk_users), \
                    self.Item.buffer(chunk_pos_items, tags=POSITIVE), \
                    self.Item.buffer(chunk_neg_items, tags=NEGATIVE)
        else:
            raise ModeError("for training only ...")


@dp.functional_datapipe("eval_user_unseens_seens_")
class Tripleter(Postprocessor):
    """Yielding [user, unseen, seen]"""

    def __init__(
        self, datapipe: Postprocessor
    ) -> None:
        """
        Parameters:
        ---

        datapipe: RecDataSet or Postprocessor
            Yielding dict of np.array.
        """
        super().__init__(datapipe)
        self.User: SparseField = self.fields[USER, ID]
        self.Item: SparseField = self.fields[ITEM, ID]
        self.prepare()

    @timemeter("Tripleter/prepare")
    def prepare(self):

        self.trainItems = [set() for _ in range(self.User.count)]
        self.validItems = [set() for _ in range(self.User.count)]
        self.testItems = [set() for _ in range(self.User.count)]

        for chunk in self.source.train():
            self.listmap(
                lambda user, item: self.trainItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        for chunk in self.source.valid():
            self.listmap(
                lambda user, item: self.validItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        for chunk in self.source.test():
            self.listmap(
                lambda user, item: self.testItems[user].add(item),
                chunk[USER, ID], chunk[ITEM, ID]
            )

        self.trainItems = [tuple(items) for items in self.trainItems]
        self.validItems = [tuple(items) for items in self.validItems]
        self.testItems = [tuple(items) for items in self.testItems]

    def sample_from_seen(self, user: int):
        if self.mode == 'valid' or self.VALID_IS_TEST:
            return self.trainItems[user] 
        else:
            return self.trainItems['user'] + self.validItems['user']

    def sample_from_unseen(self, user: int):
        return self.validItems[user] if self.mode == 'valid' else self.testItems['user']

    def forward(self):
        if self.mode == 'train':
            raise ModeError("for evaluation only ...")
        else:
            users = list(self.User.ids)
            chunk_size = self.DEFAULT_CHUNK_SIZE
            for start, end in zip(
                range(0, self.User.count, chunk_size), 
                range(chunk_size, self.User.count + chunk_size, chunk_size)
            ):
                chunk_users = users[start:end]
                chunk_unseen_items = self.listmap(
                    self.sample_from_unseen, chunk_users
                )
                chunk_seen_items = self.listmap(
                    self.sample_from_seen, chunk_users
                )
                yield self.User.buffer(chunk_users), \
                    self.Item.buffer(chunk_unseen_items, tags=UNSEEN), \
                    self.Item.buffer(chunk_seen_items, tags=SEEN)
