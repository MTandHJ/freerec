

from typing import Iterator, Dict

import torch
import torchdata.datapipes as dp
import numpy as np
import random
from math import ceil

from .base import Postprocessor, ModeError
from ..fields import SparseField
from ..tags import USER, ITEM, ID
from ...utils import errorLogger, timemeter


__all__ = ['NegativesForTrain', 'NegativesForEval', 'UniformSampler', 'TriSampler']


@dp.functional_datapipe("negatives_for_train_")
class NegativesForTrain(Postprocessor):
    """Sampling negatives for trainpipe."""

    def __init__(
        self, datapipe: Postprocessor, num_negatives: int = 1
    ) -> None:
        """
        Parameters:
        ---

        datapipe: RecDataSet or Postprocessor
            Yielding dict of np.array.
        num_negatives: int
            Sampling `num_negatives` for every piece of data.
        """
        super().__init__(datapipe)
        self.num_negatives = num_negatives
        self.User: SparseField = self.fields[USER, ID]
        self.Item: SparseField = self.fields[ITEM, ID]
        self.prepare()

    @timemeter("NegativeForTrain/prepare")
    def prepare(self):
        self.train()
        posItems = [set() for _ in range(self.User.count)]
        allItems = set(range(self.Item.count))
        self.unseen = []

        for chunk in self.source:
            list(map(
                lambda row: posItems[row[0].item()].add(row[1].item()),
                zip(chunk[self.User.name], chunk[self.Item.name])
            ))
        for items in posItems:
            self.unseen.append(list(allItems - items))
        

    def sample(self, x):
        return random.sample(self.unseen[x.item()], k=self.num_negatives)

    def __iter__(self) -> Iterator:
        if self.mode == 'train':
            for chunk in self.source:
                negatives = self.at_least_2d(np.apply_along_axis(self.sample, 1, chunk[self.User.name]))
                chunk[self.Item.name] = np.concatenate([chunk[self.Item.name], negatives], axis=1)
                yield chunk
        else:    
            raise ModeError(errorLogger("for training only ..."))
        

@dp.functional_datapipe("negatives_for_eval_")
class NegativesForEval(NegativesForTrain):
    """Sampling negatives for valid|testpipe."""

    def __init__(self, datapipe: Postprocessor, num_negatives: int = 1) -> None:
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

    @timemeter("NegativeForEval/prepare")
    def prepare(self):
        self.train()
        posItems = [set() for _ in range(self.User.count)]
        allItems = set(range(self.Item.count))
        self.negItems = []

        for chunk in self.source:
            list(map(
                lambda row: posItems[row[0].item()].add(row[1].item()),
                zip(chunk[self.User.name], chunk[self.Item.name])
            ))
        for items in posItems:
            unseen = list(allItems - items)
            self.negItems.append(random.sample(unseen, k=self.num_negatives))

    def sample(self, x):
        return self.negItems[x.item()]

    def __iter__(self) -> Iterator:
        if self.mode in ('valid', 'test'):
            for chunk in self.source:
                negatives = self.at_least_2d(np.apply_along_axis(self.sample, 1, chunk[self.User.name]))
                chunk[self.Item.name] = np.concatenate([chunk[self.Item.name], negatives], axis=1)
                yield chunk
        else:
            raise ModeError(errorLogger("for evaluation only ..."))


@dp.functional_datapipe("uniform_sampling_")
class UniformSampler(Postprocessor):
    """Uniformly sampling users and their negatives."""

    def __init__(
        self, datapipe: Postprocessor, num_negatives: int = 1
    ) -> None:
        """
        Parameters:
        ---

        datapipe: RecDataSet or Postprocessor
            Yielding dict of np.array.
        num_negatives: int
            Sampling `num_negatives` for every piece of data.
        """
        super().__init__(datapipe)
        self.num_negatives = num_negatives
        self.User: SparseField = self.fields[USER, ID]
        self.Item: SparseField = self.fields[ITEM, ID]
        self.fields = self.fields.groupby(ID)
        self.prepare()

    @timemeter("NegativeForEval/prepare")
    def prepare(self):
        self.train()
        self.posItems = [set() for _ in range(self.User.count)]

        for chunk in self.source:
            list(map(
                lambda row: self.posItems[row[0].item()].add(row[1].item()),
                zip(chunk[self.User.name], chunk[self.Item.name])
            ))

        self.posItems = [list(items) for items in self.posItems]

    def sample(self, user):
        user = user.item()
        posItems = self.posItems[user]
        posItem = random.choice(posItems)
        negItem = posItems[0]
        while negItem in posItems:
            negItem = random.randint(0, self.Item.count - 1)
        return [posItem, negItem]

    def __len__(self):
        return 1

    def __iter__(self):
        if self.mode == 'train':
            users = np.random.randint(0, self.User.count, self.datasize)
            negatives = np.apply_along_axis(self.sample, 1, users[:, None])
            yield {self.User.name: self.at_least_2d(users), self.Item.name: negatives}
        else:
            raise ModeError(errorLogger("for training only ..."))


@dp.functional_datapipe("trisample_")
class TriSampler(Postprocessor):
    """Yielding users with all items.
    The items are trinary including -1, 0, 1:
        - `-1`: This item has been used in trainset;
        - `0`: A negative item.
        - `1`: A positive item has not used in trainset.
    """

    def __init__(
        self, datapipe: Postprocessor, batch_size: int
    ) -> None:
        """
        Parameters:
        ---

        datapipe: RecDataSet or Postprocessor
            Yielding dict of np.array.
        batch_size: int
        """
        super().__init__(datapipe)
        self.batch_size = batch_size
        self.User: SparseField = self.fields[USER, ID]
        self.Item: SparseField = self.fields[ITEM, ID]
        self.fields = self.fields.groupby(ID)
        self.prepare()

    @timemeter("NegativeForEval/prepare")
    def prepare(self):
        self.seen = [set() for _ in range(self.User.count)]
        self.posItems = [set() for _ in range(self.User.count)]

        self.train()
        for chunk in self.source:
            list(map(
                lambda row: self.seen[row[0].item()].add(row[1].item()),
                zip(chunk[self.User.name], chunk[self.Item.name])
            ))

        self.test()
        for chunk in self.source:
            list(map(
                lambda row: self.posItems[row[0].item()].add(row[1].item()),
                zip(chunk[self.User.name], chunk[self.Item.name])
            ))
        self.seen = [list(elem) for elem in self.seen]
        self.posItems = [list(elem) for elem in self.posItems]

    def sample(self, user):
        user = user.item()
        seen = self.seen[user]
        posItems = self.posItems[user]
        items = np.zeros((self.Item.count,))
        items[seen] = -1
        items[posItems] = 1
        return items

    def __len__(self):
        return ceil(self.User.count / self.batch_size)

    def __iter__(self):
        if self.mode == 'train':
            raise ModeError(errorLogger("for evaluation only ..."))
        else:
            allUsers = np.arange(self.User.count)[:, None]
            for k in range(len(self)):
                start = k * self.batch_size
                end = (k + 1) * self.batch_size
                users = allUsers[start:end]
                items = np.apply_along_axis(self.sample, 1, users)
                yield {self.User.name: users, self.Item.name: items}
