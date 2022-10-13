

from typing import Iterator, Dict

import torchdata.datapipes as dp
import numpy as np
import random
import scipy.sparse as sp
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
        self.prepare()

    @timemeter("NegativeForTrain/prepare")
    def prepare(self):
        self.train()
        self.posItems = [set() for _ in range(self.User.count)]
        self.allItems = set(range(self.Item.count))

        for chunk in self.source:
            list(map(
                lambda row: self.posItems[row[0].item()].add(row[1].item()),
                zip(chunk[self.User.name], chunk[self.Item.name])
            ))

    def sample(self, x):
        seen = self.posItems[x.item()]
        unseen = list(self.allItems - seen)
        return random.choices(unseen, k=self.num_negatives)

    def __iter__(self) -> Iterator:
        if self.mode == 'train':
            for chunk in self.source:
                if self.unseen_only:
                    negatives = self.at_least_2d(
                        np.apply_along_axis(self.sample, 1, chunk[self.User.name])
                    )
                else:
                    negatives = np.random.randint(
                        0, self.Item.count, 
                        size=(len(chunk[self.User.name]), self.num_negatives)
                    )
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
        self.negItems = np.array(self.negItems)

    def __iter__(self) -> Iterator:
        if self.mode in ('valid', 'test'):
            for chunk in self.source:
                negatives = self.negItems[np.ravel(chunk[self.User.name])]
                chunk[self.Item.name] = np.concatenate([chunk[self.Item.name], negatives], axis=1)
                yield chunk
        else:
            raise ModeError(errorLogger("for evaluation only ..."))


@dp.functional_datapipe("uniform_sampling_")
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
        self.fields = self.fields.groupby(ID)
        self.prepare()

    @timemeter("NegativeForEval/prepare")
    def prepare(self):
        self.train()
        self.posItems = [set() for _ in range(self.User.count)]
        self.allItems = set(range(self.Item.count))

        for chunk in self.source:
            list(map(
                lambda row: self.posItems[row[0].item()].add(row[1].item()),
                zip(chunk[self.User.name], chunk[self.Item.name])
            ))

        self.posItems = [list(items) for items in self.posItems]

    def sample_from_seen(self, user):
        return random.choice(self.posItems[user.item()])

    def sample_from_unseen(self, user):
        seen = self.posItems[user.item()]
        unseen = list(self.allItems - set(seen))
        return random.choices(unseen, k=self.num_negatives)

    def __len__(self):
        return 1

    def __iter__(self):
        if self.mode == 'train':
            users = np.random.randint(0, self.User.count, (self.datasize, 1))
            positives = self.at_least_2d(np.apply_along_axis(self.sample_from_seen, 1, users))
            if self.unseen_only:
                negatives = self.at_least_2d(
                    np.apply_along_axis(self.sample_from_unseen, 1, users)
                )
            else:
                negatives = np.random.randint(
                    0, self.Item.count, 
                    size=(self.datasize, self.num_negatives)
                )
            yield {self.User.name: users, self.Item.name: np.concatenate((positives, negatives), axis=1)}
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
        users, items, vals = [], [], []

        self.train()
        for chunk in self.source:
            users.append(chunk[self.User.name])
            items.append(chunk[self.Item.name])
            # -1 for seen positives
            vals.append(-np.ones_like(chunk[self.Item.name])) 

        self.test()
        for chunk in self.source:
            users.append(chunk[self.User.name])
            items.append(chunk[self.Item.name])
            # 1 for unseen positives
            vals.append(np.ones_like(chunk[self.Item.name]))
        
        users = np.ravel(np.concatenate(users, axis=0))
        items = np.ravel(np.concatenate(items, axis=0))
        vals = np.ravel(np.concatenate(vals, axis=0))

        self.graph = sp.csr_array(
            (vals, (users, items)), dtype=np.int64
        )


    def __len__(self):
        return ceil(self.User.count / self.batch_size)

    def __iter__(self):
        if self.mode == 'train':
            raise ModeError(errorLogger("for evaluation only ..."))
        else:
            allUsers = np.arange(self.User.count)
            for k in range(len(self)):
                start = k * self.batch_size
                end = (k + 1) * self.batch_size
                users = allUsers[start:end]
                items = self.graph[users].todense()
                yield {self.User.name: users[:, None], self.Item.name: items}
