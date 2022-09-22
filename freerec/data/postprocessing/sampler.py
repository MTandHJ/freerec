

from typing import Iterator, Dict

import torchdata.datapipes as dp
import numpy as np
import random

from .base import Postprocessor, ModeError
from ..fields import SparseField
from ..tags import USER, ITEM, ID
from ...utils import errorLogger, timemeter


__all__ = ['NegativesForTrain', 'NegativesForEval']


@dp.functional_datapipe("negatives_for_train_")
class NegativesForTrain(Postprocessor):
    """Sampling negatives for trainpipe."""

    def __init__(
        self, datapipe: Postprocessor, num_negatives: int = 1
    ) -> None:
        super().__init__(datapipe)
        """
        num_negatives: for training, sampling from all negative items
        """
        self.num_negatives = num_negatives
        self.User: SparseField = self.fields.whichis(USER, ID)
        self.Item: SparseField = self.fields.whichis(ITEM, ID)
        self.prepare()

    @timemeter("NegativeForTrain/prepare")
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
            negItems = list(allItems - items)
            self.negItems.append(negItems)
        

    def sample(self, x):
        return random.sample(self.negItems[x.item()], k=self.num_negatives)

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
            negItems = list(allItems - items)
            self.negItems.append(random.sample(negItems, k=self.num_negatives))

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
