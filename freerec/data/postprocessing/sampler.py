

from typing import Iterator

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
        self.sample_negatives()

    @timemeter("NegativeForTrain/sample_negatives")
    def sample_negatives(self):
        self.train()
        posItems = [set() for _ in range(self.User.count)]
        allItems = set(range(self.Item.count))
        self.negItems = []
        for df in self.source:
            df = df[[self.User.name, self.Item.name]]
            for idx, items in df.groupby(self.User.name).agg(set).iterrows():
                posItems[idx] |= set(*items)
        for items in posItems:
            self.negItems.append(list(allItems - items))

    def __iter__(self) -> Iterator:
        if self.mode == 'train':
            for df in self.source:
                negs = np.stack(
                    df.agg(
                        lambda row: random.sample(self.negItems[int(row[self.User.name])], k=self.num_negatives),
                        axis=1
                    ),
                    axis=0
                )
                df[self.Item.name] = np.concatenate((df[self.Item.name].values[:, None], negs), axis=1).tolist()
                yield df
        else:    
            raise ModeError(errorLogger("for training only ..."))
        

@dp.functional_datapipe("negatives_for_eval_")
class NegativesForEval(NegativesForTrain):

    @timemeter("NegativeForEval/sample_negatives")
    def sample_negatives(self):
        self.train()
        posItems = [set() for _ in range(self.User.count)]
        allItems = set(range(self.Item.count))
        self.negItems = []
        for df in self.source:
            df = df[[self.User.name, self.Item.name]]
            for idx, items in df.groupby(self.User.name).agg(set).iterrows():
                posItems[idx] |= set(*items)
        for items in posItems:
            negItems = list(allItems - items)
            self.negItems.append(random.sample(negItems, k = self.num_negatives))

    def __iter__(self) -> Iterator:
        if self.mode in ('valid', 'test'):
            for df in self.source:
                negs = np.stack(
                    df.agg(
                        lambda row: self.negItems[int(row[self.User.name])],
                        axis=1
                    ),
                    axis=0
                )
                df[self.Item.name] = np.concatenate((df[self.Item.name].values[:, None], negs), axis=1).tolist()
                yield df
        else:
            raise ModeError(errorLogger("for evaluation only ..."))
