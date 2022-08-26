

from typing import Dict

import torch
import torchdata.datapipes as dp
import numpy as np
import scipy.sparse as sp
import random

from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import LightGCN
from freerec.criterions import BPRLoss
from freerec.data.datasets import GowallaM1, Postprocessor
from freerec.data.sparse import get_lil_matrix, sparse_matrix_to_tensor
from freerec.data.fields import Tokenizer
from freerec.data.tags import FEATURE, SPARSE, TARGET, USER, ITEM, ID
from freerec.utils import timemeter


@dp.functional_datapipe("graph_")
class Grapher(Postprocessor):


    def __init__(self, datapipe: Postprocessor) -> None:
        super().__init__(datapipe)

        self.User = next(filter(lambda field: field.match([USER, ID]), self.fields))
        self.Item = next(filter(lambda field: field.match([ITEM, ID]), self.fields))
        self.Val = next(filter(lambda field: field.match(TARGET), self.fields))
        self.parseItems()

    @timemeter("Grapher/parseItems")
    def parseItems(self):
        self.train()
        self.posItems = [set() for _ in range(self.User.count)]
        self.allItems = set(range(self.Item.count))
        self.negItems = []
        self.targetItems = [set() for _ in range(self.User.count)]
        for df in self.source:
            df = df[[self.User.name, self.Item.name]]
            for idx, items in df.groupby(self.User.name).agg(set).iterrows():
                self.posItems[idx] |= set(*items)

        self.test()
        for df in self.source:
            df = df[[self.User.name, self.Item.name]]
            for idx, items in df.groupby(self.User.name).agg(set).iterrows():
                self.targetItems[idx] |= set(*items)
        self.targetItems = [list(items) for items in self.targetItems]

    def sample(self, user: int):
        posItems = self.posItems[user]
        while True:
            negItem = random.randint(0, self.Item.count - 1)
            if negItem not in posItems:
                return [negItem]

    def byrow(self, user: int):
        labels = torch.zeros(self.Item.count, dtype=torch.long)
        labels[self.targetItems[user]] = 1
        return labels

    @timemeter("Grapher/getSparseGraph")
    def getSparseGraph(self):
        self.train()
        R = get_lil_matrix(
            self.source, self.User.name, self.Item.name, self.Val.name,
            shape=(self.User.count, self.Item.count)
        ) # ndarray: M x N
        n = self.User.count + self.Item.count
        A = sp.lil_array((n, n), dtype=np.float32)
        A[:self.User.count, self.User.count:] = R
        A[self.User.count:, :self.User.count] = R.T
        A = A.todok()

        D = A.sum(axis=1)
        D = np.power(D, -0.5)
        D[np.isinf(D)] = 0.
        D = D.reshape(-1, 1) # n x 1
        A = D * A
        A = A * D.T
        return sparse_matrix_to_tensor(A)

    def __iter__(self):
        if self.mode == 'train':
            for df in self.source:
                negs = np.stack(
                    df.agg(
                        lambda row: self.sample(int(row[self.User.name])),
                        axis=1
                    ),
                    axis=0
                )
                df[self.Item.name] = np.concatenate((df[self.Item.name].values[:, None], negs), axis=1).tolist()
                yield df
        else:
            for user in range(self.User.count):
                yield user, self.byrow(user), torch.tensor(list(self.posItems[user]))
           

class Wrapper(Postprocessor):

    def __init__(self, datapipe: Postprocessor, batch_size: int) -> None:
        super().__init__(datapipe)
        self.trainpipe = datapipe.chunk_(batch_size).dict_().tensor_().group_()
        self.otherpipe = datapipe.batch(batch_size)

    def __iter__(self):
        if self.mode == "train":
            yield from self.trainpipe
        else:
            for batch in self.otherpipe:
                users = torch.tensor([data[0] for data in batch], dtype=torch.long)
                items = torch.stack([data[1] for data in batch], axis=0)
                posItems = [data[2] for data in batch]
                yield users, items, posItems


class CoachForLightGCN(Coach):


    @timemeter("Coach/train")
    def train(self):
        self.model.train()
        self.datapipe.train()
        users: Dict[str, torch.Tensor]
        items: Dict[str, torch.Tensor]
        targets: torch.Tensor
        for users, items, targets in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}

            preds = self.model(users, items)
            pos, neg = preds[:, 0], preds[:, 1]
            loss = self.criterion(pos, neg)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=preds.size(0), mode="mean", prefix='train', pool=['LOSS'])

        self.lr_scheduler.step()


    def evaluate(self, prefix: str = 'valid'):
        self.model.eval()
        Ratings: torch.Tensor = self.model.getRatings() # M x N
        for users, items, posItems in self.dataloader:
            targets = items.to(self.device)
            preds = Ratings[users.to(self.device)]
            for k, pos in enumerate(posItems):
                preds[k][pos] = -1e10

            self.monitor(
                preds, targets,
                n=users.size(0), mode="mean", prefix=prefix,
                pool=['NDCG', 'PRECISION', 'RECALL', 'HITRATE', 'MSE', 'MAE', 'RMSE']
            )



def main():

    cfg = Parser()
    cfg.add_argument("-eb", "--embedding_dim", type=int, default=64)
    cfg.add_argument("--layers", type=int, default=3)
    cfg.set_defaults(
        fmt="{description}={embedding_dim}={optimizer}-{lr}-{weight_decay}={seed}",
        description="NeuCF",
        root="../gowalla",
        epochs=20,
        batch_size=256,
        optimizer='adam',
        lr=1e-3,
        weight_decay=1e-4,
        seed=2020
    )
    cfg.compile()

    basepipe = GowallaM1(cfg.root)
    datapipe = basepipe.pin_(buffer_size=cfg.buffer_size).shard_().graph_()
    dataset = Wrapper(datapipe, batch_size=cfg.batch_size)

    tokenizer = Tokenizer(datapipe.fields)
    tokenizer.embed(
        dim=cfg.embedding_dim, tags=(FEATURE, SPARSE)
    )
    model = LightGCN(tokenizer, datapipe).to(cfg.DEVICE)

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2)
        )
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    criterion = BPRLoss()
    criterion.regulate(tokenizer.parameters(), rtype='l2', weight=cfg.weight_decay)

    coach = CoachForLightGCN(
        model=model,
        datapipe=dataset,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=cfg.DEVICE
    )
    coach.compile(cfg, monitors=['loss', 'precision@10', 'recall@10', 'hitrate@10', 'ndcg@10', 'ndcg@20'])
    coach.fit()

if __name__ == "__main__":
    main()

