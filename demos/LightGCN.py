

from typing import Dict

import torch
import torchdata.datapipes as dp
import numpy as np
import random

from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import LightGCN
from freerec.criterions import BPRLoss
from freerec.data.datasets import GowallaM1, Postprocessor
from freerec.data.fields import Tokenizer
from freerec.data.tags import FEATURE, SPARSE, USER, ITEM, ID
from freerec.utils import timemeter


@dp.functional_datapipe("graph_")
class Grapher(Postprocessor):


    def __init__(self, datapipe: Postprocessor) -> None:
        super().__init__(datapipe)

        self.User = next(filter(lambda field: field.match([USER, ID]), self.fields))
        self.Item = next(filter(lambda field: field.match([ITEM, ID]), self.fields))
        self.pool_size = 99
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

    def getSparseGraph(self):
        self.train()
        allUsers = []
        allItems = []
        for df in self.source:
            allUsers.append(torch.tensor(df[self.User.name].values.tolist()))
            allItems.append(torch.tensor(df[self.Item.name].values.tolist()))
        allUsers = torch.cat(allUsers)
        allItems = torch.cat(allItems)
        n = self.User.count + self.Item.count
        x = torch.cat([allUsers, allItems + self.User.count, torch.arange(0, n)])
        y = torch.cat([allItems + self.User.count, allUsers, torch.arange(0, n)])
        indices = torch.stack((x, y), dim=0)
        graph = torch.sparse.IntTensor(
            indices,
            torch.ones(indices.size(-1)).int(),
            (n, n)
        )
        dense = graph.to_dense()
        D = torch.sum(dense, dim = 1).float()
        D[D==0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        indices = dense.nonzero()
        data  = dense[dense >= 1e-9]
        assert len(indices) == len(data)
        return torch.sparse_coo_tensor(indices.T, data, (n, n)).coalesce()

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
                return user, self.byrow[user]
           

class Wrapper(Postprocessor):

    def __init__(self, datapipe: Postprocessor, batch_size: int) -> None:
        super().__init__(datapipe)
        self.trainpipe = datapipe.chunk_(batch_size).dict_().tensor_().group_()
        self.otherpipe = datapipe

    def __iter__(self):
        if self.mode == "train":
            yield from self.trainpipe
        else:
            yield from self.otherpipe


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
        for user, items in self.dataloader:
            targets = items.to(self.device)
            preds = Ratings[user]

            self.monitor(
                preds, targets, 
                n=preds.size(0), mode="mean", prefix=prefix,
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

