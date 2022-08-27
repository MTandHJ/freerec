

from email.mime import base
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
        self.posItems = [list(items) for items in self.posItems]
        self.sizes = [len(items) for items in self.posItems]

        self.test()
        for df in self.source:
            df = df[[self.User.name, self.Item.name]]
            for idx, items in df.groupby(self.User.name).agg(set).iterrows():
                self.targetItems[idx] |= set(*items)
        self.targetItems = [list(items) for items in self.targetItems]

    def sample(self, user: int):
        posItems = self.posItems[user]
        posItem = posItems[np.random.randint(0, self.sizes[user])]
        while True:
            negItem = np.random.randint(0, self.Item.count)
            if negItem not in posItems:
                return [user, [posItem, negItem]]

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
            for user in np.random.randint(0, self.User.count, sum(self.sizes)):
                yield self.sample(user)
        else:
            for user in range(self.User.count):
                yield [user, self.targetItems[user], self.posItems[user]]
           

class CoachForLightGCN(Coach):


    def reg_loss(self, users, items):
        userEmbeds, itemEmbeds = self.model.oriEmbeddings(users, items)
        loss = userEmbeds.pow(2).sum() + itemEmbeds.pow(2).sum() * 2
        loss = loss / userEmbeds.size(0)
        return loss / 2

    @timemeter("Coach/train")
    def train(self):
        self.model.train()
        self.datapipe.train()
        for users, posItems, negItems in self.dataloader:
            users = {'UserID': users.view(-1, 1)}
            items = {'ItemID': torch.stack([posItems, negItems], dim=1)}
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}

            preds = self.model(users, items)
            pos, neg = preds[:, 0], preds[:, 1]
            loss = self.criterion(pos, neg) + self.reg_loss(users, items) * self.cfg.weight_decay

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=preds.size(0), mode="mean", prefix='train', pool=['LOSS'])

        self.lr_scheduler.step()

    def evaluate(self, prefix: str = 'valid'):
        self.model.eval()
        Ratings: torch.Tensor = self.model.getRatings() # M x N
        for users, items, posItems in self.dataloader:
            preds = Ratings[users]
            targets = torch.zeros_like(preds)
            n = len(users)
            for k in range(n):
                targets[k][items[k]] = 1
                preds[k][posItems[k]] = -1e10

            self.monitor(
                preds, targets,
                n=len(users), mode="mean", prefix=prefix,
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
        epochs=1000,
        batch_size=2048,
        optimizer='adam',
        lr=1e-3,
        weight_decay=1e-4,
        seed=2020
    )
    cfg.compile()

    basepipe = GowallaM1(cfg.root).graph_()
    User, Item = basepipe.User, basepipe.Item
    trainpipe = basepipe.batch_(cfg.batch_size).dataframe_(columns=[User.name, Item.name]).dict_().tensor_().group_()

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

