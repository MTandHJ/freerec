

from typing import Dict

import torch

from .base import RecSysArch

from ..data.fields import Tokenizer, SparseField
from ..data.tags import ID, ITEM, SPARSE, DENSE, FEATURE, USER
from ..data.postprocessing import Postprocessor


__all__ = ['LightGCN']


class LightGCN(RecSysArch):


    def __init__(
        self, tokenizer: Tokenizer, 
        datapipe: Postprocessor, num_layers: int = 3
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.User: SparseField = self.tokenizer.groupby((USER, ID))[0]
        self.Item: SparseField = self.tokenizer.groupby((ITEM, ID))[0]
        self.register_buffer('Graph', datapipe.getSparseGraph())
        self.num_layers = num_layers

    def preprocess(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]):
        users, items = users[self.User.name], items[self.Item.name]
        m, n = items.size()
        users = users.repeat((1, n))
        return users, items

    def oriEmbeddings(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]):
        users, items = self.preprocess(users, items)
        userEmbs = self.User.look_up(users).flatten(end_dim=-2) # B x 2 x D 
        itemEmbs = self.Item.look_up(items).flatten(end_dim=-2) # B x 2 x D
        return userEmbs, itemEmbs

    def roll(self):
        userEmbs = self.User.look_up(torch.arange(0, self.User.count).to(self.device))
        itemEmbs = self.Item.look_up(torch.arange(0, self.Item.count).to(self.device))
        curEmbs = torch.cat((userEmbs, itemEmbs), dim=0) 
        allEmbs = [curEmbs]
        for _ in range(self.num_layers):
            curEmbs = self.Graph.mm(curEmbs)
            allEmbs.append(curEmbs)
        allEmbs = torch.stack(allEmbs, dim=1) # N x L x D
        allEmbs = allEmbs.mean(dim=1)
        userEmbs, itemEmbs = torch.split(allEmbs, (self.User.count, self.Item.count))
        return userEmbs, itemEmbs

    def forward(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]):
        userEmbs, itemEmbs = self.roll()
        users, items = self.preprocess(users, items)
        users = userEmbs[users] # B x K x D
        items = itemEmbs[items] # B x K x D
        return torch.mul(users, items).sum(-1)
        
    def getRatings(self) -> torch.Tensor:
        userEmbs, itemEmbs = self.roll() # M x D | N x D
        ratings = userEmbs @ itemEmbs.T
        return ratings
