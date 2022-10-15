

from typing import Dict, Optional, Union

import torch
import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from torch_geometric.nn import LGConv

from .base import RecSysArch

from ..data.fields import Tokenizer
from ..data.tags import ID, ITEM, USER


__all__ = ['LightGCN']


class LightGCN(RecSysArch):

    def __init__(
        self, tokenizer: Tokenizer, 
        graph: Data,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.conv = LGConv()
        self.num_layers = num_layers
        self.graph = graph

        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]

        self.initialize()

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph
        T.ToSparseTensor()(self.__graph)

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device:
            self.graph.to(device)
        return super().to(device, dtype, non_blocking)

    def forward(
        self, users: Optional[Dict[str, torch.Tensor]] = None, 
        items: Optional[Dict[str, torch.Tensor]] = None
    ):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.conv(features, self.graph.adj_t)
            avgFeats += features / (self.num_layers + 1)
        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))

        if self.training: # Batch
            users, items = self.broadcast(
                users[self.User.name], items[self.Item.name]
            )
            userFeats = userFeats[users] # B x n x D
            itemFeats = itemFeats[items] # B x n x D
            userEmbs = self.User.look_up(users) # B x n x D
            itemEmbs = self.Item.look_up(items) # B x n x D
            return torch.mul(userFeats, itemFeats).sum(-1), userEmbs, itemEmbs
        else:
            return userFeats, itemFeats
