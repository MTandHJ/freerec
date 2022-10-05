

from typing import Dict, Optional

import torch
import torch.nn as nn
import dgl
import dgl.function as dglfn

from .base import RecSysArch

from ..data.fields import Tokenizer, SparseField
from ..data.tags import ID, ITEM, SPARSE, DENSE, FEATURE, USER
from ..utils import timemeter


__all__ = ['LightGCN']


class LightConv(nn.Module):


    def __init__(self, graph: dgl.DGLGraph, norm: str = 'both') -> None:
        super().__init__()

        self.norm = norm
        self.graph = graph

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph):
        self.__graph = dgl.add_self_loop(graph)

    def forward(self, features: torch.Tensor):
        with self.graph.local_scope():
            self.graph.to(features.device)
            if self.norm in ['left', 'both']:
                degs = self.graph.out_degrees().float().clamp(min=1)
                if self.norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1. / degs
                features = features * norm.view(-1, 1)
                
            self.graph.srcdata['input'] = features
            self.graph.update_all(
                message_func=dglfn.copy_src('input', 'hidden'),
                reduce_func=dglfn.sum('hidden', 'output')
            )
            features = self.graph.dstdata['output']
            if self.norm in ['left', 'both']:
                degs = self.graph.in_degrees().float().clamp(min=1)
                if self.norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1. / degs
                features = features * norm.view(-1, 1)
            return features

        
class LightGCN(RecSysArch):


    def __init__(
        self, tokenizer: Tokenizer,
        graph: dgl.DGLGraph, num_layers: int = 3
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.User: SparseField = self.tokenizer.groupby(USER, ID)[0]
        self.Item: SparseField = self.tokenizer.groupby(ITEM, ID)[0]
        self.lightconv = LightConv(graph)
        self.num_layers = num_layers

    def preprocess(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]):
        users, items = users[self.User.name], items[self.Item.name]
        m, n = items.size()
        users = users.repeat((1, n))
        return users, items

    def forward(
        self, users: Optional[Dict[str, torch.Tensor]] = None, 
        items: Optional[Dict[str, torch.Tensor]] = None
    ):
        userEmbs = self.User.look_up(torch.arange(0, self.User.count).to(self.device))
        itemEmbs = self.Item.look_up(torch.arange(0, self.Item.count).to(self.device))
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        for _ in range(self.num_layers):
            features = self.lightconv(features)
        userEmbs, itemEmbs = torch.split(features, (self.User.count, self.Item.count))
        if self.training:
            # Batch
            users, items = self.preprocess(users, items)
            users = userEmbs[users] # B x n x D
            items = itemEmbs[items] # B x n x D
            return torch.mul(users, items).sum(-1) # B x n
        else:
            return userEmbs, itemEmbs
