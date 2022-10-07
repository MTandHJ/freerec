

from typing import Dict, Optional

import torch
import torch.nn as nn
import dgl
import dgl.function as dglfn

from .base import RecSysArch

from ..data.fields import SparseToken, Tokenizer, SparseField
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
        # self.__graph = dgl.add_self_loop(graph)
        self.__graph = dgl.remove_self_loop(graph)
        if self.norm == 'left':
            degs = self.graph.out_degrees().float()
            norm = 1 / degs
        elif self.norm == 'right':
            degs = self.graph.in_degrees().float()
            norm = 1 / degs
        else:
            degs = self.graph.out_degrees().float()
            norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0.
        self.__graph.ndata['norm'] = norm.view(-1, 1)

    def forward(self, features: torch.Tensor):
        self.graph.to(features.device)
        with self.graph.local_scope():
            if self.norm in ['left', 'both']:
                features = features * self.graph.srcdata['norm']
                
            self.graph.srcdata['input'] = features
            self.graph.update_all(
                message_func=dglfn.copy_src('input', 'hidden'),
                reduce_func=dglfn.sum('hidden', 'output')
            )
            features = self.graph.dstdata['output']
            if self.norm in ['right', 'both']:
                features = features * self.graph.srcdata['norm']
            return features

        
class LightGCN(RecSysArch):


    def __init__(
        self, tokenizer: Tokenizer,
        graph: dgl.DGLGraph, num_layers: int = 3
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.User: SparseToken = self.tokenizer.groupby(USER, ID)[0]
        self.Item: SparseToken = self.tokenizer.groupby(ITEM, ID)[0]
        self.lightconv = LightConv(graph)
        self.num_layers = num_layers

        self.initialize()

    def preprocess(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]):
        users, items = users[self.User.name], items[self.Item.name]
        m, n = items.size()
        users = users.repeat((1, n))
        return users, items

    def forward(
        self, users: Optional[Dict[str, torch.Tensor]] = None, 
        items: Optional[Dict[str, torch.Tensor]] = None
    ):
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight
        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        allFeats = [features]
        for _ in range(self.num_layers):
            features = self.lightconv(features)
            allFeats.append(features)
        allFeats = torch.stack(allFeats, dim=1).mean(dim=1)
        userFeats, itemFeats = torch.split(allFeats, (self.User.count, self.Item.count))
        if self.training: # Batch
            users, items = self.preprocess(users, items)
            userFeats = userFeats[users] # B x n x D
            itemFeats = itemFeats[items] # B x n x D
            userEmbs = self.User.look_up(users) # B x n x D
            itemEmbs = self.Item.look_up(items) # B x n x D
            return torch.mul(userFeats, itemFeats).sum(-1), userEmbs, itemEmbs
        else:
            return userFeats, itemFeats
