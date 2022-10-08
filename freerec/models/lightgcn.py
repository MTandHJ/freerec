

from typing import Dict, Optional

import torch
import torch_geometric
from torch_geometric.data.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from .base import RecSysArch

from ..data.fields import Tokenizer
from ..data.tags import ID, ITEM, SPARSE, DENSE, FEATURE, USER


__all__ = ['LightGCN']



class LightConv(MessagePassing):

    def __init__(self) -> None:
        super().__init__(aggr='add')

    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        edge_index, _ = add_self_loops(edge_index)
        row, col = edge_index
        deg = degree(col, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return x_j * norm.view(-1, 1)


class LightGCN(RecSysArch):

    def __init__(
        self, tokenizer: Tokenizer, 
        graph: Data,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.graph = graph
        self.conv = LightConv()
        self.num_layers = num_layers

        self.User, self.Item = self.tokenizer[USER, ID], self.tokenizer[ITEM, ID]

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
            features = self.conv(features, self.graph.edge_index)
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

