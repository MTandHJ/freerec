

from typing import Dict, Optional, Union

import torch
import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from torch_geometric.nn import LGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import freerec
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import RecSysArch
from freerec.criterions import BPRLoss
from freerec.data.fields import FieldModule, FieldList, BufferField
from freerec.data.tags import USER, ITEM, ID


cfg = Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--layers", type=int, default=3)
cfg.set_defaults(
    description="LightGCN",
    root="../data",
    dataset='Gowalla_m1',
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()


class LightGCN(RecSysArch):

    def __init__(
        self, fields: FieldModule, 
        graph: Data,
        num_layers: int = 3
    ) -> None:
        super().__init__()

        self.fields = fields
        self.conv = LGConv(False)
        self.num_layers = num_layers
        self.User, self.Item = self.fields[USER, ID], self.fields[ITEM, ID]
        self.graph = graph

        self.initialize()

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph: Data):
        self.__graph = graph
        T.ToSparseTensor()(self.__graph)
        self.__graph.adj_t = gcn_norm(
            self.__graph.adj_t, num_nodes=self.User.count + self.Item.count,
            add_self_loops=False
        )

    def to(
        self, device: Optional[Union[int, torch.device]] = None, 
        dtype: Optional[Union[torch.dtype, str]] = None, 
        non_blocking: bool = False
    ):
        if device:
            self.graph.to(device)
        return super().to(device, dtype, non_blocking)

    def forward(
        self, users: Optional[BufferField] = None, 
        positives: Optional[BufferField] = None,
        negatives: Optional[BufferField] = None
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
            users, items = users.data, torch.cat(
                [positives.data, negatives.data], dim=1
            )
            userFeats = userFeats[users] # B x 1 x D
            itemFeats = itemFeats[items] # B x n x D
            userEmbs = self.User.look_up(users) # B x 1 x D
            itemEmbs = self.Item.look_up(items) # B x n x D
            return torch.mul(userFeats, itemFeats).sum(-1), userEmbs, itemEmbs
        else:
            return userFeats, itemFeats



class CoachForLightGCN(Coach):


    def reg_loss(self, userEmbds, itemEmbds):
        userEmbds, itemEmbds = userEmbds.flatten(1), itemEmbds.flatten(1)
        loss = userEmbds.pow(2).sum() + itemEmbds.pow(2).sum()
        loss = loss / userEmbds.size(0)
        return loss / 2

    def train_per_epoch(self):
        for data in self.dataloader:
            users, positives, negatives = data.to(self.device)
            preds, users, items = self.model(users, positives, negatives)
            pos, neg = preds[:, 0], preds[:, 1]
            reg_loss = self.reg_loss(users.flatten(1), items.flatten(1)) * self.cfg.weight_decay
            loss = self.criterion(pos, neg) + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=preds.size(0), mode="mean", prefix='train', pool=['LOSS'])

    def evaluate(self, prefix: str = 'valid'):
        userFeats, itemFeats = self.model()
        data: FieldList
        for data in self.dataloader:
            users, unseen, seen = data.to(self.device)
            seen = seen.to_csr().to(self.device).to_dense().bool()
            targets = unseen.to_csr().to(self.device).to_dense()
            users = userFeats[users].flatten(1) # B x D
            items = itemFeats.flatten(1) # N x D
            preds = users @ items.T # B x N
            preds[seen] = -1e10

            self.monitor(
                preds, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['NDCG', 'RECALL']
            )


def main():

    basepipe = getattr(freerec.data.datasets, cfg.dataset)(cfg.root)
    trainpipe = basepipe.train_uniform_user_(num_negatives=1).split_(cfg.batch_size).tensor_()
    validpipe = basepipe.eval_user_unseens_seens_().split_(1024).tensor_()
    dataset = trainpipe.wrap_(validpipe)

    tokenizer = FieldModule(basepipe.fields)
    tokenizer.embed(
        cfg.embedding_dim, ID
    )
    model = LightGCN(
        tokenizer, basepipe.train().to_graph((USER, ID), (ITEM, ID)), num_layers=cfg.layers
    )

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
        )
    criterion = BPRLoss()

    coach = CoachForLightGCN(
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=None,
        device=cfg.device
    )
    coach.compile(cfg, monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'])
    coach.fit()



if __name__ == "__main__":
    main()

