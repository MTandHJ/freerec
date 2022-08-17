



from typing import Dict, List

import torch

from freerec.dict2obj import Config
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import NeuCF
from freerec.criterions import BCELoss
from freerec.data.datasets import MovieLens1M
from freerec.data.fields import Tokenizer
from freerec.data.tags import FEATURE, SPARSE, DENSE, USER, ITEM, ID, TARGET
from freerec.data.fields import SparseField, DenseField
from freerec.data.preprocessing import Binarizer
from freerec.utils import timemeter



class MovieLens1M_(MovieLens1M):
    """
    MovieLens1M: (user, item, rating, timestamp)
        https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets
    """

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=0, dtype=int, tags=[USER, ID, FEATURE]),
            SparseField(name='ItemID', na_value=0, dtype=int, tags=[ITEM, ID, FEATURE]),
        ],
        dense = [DenseField(name="Timestamp", na_value=0., dtype=float, transformer='none', tags=FEATURE)],
        # target: 0|1
        target = [DenseField(name='Rating', na_value=None, dtype=int, transformer=Binarizer(threshold=1), tags=TARGET)]
    )

    _cfg.fields = _cfg.sparse + _cfg.target + _cfg.dense



class CoachForNCF(Coach):


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
            targets = targets.to(self.device)

            preds = self.model(users, items)
            m, n = preds.size()
            targets = targets.repeat((1, n))
            targets[:, 1:].fill_(0)
            loss = self.criterion(preds, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.callback(loss.item(), n=m * n, mode="mean", prefix='train', pool=['LOSS'])

        self.lr_scheduler.step()


    def evaluate(self, prefix: str = 'valid'):
        self.model.eval()
        users: Dict[str, torch.Tensor]
        items: Dict[str, torch.Tensor]
        targets: torch.Tensor
        running_preds: List[torch.Tensor] = []
        running_targets: List[torch.Tensor] = []
        for users, items, targets in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}
            targets = targets.to(self.device)

            preds = self.model(users, items)
            m, n = preds.size()
            targets = targets.repeat((1, n))
            targets[:, 1:].fill_(0)
            loss = self.criterion(preds, targets)

            running_preds.append(preds.detach().clone().cpu())
            running_targets.append(targets.detach().clone().cpu())

            self.callback(loss, n=m * n, mode="mean", prefix=prefix, pool=['LOSS'])

        running_preds = torch.cat(running_preds)
        running_targets = torch.cat(running_targets)
        self.callback(
            running_preds, running_targets, 
            n=m, mode="mean", prefix=prefix,
            pool=['NDCG', 'PRECISION', 'RECALL', 'HITRATE', 'MSE', 'MAE', 'RMSE']
        )



def main():

    import copy

    cfg = Parser()
    cfg.add_argument("-eb", "--embedding_dim", type=int, default=8)
    cfg.add_argument("-neg", "--num_negs", type=int, default=4)
    cfg.set_defaults(
        fmt="{description}={embedding_dim}={optimizer}-{lr}-{weight_decay}={seed}",
        description="NeuCF",
        root="../movielens",
        epochs=20,
        batch_size=256,
        optimizer='adam',
        lr=1e-3,
        weight_decay=0.,
    )
    cfg.compile()

    basepipe = MovieLens1M_(cfg.root)
    datapipe = basepipe.pin_(buffer_size=cfg.buffer_size).sample_negative_(num_negatives=cfg.num_negs)
    datapipe = datapipe.chunk_(batch_size=cfg.batch_size).dict_().tensor_().group_()

    tokenizer_mf = Tokenizer(datapipe.fields)
    tokenizer_mf.embed(
        dim=cfg.embedding_dim, tags=(FEATURE, SPARSE)
    )
    tokenizer_mlp = copy.deepcopy(tokenizer_mf)
    model = NeuCF(tokenizer_mf, tokenizer_mlp).to(cfg.DEVICE)

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
    criterion = BCELoss()
    criterion.regulate(model.parameters(), rtype='l2', weight=cfg.weight_decay)

    coach = CoachForNCF(
        model=model,
        datapipe=datapipe,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=cfg.DEVICE
    )
    coach.compile(cfg, callbacks=['loss', 'precision@10', 'recall@10', 'hitrate@10', 'ndcg@10', 'ndcg@20'])
    coach.fit()

if __name__ == "__main__":
    main()


