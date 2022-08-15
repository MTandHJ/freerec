



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
            m = targets.size(0)

            preds = self.model(users, items)
            n = len(preds) // m
            preds = preds.view(m, n)
            targets = targets.repeat((1, n))
            targets[:, 1:].fill_(0)
            loss = self.criterion(preds, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.callback(loss.item(), n=targets.size(0), mode="mean", prefix='train', pool=['LOSS'])

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
            m = targets.size(0)

            preds = self.model(users, items)
            n = len(preds) // m
            preds = preds.view(m, n)
            targets = targets.repeat((1, n))
            targets[:, 1:].fill_(0)
            loss = self.criterion(preds, targets)

            running_preds.append(preds.detach().clone().cpu())
            running_targets.append(targets.detach().clone().cpu())

            self.callback(loss, n=targets.size(0), mode="mean", prefix=prefix, pool=['LOSS'])

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
    cfg.set_defaults(
        fmt="{description}={embedding_dim}={optimizer}-{lr}-{weight_decay}={seed}",
        description="NeuCF",
        root="../movielens",
        epochs=200,
        batch_size=1024,
        buffer_size=10240,
        optimizer='adam',
        lr=1e-3,
    )
    cfg.compile()

    basepipe = MovieLens1M_(cfg.root)
    datapipe = basepipe.dataframe_(buffer_size=cfg.buffer_size).encode_().sample_negative_(num_negatives=4)
    datapipe = datapipe.chunk_(batch_size=cfg.batch_size).dict_().tensor_().group_()

    tokenizer_mf = Tokenizer(datapipe.fields)
    tokenizer_mf.embed(
        dim=cfg.embedding_dim, tags=(FEATURE, SPARSE)
    )
    tokenizer_mf.embed(
        dim=cfg.embedding_dim, tags=(FEATURE, DENSE), linear=True
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
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = BCELoss()
    criterion.regulate(
        model.parameters(), rtype='l2', weight=cfg.weight_decay
    )

    coach = CoachForNCF(
        model=model,
        datapipe=datapipe,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=cfg.DEVICE
    )
    coach.compile(cfg, callbacks=['loss', 'mse', 'mae', 'rmse', 'precision@10', 'recall@10', 'hitrate@10', 'ndcg@10'])
    coach.fit()

if __name__ == "__main__":
    main()


