

from typing import List

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



class MovieLens1M_(MovieLens1M):
    """
    MovieLens1M: (user, item, rating, timestamp)
        https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets
    """

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=0, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=0, dtype=int, tags=[ITEM, ID]),
        ],
        dense = [DenseField(name="Timestamp", na_value=0., dtype=float, transformer='none', tags=FEATURE)],
        # target: 0|1
        target = [DenseField(name='Rating', na_value=None, dtype=int, transformer=Binarizer(threshold=1), tags=TARGET)]
    )

    _cfg.fields = _cfg.sparse + _cfg.target + _cfg.dense



class CoachForNCF(Coach):


    def train_per_epoch(self):
        self.model.train()
        self.dataset.train()
        Target = self.fields.whichis(TARGET)
        for users, items, targets in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}
            targets = targets[Target.name].to(self.device)

            preds = self.model(users, items)
            m, n = preds.size()
            targets = targets.repeat((1, n))
            targets[:, 1:].fill_(0)
            loss = self.criterion(preds, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=m * n, mode="mean", prefix='train', pool=['LOSS'])

        self.lr_scheduler.step()


    def evaluate(self, prefix: str = 'valid'):
        self.model.eval()
        running_preds: List[torch.Tensor] = []
        running_targets: List[torch.Tensor] = []
        Target = self.fields.whichis(TARGET)
        for users, items, targets in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}
            targets = targets[Target.name].to(self.device)

            preds = self.model(users, items)
            m, n = preds.size()
            targets = targets.repeat((1, n))
            targets[:, 1:].fill_(0)
            loss = self.criterion(preds, targets)

            running_preds.append(preds.detach().clone().cpu())
            running_targets.append(targets.detach().clone().cpu())

            self.monitor(loss, n=m * n, mode="mean", prefix=prefix, pool=['LOSS'])

        running_preds = torch.cat(running_preds)
        running_targets = torch.cat(running_targets)
        self.monitor(
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
    datapipe = basepipe.pin_(buffer_size=cfg.buffer_size).shard_()
    trainpipe = datapipe.negatives_for_train_(num_negatives=cfg.num_negs)
    validpipe = datapipe.negatives_for_eval_(num_negatives=99) # 1:99
    dataset = trainpipe.wrap_(validpipe).chunk_(cfg.batch_size).dict_().tensor_().group_()

    tokenizer_mf = Tokenizer(basepipe.fields)
    tokenizer_mf.embed(
        cfg.embedding_dim, ID
    )
    tokenizer_mlp = copy.deepcopy(tokenizer_mf)
    model = NeuCF(tokenizer_mf, tokenizer_mlp).to(cfg.device)

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2), 
            weight_decay=cfg.weight_decay
        )
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    criterion = BCELoss()

    coach = CoachForNCF(
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=cfg.device
    )
    coach.compile(cfg, monitors=['loss', 'precision@10', 'recall@10', 'hitrate@10', 'ndcg@10', 'ndcg@20'])
    coach.fit()

if __name__ == "__main__":
    main()


