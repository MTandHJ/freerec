


import torch
import torchdata.datapipes as dp

import freerec
from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.data.datasets import RecDataSet, MovieLens1M
from freerec.data.utils import DataLoader, TQDMDataLoader
from freerec.data.fields import Tokenizer
from freerec.dict2obj import Config

from sklearn.preprocessing import Binarizer

from collections import defaultdict

class MovieLens1M_(MovieLens1M):
    """
    MovieLens1M: (user, item, rating, timestamp)
        https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets
    """

    _cfg = Config(
        sparse = [freerec.data.fields.SparseField(name=name, na_value=0, dtype=int) for name in ('User', 'Item')],
        dense = [freerec.data.fields.DenseField(name="Timestamp", na_value=0., dtype=float, transformer='none')],
        target = [freerec.data.fields.LabelField(name='Rating', na_value=None, dtype=int, transformer=Binarizer(threshold=1))]
    )
    _cfg.fields = _cfg.sparse + _cfg.target + _cfg.dense

    _active = False


class Pairer(RecDataSet):

    def __init__(self, positive: RecDataSet, ratio: int = 1) -> None:
        super().__init__()
        self.positive = positive
        self.ratio = ratio
        self.cfg = self.positive.cfg


    def summary(self):
        self.negative = defaultdict[set]



def main():

    cfg = Parser()
    cfg.add_argument("-eb", "--embedding_dim", type=int, default=4)
    cfg.set_defaults(
        fmt="{description}={embedding_dim}={optimizer}-{lr}-{weight_decay}={seed}",
        description="DeepFM",
        root="../criteo"
    )
    cfg.compile()

    datapipe = Criteo(cfg.root, split='train').encoder(batch_size=cfg.batch_size, buffer_size=cfg.buffer_size)
    _DataLoader = TQDMDataLoader if cfg.progress else DataLoader
    trainloader = _DataLoader(datapipe, num_workers=cfg.num_workers)
    validloader = trainloader

    tokenizer = Tokenizer(datapipe.fields)
    tokenizer.embed(
        dim=cfg.embedding_dim, features='sparse'
    )
    tokenizer.embed(
        dim=cfg.embedding_dim, features='dense', linear=True
    )
    model = DeepFM(tokenizer).to(cfg.DEVICE)

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum, weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay
        )
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    coach = Coach(
        model=model,
        criterion=BCELoss(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=cfg.DEVICE
    )
    coach.compile(cfg, callbacks=['loss', 'mse', 'mae', 'rmse', 'precision@100', 'recall@100', 'hitrate@100'])
    coach.fit(trainloader, validloader, epochs=cfg.epochs)


if __name__ == "__main__":
    main()




