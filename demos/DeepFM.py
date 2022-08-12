

import torch
import torch.nn as nn

from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import DeepFM
from freerec.data.datasets import Criteo
from freerec.data.utils import DataLoader, TQDMDataLoader
from freerec.data.fields import Tokenizer
from freerec.data.tags import SPARSE, DENSE, FEATURE
from freerec.criterions import BCELoss



def main():

    cfg = Parser()
    cfg.add_argument("-eb", "--embedding_dim", type=int, default=4)
    cfg.set_defaults(
        fmt="{description}={embedding_dim}={optimizer}-{lr}-{weight_decay}={seed}",
        description="DeepFM",
        root="../criteo"
    )
    cfg.compile()

    basedp = Criteo(cfg.root, split='train')
    datapipe = basedp.frame(shuffle=True).encode().chunk(batch_size=cfg.batch_size).dict().tensor()
    _DataLoader = TQDMDataLoader if cfg.progress else DataLoader
    trainloader = _DataLoader(datapipe, num_workers=cfg.num_workers)
    validloader = trainloader

    tokenizer = Tokenizer(datapipe.fields)
    tokenizer.embed(
        dim=cfg.embedding_dim, tags=(FEATURE, SPARSE)
    )
    tokenizer.embed(
        dim=cfg.embedding_dim, tags=(FEATURE, DENSE), linear=True
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




