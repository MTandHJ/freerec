

from typing import List, Dict

import torch

from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import DeepFM
from freerec.data.datasets import Criteo
from freerec.data.fields import Tokenizer
from freerec.data.tags import SPARSE, DENSE, FEATURE, TARGET
from freerec.criterions import BCELoss


class CoachforDeepFM(Coach):


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
            loss = self.criterion(preds, targets)

            running_preds.append(preds.detach().clone().cpu().flatten())
            running_targets.append(targets.detach().clone().cpu().flatten())
            
            self.callback(loss, n=targets.size(0), mode="mean", prefix=prefix, pool=['LOSS'])

        running_preds = torch.cat(running_preds)
        running_targets = torch.cat(running_targets)
        self.callback(
            running_preds, running_targets, 
            n=running_targets.size(0), mode="mean", prefix=prefix,
            pool=['PRECISION', 'RECALL', 'HITRATE', 'MSE', 'MAE', 'RMSE']
        )

def main():

    cfg = Parser()
    cfg.add_argument("-eb", "--embedding_dim", type=int, default=4)
    cfg.set_defaults(
        fmt="{description}={embedding_dim}={optimizer}-{lr}-{weight_decay}={seed}",
        description="DeepFM",
        root="../criteo"
    )
    cfg.compile()

    basepipe = Criteo(cfg.root)
    datapipe = basepipe.dataframe_(shuffle=True).encode_().chunk_(batch_size=cfg.batch_size).dict_().tensor_().group_()

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

    coach = CoachforDeepFM(
        model=model,
        datapipe=datapipe,
        criterion=BCELoss(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=cfg.DEVICE
    )
    coach.compile(cfg, callbacks=['loss', 'mse', 'mae', 'rmse', 'precision@100', 'recall@100', 'hitrate@100'])
    coach.fit()


if __name__ == "__main__":
    main()




