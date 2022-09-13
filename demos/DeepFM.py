



from typing import Dict, List

import torch

from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import DeepFM
from freerec.criterions import BCELoss
from freerec.data.datasets import Criteo
from freerec.data.fields import Tokenizer
from freerec.data.tags import FEATURE, SPARSE, DENSE



class CoachForDeepFM(Coach):


    def train_per_epoch(self):
        self.model.train()
        self.dataset.train()
        users: Dict[str, torch.Tensor]
        items: Dict[str, torch.Tensor]
        targets: torch.Tensor
        for users, items, targets in self.dataloader:
            users = {name: val.to(self.device) for name, val in users.items()}
            items = {name: val.to(self.device) for name, val in items.items()}
            targets = targets.to(self.device)

            preds = self.model(users, items)
            loss = self.criterion(preds, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=targets.size(0), mode="mean", prefix='train', pool=['LOSS'])

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
            loss = self.criterion(preds, targets)

            running_preds.append(preds.detach().clone().cpu().flatten())
            running_targets.append(targets.detach().clone().cpu().flatten())

            self.monitor(loss, n=targets.size(0), mode="mean", prefix=prefix, pool=['LOSS'])

        running_preds = torch.cat(running_preds)
        running_targets = torch.cat(running_targets)
        self.monitor(
            running_preds, running_targets, 
            n=1, mode="mean", prefix=prefix,
            pool=['NDCG', 'PRECISION', 'RECALL', 'HITRATE', 'MSE', 'MAE', 'RMSE']
        )



def main():

    import copy

    cfg = Parser()
    cfg.add_argument("-eb", "--embedding_dim", type=int, default=8)
    cfg.set_defaults(
        fmt="{description}={embedding_dim}={optimizer}-{lr}-{weight_decay}={seed}",
        description="DeepFM",
        root="../criteo",
        epochs=200,
        batch_size=256,
        optimizer='adam',
        lr=1e-3,
    )
    cfg.compile()

    basepipe = Criteo(cfg.root)
    datapipe = basepipe.pin_(buffer_size=cfg.buffer_size)
    datapipe = datapipe.chunk_(batch_size=cfg.batch_size).dict_().tensor_().group_()

    tokenizer = Tokenizer(datapipe.fields)
    tokenizer.embed(
        cfg.embedding_dim, (FEATURE, SPARSE)
    )
    tokenizer.embed(
        cfg.embedding_dim, (FEATURE, DENSE), linear=True
    )
    model = DeepFM(tokenizer).to(cfg.DEVICE)

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
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = BCELoss()

    coach = CoachForDeepFM(
        model=model,
        dataset=datapipe,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=cfg.DEVICE
    )
    coach.compile(cfg, monitors=['loss', 'mse', 'mae', 'rmse', 'precision@10', 'recall@10', 'hitrate@10', 'ndcg@10'])
    coach.fit()

if __name__ == "__main__":
    main()


