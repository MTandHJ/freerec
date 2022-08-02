

from gc import callbacks
from typing import Callable, Iterable, List, Dict

import torch
import os
from functools import partial
from collections import defaultdict


from .dict2obj import Config
from .utils import AverageMeter, getLogger
from .metrics import *



DEFAULT_METRICS = {
    'LOSS': None,
    'NDCG': normalized_dcg,
    'RECALL': recall,
    'PRECISION': precision,
    'HITRATE': hit_rate,
    'MSE': mean_squared_error,
    'MAE': mean_abs_error,
    'RMSE': root_mse
}

DEFAULT_FMTS = {
    'LOSS': ".5f",
    'NDCG': ".4f",
    'RECALL': ".4f",
    'PRECISION': ".3%",
    'HITRATE': ".3%",
    'MSE': ".4f",
    'MAE': ".4f",
    'RMSE': ".4f"
}


class Coach:
    
    def __init__(
        self, model: torch.nn.Module,
        criterion: Callable, 
        optimizer: torch.optim.Optimizer, 
        lr_scheduler,
        device: torch.device
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self._best = 0.
        self.steps = 0

      
    def save(self) -> None:
        torch.save(self.model.state_dict(), os.path.join(self.cfg.INFO_PATH, self.cfg.FILENAME))

    def save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.cfg.INFO_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = dict()
        checkpoint['epoch'] = epoch
        for module in self.CHECKPOINT_MODULES:
            checkpoint[module] = getattr(self, module).state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self) -> int:
        path = os.path.join(self.cfg.INFO_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = torch.load(path)
        for module in self.CHECKPOINT_MODULES:
            getattr(self, module).load_state_dict(checkpoint[module])
        return checkpoint['epoch']

    def save_best(self, path: str, prefix: str): ...

    def check_best(self, results: dict): ...
    
    def compile(
        self, cfg: Config, callbacks: List[str]
    ):
        self.cfg = cfg
        self.meters = Config(
            train=defaultdict(list),
            valid=defaultdict(list)
        )
        for name in callbacks:
            if '@' in callback:
                callback, K = name.split('@')
                for prefix in self.meters:
                    self.meters[prefix][callback].append(
                        AverageMeter(
                            name=name,
                            metric=partial(DEFAULT_METRICS[callback], k=int(K)),
                            fmt=DEFAULT_FMTS[callback]
                        )
                    )
            else:
                callback = name
                for prefix in self.meters:
                    self.meters[prefix][callback].append(
                        AverageMeter(
                            name=name,
                            metric=DEFAULT_METRICS[callback],
                            fmt=DEFAULT_FMTS[callback]
                        )
                    )
        self.callbacks = set(self.meters.train.keys())

    def summary(self, epoch: int):
        for prefix, callbacks in self.meters.items():
            callbacks: defaultdict[str, List[AverageMeter]]
            infos = [f"[Epoch: {epoch:<4d}]" + prefix + " >>> "]
            for meters in callbacks.values():
                infos += [str(meter) for meter in meters if meter.active]
            getLogger().info('\t'.join(infos))

    def over(self):
        for prefix, callbacks in self.meters.items():
            callbacks: defaultdict[str, List[AverageMeter]]
            for meters in callbacks.values():
                for meter in meters:
                    meter.plot()
                    meter.save(path=self.cfg.LOG_PATH, prefix=prefix)

    def train(self, trainloader: Iterable):
        self.model.train()
        for data in trainloader:
            data = {item.to(self.device) for item in data}

            self.model.train() # make sure in training mode
            outs = self.model(data)
            loss = self.criterion(outs, data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.lr_scheduler.step() # TODO: step() per epoch or per mini-batch ?
        return self.meter.loss.avg


    def evaluate(self, dataloader: Iterable): ...

    def fit(
        self, trainloader: Iterable, validloader: Iterable,
        *, epochs: int, start_epoch: int = 0
    ):
        for epoch in range(start_epoch, epochs):
            if epoch % self.cfg.CHECKPOINT_FREQ == 0:
                self.save_checkpoint(epoch)
            if epoch % self.cfg.EVAL_FREQ == 0:
                if self.cfg.EVAL_TRAIN:
                    self.evaluate(trainloader)
                if self.cfg.EVAL_VALID:
                    results = self.evaluate(validloader)
                    self.check_best(results)
            
            self.train(trainloader)

            self.summary(epoch)
        
        self.over()

