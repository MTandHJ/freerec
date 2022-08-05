

from typing import Callable, Iterable, List, Dict, Optional

import torch
import os
from functools import partial
from collections import defaultdict


from .dict2obj import Config
from .utils import AverageMeter, getLogger, timemeter
from .metrics import *


DEFAULT_METRICS = {
    'LOSS': None,
    #############
    'MSE': mean_squared_error,
    'MAE': mean_abs_error,
    'RMSE': root_mse,
    #############
    'PRECISION': precision,
    'RECALL': recall,
    'HITRATE': hit_rate,
    #############
    'NDCG': normalized_dcg,
    'MRR': mean_reciprocal_rank,
    'MAP': mean_average_precision
}

DEFAULT_FMTS = {
    'LOSS': ".5f",
    #############
    'MSE': ".4f",
    'MAE': ".4f",
    'RMSE': ".4f",
    #############
    'PRECISION': ".3%",
    'RECALL': ".3%",
    'HITRATE': ".3%",
    #############
    'NDCG': ".4f",
    'MRR': ".4f",
    'MAP': ".4f",
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
        torch.save(self.model.state_dict(), os.path.join(self.cfg.INFO_PATH, self.cfg.SAVED_FILENAME))

    def save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.cfg.INFO_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = dict()
        checkpoint['epoch'] = epoch
        for module in self.cfg.CHECKPOINT_MODULES:
            checkpoint[module] = getattr(self, module).state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self) -> int:
        path = os.path.join(self.cfg.INFO_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = torch.load(path)
        for module in self.cfg.CHECKPOINT_MODULES:
            getattr(self, module).load_state_dict(checkpoint[module])
        return checkpoint['epoch']

    def save_best(self, path: str, prefix: str): ...

    def check_best(self, results: dict): ...

    @timemeter("Coach/compile")
    def compile(
        self, cfg: Config, callbacks: List[str]
    ):
        self.cfg = cfg
        self.meters = Config(
            train=defaultdict(list),
            valid=defaultdict(list)
        )
        for name in callbacks:
            name = name.upper()
            if '@' in name:
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

    def callback(
        self, *values,
        n: int = 1, mode: str = 'mean', 
        prefix: str = 'train', pool: Optional[List] = None
    ):
        metrics: Dict[List] = self.meters[prefix]
        for metric in pool:
            for meter in metrics.get(metric, []):
                meter(*values, n=n, mode=mode)

    @timemeter("Coach/step")
    def step(self, epoch: int):
        for prefix, callbacks in self.meters.items():
            callbacks: defaultdict[str, List[AverageMeter]]
            infos = [f"[Epoch: {epoch:<3d}] " + prefix.upper() + " >>> "]
            for meters in callbacks.values():
                infos += [meter.step() for meter in meters if meter.active]
            getLogger().info(' || '.join(infos))

    @timemeter("Coach/summary")
    def summary(self):
        for prefix, callbacks in self.meters.items():
            callbacks: defaultdict[str, List[AverageMeter]]
            for meters in callbacks.values():
                for meter in meters:
                    meter.plot()
                    meter.save(path=self.cfg.LOG_PATH, prefix=prefix)

    @timemeter("Coach/train")
    def train(self, trainloader: Iterable):
        self.model.train()
        inputs: Dict[str, torch.Tensor]
        targets: torch.Tensor
        for inputs, targets in trainloader:
            inputs = {field: val.to(self.device) for field, val in inputs.items()}
            targets = targets.to(self.device)

            outs = self.model(inputs)
            loss = self.criterion(outs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.callback(loss.item(), n=targets.size(0), mode="mean", prefix='train', pool=['LOSS'])

        self.lr_scheduler.step() # TODO: step() per epoch or per mini-batch ?

    @timemeter("Coach/evaluate")
    @torch.no_grad()
    def evaluate(self, dataloader: Iterable, prefix: str = 'valid'):
        self.model.eval()
        inputs: Dict[str, torch.Tensor]
        targets: Dict[str, torch.Tensor]
        running_preds: List[torch.Tensor] = []
        running_targets: List[torch.Tensor] = []
        for inputs, targets in dataloader:
            inputs = {field: val.to(self.device) for field, val in inputs.items()}
            targets = targets.to(self.device)

            preds = self.model(inputs)
            loss = self.criterion(preds, targets)

            running_preds.append(preds.detach().clone().cpu().flatten())
            running_targets.append(targets.detach().clone().cpu().flatten())
            
            if prefix == 'valid':
                self.callback(loss, n=targets.size(0), mode="mean", prefix=prefix, pool=['LOSS'])

        running_preds = torch.cat(running_preds)
        running_targets = torch.cat(running_targets)
        self.callback(
            running_preds, running_targets, 
            n=running_targets.size(0), mode="mean", prefix=prefix,
            pool=['PRECISION', 'RECALL', 'HITRATE', 'MSE', 'MAE', 'RMSE']
        )

    @timemeter("Coach/resume")
    def resume(self):
        start_epoch = self.load_checkpoint() if self.cfg.resume else 0
        getLogger().info(f"Load the recent checkpoint and train from epoch: {start_epoch}")
        return start_epoch

    @timemeter("Coach/fit")
    def fit(
        self, trainloader: Iterable, validloader: Iterable,
        *, epochs: int
    ):
        start_epoch = self.resume()
        for epoch in range(start_epoch, epochs):
            if epoch % self.cfg.CHECKPOINT_FREQ == 0:
                self.save_checkpoint(epoch)
            if epoch % self.cfg.EVAL_FREQ == 0:
                if self.cfg.EVAL_TRAIN:
                    self.evaluate(trainloader, prefix='train')
                if self.cfg.EVAL_VALID:
                    results = self.evaluate(validloader, prefix='valid')
                    self.check_best(results)
            
            self.train(trainloader)

            self.step(epoch)
        self.save()
        self.summary()

