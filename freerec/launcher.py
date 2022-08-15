

from typing import Callable, List, Dict, Optional

import torch
import os
from functools import partial
from collections import defaultdict

from .data.datasets.base import BaseSet
from .data.utils import TQDMDataLoader, DataLoader
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
        datapipe: BaseSet,
        criterion: Callable, 
        optimizer: torch.optim.Optimizer, 
        lr_scheduler,
        device: torch.device
    ):
        self.model = model
        self.datapipe = datapipe
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

    @timemeter("Coach/resume")
    def resume(self):
        start_epoch = self.load_checkpoint() if self.cfg.resume else 0
        getLogger().info(f"[Coach] >>> Load the recent checkpoint and train from epoch: {start_epoch}")
        return start_epoch

    @timemeter("Coach/compile")
    def compile(
        self, cfg: Config, callbacks: List[str]
    ):
        self.cfg = cfg
        # meters for train|valid|test
        self.meters = Config()
        self.meters['train'] = {
            'LOSS': [
                AverageMeter(
                    name='LOSS', 
                    metric=DEFAULT_METRICS['LOSS'], 
                    fmt=DEFAULT_FMTS['LOSS']
                )
            ]
        }
        self.meters['valid'] = defaultdict(list)
        self.meters['test'] = defaultdict(list)

        for name in callbacks:
            name = name.upper()
            if '@' in name:
                callback, K = name.split('@')
                for prefix in ('valid', 'test'):
                    self.meters[prefix][callback].append(
                        AverageMeter(
                            name=name,
                            metric=partial(DEFAULT_METRICS[callback], k=int(K)),
                            fmt=DEFAULT_FMTS[callback]
                        )
                    )
            else:
                callback = name
                for prefix in ('valid', 'test'):
                    self.meters[prefix][callback].append(
                        AverageMeter(
                            name=name,
                            metric=DEFAULT_METRICS[callback],
                            fmt=DEFAULT_FMTS[callback]
                        )
                    )

        # dataloader
        self.load_dataloader()

    def load_dataloader(self):
        _DataLoader = TQDMDataLoader if self.cfg.progress else DataLoader
        self.dataloader = _DataLoader(
            datapipe=self.datapipe, num_workers=self.cfg.num_workers
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
            infos = [f"[Coach] >>> {prefix.upper():5} @Epoch: {epoch:<3d} " + " >>> "]
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

            logits = self.model(users, items)
            m, n = logits.size()
            targets = targets.repeat((m, n))
            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.callback(loss.item(), n=m * n, mode="mean", prefix='train', pool=['LOSS'])

        self.lr_scheduler.step() # TODO: step() per epoch or per mini-batch ?

    @torch.no_grad()
    def evaluate(self, prefix: str = 'valid'):
        # self.model.eval()
        # users: Dict[str, torch.Tensor]
        # items: Dict[str, torch.Tensor]
        # targets: torch.Tensor
        # for users, items, targets in self.dataloader:
        #     users = {name: val.to(self.device) for name, val in users.items()}
        #     items = {name: val.to(self.device) for name, val in items.items()}
        #     targets = targets.to(self.device)

        #     preds = self.model(users, items)
        #     loss = self.criterion(preds, targets)

        #     self.callback(loss, n=targets.size(0), mode="mean", prefix=prefix, pool=['LOSS'])
        #     self.callback(
        #         preds.detach().clone().cpu(), targets.detach().clone().cpu(),
        #         n=targets.size(0), mode="mean", prefix=prefix,
        #         pool=['PRECISION', 'RECALL', 'HITRATE', 'MSE', 'MAE', 'RMSE']
        #     )
        raise NotImplementedError() # TODO: row-wise ?

    @timemeter("Coach/valid")
    def valid(self):
        self.datapipe.valid() # TODO: multiprocessing pitfall ???
        self.model.eval()
        return self.evaluate(prefix='valid')

    @timemeter("Coach/test")
    def test(self):
        self.datapipe.test()
        self.model.eval()
        return self.evaluate(prefix='test')

    @timemeter("Coach/fit")
    def fit(self):
        start_epoch = self.resume()
        for epoch in range(start_epoch, self.cfg.epochs):
            if epoch % self.cfg.CHECKPOINT_FREQ == 0:
                self.save_checkpoint(epoch)
            if epoch % self.cfg.EVAL_FREQ == 0:
                if self.cfg.EVAL_VALID:
                    self.check_best(self.valid())
                if self.cfg.EVAL_TEST:
                    self.test()
            self.train()

            self.step(epoch)
        self.save()

        # last epoch
        self.check_best(self.valid())
        self.test()
        self.step(self.cfg.epochs)

        # visualization
        self.summary()


class CoachForCTR(Coach): ...

class CoachForMatching(Coach): ...