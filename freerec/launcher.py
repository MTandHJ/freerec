

from typing import Callable, List, Dict, Optional, Tuple, Union

import torch
import os
from functools import partial
from collections import defaultdict
from tqdm import tqdm

from .data.datasets.base import BaseSet
from .data.fields import Field, Fielder
from .data.dataloader import DataLoader
from .dict2obj import Config
from .utils import AverageMeter, infoLogger, timemeter
from .metrics import *


__all__ = ['Coach']


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
        dataset: BaseSet,
        criterion: Callable, 
        optimizer: torch.optim.Optimizer, 
        lr_scheduler,
        device: torch.device
    ):
        self.model = model
        self.dataset = dataset
        self.fields: Fielder[Field] = self.dataset.fields
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
      
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
        infoLogger(f"[Coach] >>> Load the recent checkpoint and train from epoch: {start_epoch}")
        return start_epoch

    @timemeter("Coach/compile")
    def compile(
        self, cfg: Config, monitors: List[str]
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

        for name in monitors:
            name = name.upper()
            if '@' in name:
                monitor, K = name.split('@')
                for prefix in ('valid', 'test'):
                    self.meters[prefix][monitor].append(
                        AverageMeter(
                            name=name,
                            metric=partial(DEFAULT_METRICS[monitor], k=int(K)),
                            fmt=DEFAULT_FMTS[monitor]
                        )
                    )
            else:
                monitor = name
                for prefix in ('valid', 'test'):
                    self.meters[prefix][monitor].append(
                        AverageMeter(
                            name=name,
                            metric=DEFAULT_METRICS[monitor],
                            fmt=DEFAULT_FMTS[monitor]
                        )
                    )

        # dataloader
        self.load_dataloader()

    def load_dataloader(self):
        self.__dataloader = DataLoader(
            datapipe=self.dataset, num_workers=self.cfg.num_workers
        )
    
    @property
    def dataloader(self):
        if self.cfg.verbose:
            return tqdm(
                self.__dataloader,
                leave=False, desc="վ'ᴗ' ի-"
            )
        else:
            return self.__dataloader

    def monitor(
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
        for prefix, monitors in self.meters.items():
            monitors: defaultdict[str, List[AverageMeter]]
            infos = [f"[Coach] >>> {prefix.upper():5} @Epoch: {epoch:<3d} >>> "]
            for meters in monitors.values():
                infos += [meter.step() for meter in meters if meter.active]
            infoLogger(' || '.join(infos))

    @timemeter("Coach/summary")
    def summary(self):
        for prefix, monitors in self.meters.items():
            monitors: defaultdict[str, List[AverageMeter]]
            for meters in monitors.values():
                for meter in meters:
                    meter.plot()
                    meter.save(path=self.cfg.LOG_PATH, prefix=prefix)


    def train_per_epoch(self):
        raise NotImplementedError()

    def evaluate(self, prefix: str = 'valid'):
        raise NotImplementedError()


    @timemeter("Coach/train")
    def train(self):
        self.dataset.train()
        self.model.train()
        return self.train_per_epoch()

    @timemeter("Coach/valid")
    @torch.no_grad()
    def valid(self):
        self.dataset.valid() # TODO: multiprocessing pitfall ???
        self.model.eval()
        return self.evaluate(prefix='valid')

    @timemeter("Coach/test")
    @torch.no_grad()
    def test(self):
        self.dataset.test()
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