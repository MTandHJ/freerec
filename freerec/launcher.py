

from typing import Callable, Iterable, List

import torch
from .dict2obj import Config
from .utils import  save_checkpoint

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

        self._best_nat = 0.
        self._best_rob = 0.
        self.steps = 0

    def summary(self, log_path: str):
        self.logger.plotter.plot()
        self.logger.plotter.save(log_path)

    def save_best_nat(self, path: str, prefix: str = PRE_BESTNAT): ...

    def check_best(self, results: dict): ...
       
    def save(self, path: str, filename: str = SAVED_FILENAME) -> None:
        torch.save(self.model.state_dict(), os.path.join(path, filename))
    
    def compile(
        self, cfg: Config, callbacks: List[str]
    ):
        self.cfg = cfg
        self.callbacks = callbacks

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

        self.lr_scheduler.step() # update the learning rate
        return self.meter.loss.avg


    def evaluate(self, dataloader: Iterable): ...

    def fit(
        self, trainloader: Iterable, validloader: Iterable,
        *, epochs: int, start_epoch: int = 0
    ):
        for epoch in range(start_epoch, epochs):
            if epoch % self.cfg.SAVE_FREQ == 0:
                save_checkpoint(
                    self.cfg.INFO_PATH, epoch,
                    model=self.model, optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler
                )
            if epoch % self.cfg.EVAL_FREQ == 0:
                if self.cfg.EVAL_TRAIN:
                    self.evaluate(trainloader)
                if self.cfg.EVAL_VALID:
                    results = self.evaluate(validloader)
                    self.check_best(results)
            
            loss = self.train(trainloader)

