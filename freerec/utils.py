

from typing import Callable, Optional, Any
import torch
import numpy as np

import logging
import time
import random
import os
from freeplot.base import FreePlot

from .dict2obj import Config


LOGGER = Config(
    name='RecSys', filename='log.txt', level=logging.DEBUG,
    filelevel=logging.DEBUG, consolelevel=logging.INFO,
    formatter=Config(
        filehandler=logging.Formatter('%(asctime)s:\t%(message)s'),
        consolehandler=logging.Formatter('%(message)s')
    )
)


def _unitary(value):
    return value

class AverageMeter:

    def __init__(self, name: str, metric: Optional[Callable] = None, fmt: str = ".5f"):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.history = []
        self.__metric = metric if metric else _unitary

    def reset(self) -> None:
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0
        self.active = False

    def update(self, val: float, n: int = 1, mode: str = "mean") -> None:
        self.val = val
        self.count += n
        if mode == "mean":
            self.sum += val * n
        elif mode == "sum":
            self.sum += val
        else:
            raise ValueError(f"Receive mode {mode} but 'mean' or 'sum' expected ...")
        self.avg = self.sum / self.count

    def step(self) -> str:
        self.history.append(self.avg)
        info = str(self)
        self.reset()
        return info

    def callback(self, *values):
        val = self.__metric(*values)
        try:
            return val.item() # Tensor|ndarray
        except AttributeError:
            return val # float

    def plot(self) -> None:
        self.fp = FreePlot(
            shape=(1, 1),
            figsize=(2.2, 2),
            titles=(self.name,),
            dpi=300, latex=False
        )
        timeline = np.arange(1, len(self.history) + 1)
        self.fp.lineplot(timeline, self.history, marker='')
        self.fp.set_title(y=.98)
    
    def save(self, path: str, prefix: str = '') -> None:
        filename = f"{prefix}{self.name}.png"
        self.fp.savefig(os.path.join(path, filename))

    def __str__(self):
        fmtstr = "{name} Avg: {avg:{fmt}}"
        return fmtstr.format(**self.__dict__)

    def __call__(self, *values, n: int = 1, mode: str = "mean")  -> None:
        self.active = True
        self.update(
            val = self.callback(*values),
            n = n,
            mode = mode
        )


def set_logger(
    path: str,
    log2file: bool = True, log2console: bool = True
) -> None:
    logger = logging.getLogger(LOGGER.name)
    logger.setLevel(LOGGER.level)

    if log2file:
        handler = logging.FileHandler(
            os.path.join(path, LOGGER.filename), 
            encoding='utf-8'
        )
        handler.setLevel(LOGGER.filelevel)
        handler.setFormatter(LOGGER.formatter.filehandler)
        logger.addHandler(handler)
    if log2console:
        handler = logging.StreamHandler()
        handler.setLevel(LOGGER.consolelevel)
        handler.setFormatter(LOGGER.formatter.consolehandler)
        logger.addHandler(handler)
    logger.debug("========================================================================")
    logger.debug("========================================================================")
    logger.debug("========================================================================")
    return logger

def getLogger():
    return logging.getLogger(LOGGER.name)

def infoLogger(words: str):
    getLogger().info(words)

def debugLogger(words: str):
    getLogger().debug(words)

def warnLogger(warn: str):
    words = f"\033[1;31m {warn} \033[0m"
    getLogger().info(words)

def timemeter(prefix=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            results = func(*args, **kwargs)
            end = time.time()
            infoLogger(f"[Wall TIME] >>> {prefix} takes {end-start:.6f} seconds ...")
            return  results
        wrapper.__doc__ = func.__doc__
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

def mkdirs(*paths: str) -> None:
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def activate_benchmark(benchmark: bool) -> None:
    from torch.backends import cudnn
    if benchmark:
        infoLogger(f"[Seed] >>> Activate benchmark")
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        infoLogger(f"[Seed] >>> Deactivate benchmark")
        cudnn.benchmark, cudnn.deterministic = False, True

def set_seed(seed: int) -> None:
    if seed == -1:
        seed = random.randint(0, 2048)
        infoLogger(f"[Seed] >>> Set seed randomly: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

