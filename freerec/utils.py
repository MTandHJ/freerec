

from typing import Callable, Optional, Dict, List
import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import logging
import time
import random
import os
from collections import defaultdict
from freeplot.base import FreePlot
from freeplot.utils import export_pickle

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
        self.__history = []
        self.__metric = metric if metric else _unitary

    @property
    def history(self):
        return self.__history

    @history.setter
    def history(self, val: List):
        self.__history = val.copy()

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

    def plot(self, freq: int = 1) -> None:
        self.fp = FreePlot(
            shape=(1, 1),
            figsize=(2.2, 2),
            titles=(self.name,),
            dpi=300, latex=False
        )
        timeline = np.arange(len(self.history)) * freq
        self.fp.lineplot(timeline, self.history, marker='')
        self.fp.set_title(y=.98)

    def argbest(self, caster: Callable, freq: int = 1):
        if len(self.history) == 0:
            return '-', '-'
        indices = np.argsort(self.history)
        if caster is min:
            return indices[0] * freq, self.history[indices[0]]
        elif caster is max:
            return indices[-1] * freq, self.history[indices[-1]]
        else:
            raise ValueError("caster should be min or max ...")
    
    def save(self, path: str, prefix: str = '') -> None:
        filename = f"{prefix}{self.name}.png"
        self.fp.savefig(os.path.join(path, filename))
        return filename

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


class Monitor(Config):

    def state_dict(self) -> Dict:
        state_dict = defaultdict(dict)
        monitors: Dict[str, List[AverageMeter]]
        for prefix, monitors in self.items():
            for metric, meters in monitors.items():
                state_dict[prefix][metric] = dict()
                for meter in meters:
                    state_dict[prefix][metric][meter.name] = meter.history
        return state_dict

    def load_state_dict(self, state_dict: Dict, strict: bool = False):
        monitors: Dict[str, List[AverageMeter]]
        for prefix, monitors in self.items():
            for metric, meters in monitors.items():
                for meter in meters:
                    meter.history = state_dict[prefix][metric].get(meter.name, meter.history)

    
    def save(self, path: str, filename: str = 'monitors.pickle'):
        file_ = os.path.join(path, filename)
        export_pickle(self.state_dict(), file_)

    def write(self, path: str):
        with SummaryWriter(path) as writer:
            monitors: Dict[str, List[AverageMeter]]
            for prefix, monitors in self.items():
                for metric, meters in monitors.items():
                    for meter in meters:
                        for t, val in enumerate(meter.history):
                            writer.add_scalar(
                                '/'.join([prefix, meter.name]),
                                val,
                                t
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

