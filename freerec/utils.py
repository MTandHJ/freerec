

import torch
import numpy as np

import logging
import time
import random
import os
from .dict2obj import Config


LOGGER = Config(
    name='RecSys', filename='log.txt', level=logging.DEBUG,
    filelevel=logging.DEBUG, consolelevel=logging.INFO,
    formatter=Config(
        filehandler=logging.Formatter('%(asctime)s:\t%(message)s'),
        consolehandler=logging.Formatter('%(message)s')
    )
)



class AverageMeter:

    def __init__(self, name: str, fmt: str = ".5f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1, mode: str = "mean") -> None:
        self.val = val
        self.count += n
        if mode == "mean":
            self.sum += val * n
        elif mode == "sum":
            self.sum += val
        else:
            raise ValueError(f"Receive mode {mode} but [mean|sum] expected ...")
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} Avg:{avg:{fmt}}"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:

    def __init__(self, *meters: AverageMeter, prefix: str = ""):
        self.meters = list(meters)
        self.prefix = prefix

    def display(self, *, epoch: int = 8888) -> None:
        entries = [f"[Epoch: {epoch:<4d}]" + self.prefix]
        entries += [str(meter) for meter in self.meters]
        logger = getLogger()
        logger.info('\t'.join(entries))

    def add(self, *meters: AverageMeter) -> None:
        self.meters += list(meters)

    def step(self) -> None:
        for meter in self.meters:
            meter.reset()


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

def timemeter(prefix=""):
    def decorator(func):
        logger = getLogger()
        def wrapper(*args, **kwargs):
            start = time.time()
            results = func(*args, **kwargs)
            end = time.time()
            logger.info(f"[Wall TIME]- {prefix} takes {end-start:.6f} seconds ...")
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



# # load model's parameters
# def load(
#     model: nn.Module, 
#     path: str, 
#     device: torch.device,
#     filename: str,
#     strict: bool = True, 
#     except_key: Optional[str] = None
# ) -> None:

#     filename = os.path.join(path, filename)
#     if str(device) == "cpu":
#         state_dict = torch.load(filename, map_location="cpu")
        
#     else:
#         state_dict = torch.load(filename)
#     if except_key is not None:
#         except_keys = list(filter(lambda key: except_key in key, state_dict.keys()))
#         for key in except_keys:
#             del state_dict[key]
#     model.load_state_dict(state_dict, strict=strict)
#     model.eval()


def save_checkpoint(path: str, epoch: int, **kwargs) -> None:
    path = os.path.join(path, "checkpoint.tar")
    checkpoint = dict()
    checkpoint['epoch'] = epoch
    for key, module in kwargs.items():
        checkpoint[key] = module.state_dict()
    torch.save(checkpoint, path)

# load the checkpoint
def load_checkpoint(
    path: str, **kwargs
) -> int:
    path = os.path.join(path, "checkpoint.tar")
    checkpoint = torch.load(path)
    for key, module in kwargs.items():
        module.load_state_dict(checkpoint[key])
    epoch = checkpoint['epoch']
    return epoch

def activate_benchmark(benchmark: bool) -> None:
    from torch.backends import cudnn
    if benchmark:
        getLogger().info(f"[Seed] >>> Activate benchmark")
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        getLogger().info(f"[Seed] >>> Deactivate benchmark")
        cudnn.benchmark, cudnn.deterministic = False, True

def set_seed(seed: int) -> None:
    if seed == -1:
        seed = random.randint(0, 1024)
        logger = getLogger()
        logger.info(f"[Seed] >>> Set seed randomly: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

