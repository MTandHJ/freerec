

from typing import Callable, Dict, List, Union
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import logging, time, random, os
from collections import defaultdict
from freeplot.base import FreePlot
from freeplot.utils import export_pickle

from .dict2obj import Config
from .ddp import is_main_process, all_gather


LOGGER = Config(
    name='RecSys', filename='log.txt', level=logging.DEBUG,
    filelevel=logging.DEBUG, consolelevel=logging.INFO,
    formatter=Config(
        filehandler=logging.Formatter('%(asctime)s:\t%(message)s'),
        consolehandler=logging.Formatter('%(message)s')
    ),
    info=print,
    debug=print
)

COLOR = {
    'current': "{0}",
    'cpu': "{0}",
    0: "{0}",
    1: "\033[1;35m{0}\033[0m",
    2: "\033[1;34m{0}\033[0m",
    3: "\033[1;33m{0}\033[0m",
    4: "\033[1;38m{0}\033[0m",
    5: "\033[1;32m{0}\033[0m"
}


class AverageMeter:
    r"""
    Computes and stores the average and current value of a metric.

    Parameters:
    -----------
    name: str
        The name of the meter.
    metric: Callable
        Metric function.
    fmt: str, optional (default: '.5f')
        Output format.
    best_caster: Callable, optional (default: max)
        The best caster between `min` or `max` based on the metric.
    """

    def __init__(
        self, name: str, metric: Callable, 
        fmt: str = ".5f", best_caster: Callable = max
    ):
        assert isinstance(metric, Callable), f"metric should be Callable but {type(metric)} received ..."
        self.name = name
        self.fmt = fmt
        self.caster = best_caster
        self.reset()
        self.__history = []
        self.__metric = metric

    @property
    def history(self):
        """Get the historical results."""
        return self.__history

    @history.setter
    def history(self, val: List):
        """Set the historical results."""
        self.__history = val.copy()

    def reset(self) -> None:
        """Reset the meter values"""
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0
        self.active = False

    def gather(self, val: float, n: int, mode: str):
        """Gather metrics from other processes (DDP)."""
        vals = all_gather(val)
        ns = all_gather(n)
        if mode == "mean":
            val = sum([val * n for val, n in zip(vals, ns)])
        elif mode == "sum":
            val = sum(vals)
        else:
            raise ValueError(f"Receive mode {mode} but 'mean' or 'sum' expected ...")
        return val, int(sum(ns))

    def update(self, val: float, n: int = 1, mode: str = "mean") -> None:
        r"""
        Updates the meter.

        Parameters:
        ----------
        val: float
            Value.
        n: int, optional (default: 1)
            Batch size.
        mode: str, optional (default: "mean")
            Mode: 'sum'|'mean'.
        """
        val, n = self.gather(val, n, mode)
        self.val = val
        self.count += n
        self.sum += val
        self.avg = self.sum / self.count

    def step(self) -> str:
        r"""
        Saves the average value and resets the meter.

        Returns:
        --------
        info: str
            Average value in a formatted string.
        """
        self.history.append(self.avg)
        info = str(self)
        self.reset()
        return info

    def check(self, *values):
        r"""
        Check the metric value.

        Parameters:
        -----------
        values: variable-length argument list
            Metric function arguments.

        Returns:
        --------
        The metric value.
        """
        val = self.__metric(*values)
        if isinstance(val, torch.Tensor):
            val = val.cpu().numpy()
        else:
            val = np.array(val)
        if np.isnan(val) or np.isinf(val):
            raise ValueError(
                f"The metric of {self.name} got an unexpected value: {val.item()}."
            )
        return val.item()

    def plot(self, freq: int = 1) -> None:
        r"""
        Plots the meter values.
        
        Parameters:
        -----------
        freq: int, optional (default: 1)
            Plot frequency.
        """
        self.fp = FreePlot(
            shape=(1, 1),
            titles=(self.name,),
            dpi=300, latex=False
        )
        timeline = np.arange(len(self.history)) * freq
        self.fp.lineplot(timeline, self.history, marker='')
        self.fp.set_title(y=.98)

    def save(self, path: str, prefix: str = '') -> None:
        r"""
        Save the curves as a PNG file.

        Parameters:
        -----------
        path : str
            The path to save the file to.
        prefix : str, optional
            The prefix to add to the filename.

        Returns:
        --------
        filename : str
            The filename of the saved file.
        """
        filename = f"{prefix}{self.name}.png"
        self.fp.savefig(os.path.join(path, filename))
        return filename

    def which_is_better(self, other: float) -> bool:
        return self.caster(self.avg, other)

    def argbest(self, freq: int = 1) -> float:
        r"""
        Return the index and value of the best result in history.

        Parameters:
        -----------
        freq : int, optional
            The evaluation frequency, defaults to 1.

        Returns:
        --------
        index : float
            The index of the best result.
            If no available data, returns -1.
        value : float
            The value of the best result.
            If no available data, returns -1.
        """
        if len(self.history) == 0:
            return -1, -1
        indices = np.argsort(self.history)
        if self.caster is min:
            return indices[0] * freq, self.history[indices[0]]
        elif self.caster is max:
            return indices[-1] * freq, self.history[indices[-1]]
        else:
            raise ValueError("caster should be `min' or `max' ...")

    def __str__(self):
        r"""
        Return the string representation of this object.

        Returns:
        --------
        str
            The string representation of this object.
        """
        fmtstr = "{name} Avg: {avg:{fmt}}"
        return fmtstr.format(**self.__dict__)

    def __call__(self, *values, n: int = 1, mode: str = "mean")  -> None:
        r"""
        Add a new data point to the history.

        Parameters:
        -----------
        values : list or tuple
            The value(s) to add to the history.
        n : int, optional
            The number of times to add each value, defaults to 1.
        mode : str, optional
            The mode of adding values: 'mean' or 'sum', defaults to 'mean'.
        """
        self.active = True
        self.update(
            val = self.check(*values),
            n = n,
            mode = mode
        )


class Monitor(Config):

    def state_dict(self) -> Dict:
        r"""
        Return the state dictionary of monitors.

        Returns:
        --------
        state_dict : Dict
            The state dictionary that contains the history of all monitors.
        """
        state_dict = defaultdict(dict)
        monitors: Dict[str, List[AverageMeter]]
        for prefix, monitors in self.items():
            for metric, meters in monitors.items():
                state_dict[prefix][metric] = dict()
                for meter in meters:
                    state_dict[prefix][metric][meter.name] = meter.history
        return state_dict

    def load_state_dict(self, state_dict: Dict, strict: bool = False):
        r"""
        Load the history of monitors from the given state dictionary.

        Parameters:
        -----------
        state_dict : Dict
            The state dictionary that contains the history of monitors.
        strict : bool, optional (default=False)
            Whether to strictly enforce that the keys in the state dictionary match the keys in the monitor.
        """
        monitors: Dict[str, List[AverageMeter]]
        for prefix, monitors in self.items():
            for metric, meters in monitors.items():
                for meter in meters:
                    meter.history = state_dict[prefix][metric].get(meter.name, meter.history)

    def save(self, path: str, filename: str = 'monitors.pickle'):
        r"""
        Save the current state of the monitors to disk.

        Parameters:
        -----------
        path : str
            The path to the directory to save the state.
        filename : str, optional (default='monitors.pickle')
            The name of the file to save the state.
        """
        file_ = os.path.join(path, filename)
        export_pickle(self.state_dict(), file_)

    def write(self, path: str):
        r"""
        Write the history of monitors to Tensorboard.

        Parameters:
        -----------
        path : str
            The path to the directory to write the history to.
        """
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
    r"""
    Set up a logger instance.

    Parameters:
    -----------
    path : str
        The path of the log file.
    log2file : bool, optional
        Whether to log messages to a file. Default is True.
    log2console : bool, optional
        Whether to log messages to console. Default is True.

    Returns:
    --------
    logger : logging.Logger
        A configured logger instance.
    """
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
    LOGGER['info'] = logger.info
    LOGGER['debug'] = logger.debug
    return logger

def set_color(device: Union[int, str]):
    r"""
    Set a color for the output of the specified device.

    Parameters:
    -----------
    device : Union[int, str]
        The device identifier, which can be an integer or string.
    """
    try:
        COLOR['current'] = COLOR[device]
    except KeyError:
        pass

def infoLogger(words: str, main_process_only: bool = True):
    r"""
    Log an info-level message.

    Parameters:
    -----------
    words : str
        The message to log.
    main_process_only: bool, default to `True`
        `True`: print for main process only

    Returns:
    --------
    words : str
        The message that was logged.
    """
    if main_process_only and not is_main_process():
        return
    words = COLOR['current'].format(words)
    LOGGER.info(words)
    return words

def debugLogger(words: str):
    r"""
    Log a debug-level message.

    Parameters:
    -----------
    words : str
        The message to log.

    Returns:
    --------
    words : str
        The message that was logged.
    """
    LOGGER.debug(words)
    return words

def warnLogger(warn: str):
    r"""
    Log a warning-level message.

    Parameters:
    -----------
    warn : str
        The warning message to log.

    Returns:
    --------
    words : str
        The warning message that was logged.
    """
    words = f"\033[1;31m[Warning] >>> {warn} \033[0m"
    LOGGER.info(words)
    return words

def timemeter(func):
    r"""
    A decorator to measure the running time of a function.

    Parameters:
    -----------
    prefix : str, optional
        A prefix to be displayed in the logging message, by default "".

    Returns:
    --------
    wrapper : function
        The decorated function.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        infoLogger(f"[Wall TIME] >>> {func.__qualname__} takes {end-start:.6f} seconds ...", False)
        return  results
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    return wrapper

def mkdirs(*paths: str) -> None:
    r"""
    Create directories.

    Parameters:
    -----------
    *paths : str
        Paths of directories to create.
    """
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

def activate_benchmark(benchmark: bool) -> None:
    r"""
    Activate or deactivate the cudnn benchmark mode.

    Parameters:
    -----------
    benchmark : bool
        Whether to activate the benchmark mode.
    """
    from torch.backends import cudnn
    if benchmark:
        infoLogger(f"[Benchmark] >>> cudnn.benchmark == True | cudnn.deterministic == False")
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        infoLogger(f"[Benchmark] >>> cudnn.benchmark == False | cudnn.deterministic == True")
        cudnn.benchmark, cudnn.deterministic = False, True

def set_seed(seed: int) -> int:
    r"""
    Set the seed for the random number generators.

    Parameters:
    -----------
    seed : int
        The seed to set. If seed is -1, a random seed between 0 and 2048 will be generated.

    Returns:
    --------
    seed : int
        The actual seed used.
    """
    if seed == -1:
        seed = random.randint(0, 2048)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    infoLogger(f"[Seed] >>> Set seed: {seed}")
    return seed

