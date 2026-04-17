import logging
import os
import pickle
import random
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, NoReturn, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from .ddp import all_gather, is_main_process
from .dict2obj import Config

LOGGER = Config(
    name="RecSys",
    filename="log.txt",
    level=logging.DEBUG,
    filelevel=logging.DEBUG,
    consolelevel=logging.INFO,
    formatter=Config(
        filehandler=logging.Formatter("%(asctime)s:\t%(message)s"),
        consolehandler=logging.Formatter("%(message)s"),
    ),
    info=print,
    debug=print,
)

COLOR = {
    "current": "{0}",
    "cpu": "{0}",
    0: "{0}",
    1: "\033[1;35m{0}\033[0m",
    2: "\033[1;34m{0}\033[0m",
    3: "\033[1;33m{0}\033[0m",
}


class AverageMeter:
    r"""Compute and store the average and current value of a metric.

    Parameters
    ----------
    name : str
        The name of the meter.
    metric : Callable
        Metric function.
    fmt : str, optional
        Output format, by default ``'.5f'``.
    best_caster : Callable, optional
        The best caster between ``min`` or ``max`` based on the metric,
        by default ``max``.
    """

    def __init__(
        self, name: str, metric: Callable, fmt: str = ".5f", best_caster: Callable = max
    ):
        r"""Initialize the meter."""
        assert isinstance(metric, Callable), (
            f"metric should be Callable but {type(metric)} received ..."
        )
        self.name = name
        self.fmt = fmt
        self.caster = best_caster
        self.reset()
        self.__history = []
        self.__metric = metric

    @property
    def history(self):
        r"""Get the historical results."""
        return self.__history

    @history.setter
    def history(self, val: List):
        r"""Set the historical results."""
        self.__history = val.copy()

    def reset(self) -> None:
        r"""Reset the meter values."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.active = False

    def gather(self, val: float, n: int, reduction: str):
        r"""Gather metrics from other processes (DDP)."""
        vals = all_gather(val)
        ns = all_gather(n)
        if reduction == "mean":
            val = sum([val * n for val, n in zip(vals, ns)])
        elif reduction == "sum":
            val = sum(vals)
        else:
            raise ValueError(
                f"Receive reduction {reduction} but 'mean' or 'sum' expected ..."
            )
        return val, int(sum(ns))

    def update(self, val: float, n: int = 1, reduction: str = "mean") -> None:
        r"""Update the meter with a new value.

        Parameters
        ----------
        val : float
            The metric value.
        n : int, optional
            Batch size, by default 1.
        reduction : str, optional
            Reduction mode: ``'sum'`` or ``'mean'``, by default ``'mean'``.
        """
        val, n = self.gather(val, n, reduction)
        self.val = val
        self.count += n
        self.sum += val
        self.avg = self.sum / self.count

    def step(self) -> str:
        r"""Save the average value to history and reset the meter.

        Returns
        -------
        str
            Average value in a formatted string.
        """
        self.history.append(self.avg)
        info = str(self)
        self.reset()
        return info

    def check(self, *values):
        r"""Compute and validate the metric value.

        Parameters
        ----------
        *values
            Arguments forwarded to the metric function.

        Returns
        -------
        float
            The computed metric value.

        Raises
        ------
        ValueError
            If the metric value is NaN or Inf.
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
        r"""Plot the meter history as a line chart.

        Parameters
        ----------
        freq : int, optional
            Evaluation frequency used to scale the x-axis, by default 1.
        """
        timeline = np.arange(len(self.history)) * freq
        self.fig = plt.figure(dpi=300)
        ax = self.fig.gca()
        ax.plot(timeline, self.history, marker="", figure=self.fig)
        ax.set_title(self.name)

    def save(self, path: str, mode: str = "") -> None:
        r"""Save the plotted curve as a PNG file.

        Parameters
        ----------
        path : str
            Directory path to save the file to.
        mode : str, optional
            Prefix to add to the filename, by default ``''``.

        Returns
        -------
        str
            The filename of the saved file.
        """
        filename = f"{mode}{self.name}.png"
        self.fig.savefig(os.path.join(path, filename))
        plt.close(self.fig)
        return filename

    def which_is_better(self, other: float) -> bool:
        r"""Return the better value between the current average and *other*."""
        return self.caster(self.avg, other)

    def argbest(self, freq: int = 1) -> float:
        r"""Return the index and value of the best result in history.

        Parameters
        ----------
        freq : int, optional
            Evaluation frequency used to scale the returned index,
            by default 1.

        Returns
        -------
        index : int
            The scaled index of the best result, or -1 if history is empty.
        value : float
            The best metric value, or -1 if history is empty.

        Raises
        ------
        ValueError
            If ``caster`` is neither ``min`` nor ``max``.
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
        r"""Return a formatted string of the current average."""
        fmtstr = "{name} Avg: {avg:{fmt}}"
        return fmtstr.format(**self.__dict__)

    def __call__(self, *values, n: int = 1, reduction: str = "mean") -> None:
        r"""Compute the metric and update the meter.

        Parameters
        ----------
        *values
            Arguments forwarded to the metric function.
        n : int, optional
            Batch size, by default 1.
        reduction : str, optional
            Reduction mode: ``'mean'`` or ``'sum'``, by default ``'mean'``.
        """
        self.active = True
        self.update(val=self.check(*values), n=n, reduction=reduction)


class Monitor(Config):
    r"""A collection of :class:`AverageMeter` instances organized by prefix and metric.

    Inherits from :class:`~freerec.dict2obj.Config` and provides
    serialization, state management, and TensorBoard integration.
    """

    def state_dict(self) -> Dict:
        r"""Return the state dictionary of all monitors.

        Returns
        -------
        dict
            Nested dictionary mapping ``prefix -> metric -> meter.name -> history``.
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
        r"""Load the history of monitors from a state dictionary.

        Parameters
        ----------
        state_dict : dict
            The state dictionary that contains the history of monitors.
        strict : bool, optional
            Whether to strictly enforce that the keys in *state_dict*
            match the keys in the monitor, by default ``False``.
        """
        monitors: Dict[str, List[AverageMeter]]
        for prefix, monitors in self.items():
            for metric, meters in monitors.items():
                for meter in meters:
                    meter.history = state_dict[prefix][metric].get(
                        meter.name, meter.history
                    )

    def save(self, path: str, filename: str = "monitors.pickle"):
        r"""Save the current state of the monitors to disk via :func:`export_pickle`.

        Parameters
        ----------
        path : str
            Directory path to save the state file.
        filename : str, optional
            Name of the output file, by default ``'monitors.pickle'``.
        """
        file_ = os.path.join(path, filename)
        export_pickle(self.state_dict(), file_)

    def write(self, path: str):
        r"""Write the history of all monitors to TensorBoard.

        Parameters
        ----------
        path : str
            Directory path for the :class:`~torch.utils.tensorboard.SummaryWriter` log.
        """
        with SummaryWriter(path) as writer:
            monitors: Dict[str, List[AverageMeter]]
            for prefix, monitors in self.items():
                for metric, meters in monitors.items():
                    for meter in meters:
                        for t, val in enumerate(meter.history):
                            writer.add_scalar("/".join([prefix, meter.name]), val, t)


def export_pickle(data: Any, file: str) -> NoReturn:
    r"""Export data into pickle format.

    Parameters
    ----------
    data : Any
        The object to serialize.
    file : str
        Destination file path.

    Raises
    ------
    ExportError
        If pickling or writing to disk fails.
    """
    fh = None
    try:
        fh = open(file, "wb")
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)
    except (EnvironmentError, pickle.PicklingError) as err:
        ExportError_ = type("ExportError", (Exception,), dict())
        raise ExportError_(f"Export Error: {err}")
    finally:
        if fh is not None:
            fh.close()


def import_pickle(file: str) -> Any:
    r"""Import data from a pickle file.

    Parameters
    ----------
    file : str
        Source file path.

    Returns
    -------
    Any
        The deserialized object.

    Raises
    ------
    ImportError
        If unpickling or reading from disk fails.
    """
    fh = None
    try:
        fh = open(file, "rb")
        return pickle.load(fh)
    except (EnvironmentError, pickle.UnpicklingError) as err:
        raise ImportError(f"Import Error: {err}")
    finally:
        if fh is not None:
            fh.close()


def export_yaml(data: Any, file: str) -> NoReturn:
    r"""Export data into YAML format.

    Parameters
    ----------
    data : Any
        The object to serialize.
    file : str
        Destination file path.
    """
    with open(file, encoding="UTF-8", mode="w") as fh:
        fh.write(yaml.dump(data, sort_keys=False))


def import_yaml(file: str) -> Any:
    r"""Import data from a YAML file.

    Parameters
    ----------
    file : str
        Source file path.

    Returns
    -------
    Any
        The deserialized object.
    """
    with open(file, encoding="UTF-8", mode="r") as f:
        data = yaml.full_load(f)
    return data


def set_logger(path: str, log2file: bool = True, log2console: bool = True) -> None:
    r"""Set up a logger instance.

    Parameters
    ----------
    path : str
        Directory path where the log file will be created.
    log2file : bool, optional
        Whether to log messages to a file, by default ``True``.
    log2console : bool, optional
        Whether to log messages to the console, by default ``True``.

    Returns
    -------
    :class:`logging.Logger`
        A configured logger instance.
    """
    logger = logging.getLogger(LOGGER.name)
    logger.setLevel(LOGGER.level)

    if log2file:
        handler = logging.FileHandler(
            os.path.join(path, LOGGER.filename), encoding="utf-8"
        )
        handler.setLevel(LOGGER.filelevel)
        handler.setFormatter(LOGGER.formatter.filehandler)
        logger.addHandler(handler)
    if log2console:
        handler = logging.StreamHandler()
        handler.setLevel(LOGGER.consolelevel)
        handler.setFormatter(LOGGER.formatter.consolehandler)
        logger.addHandler(handler)
    logger.debug(
        "========================================================================"
    )
    logger.debug(
        "========================================================================"
    )
    logger.debug(
        "========================================================================"
    )
    logger.propagate = False
    LOGGER["info"] = logger.info
    LOGGER["debug"] = logger.debug
    return logger


def set_color(device: Union[int, str]):
    r"""Set the ANSI color code for terminal output of the specified device.

    Parameters
    ----------
    device : Union[int, str]
        The device identifier, which can be an integer or string.
    """
    try:
        if isinstance(device, int):
            device = device % 4
        COLOR["current"] = COLOR[device]
    except KeyError:
        pass


def infoLogger(words: str, main_process_only: bool = True):
    r"""Log an info-level message.

    Parameters
    ----------
    words : str
        The message to log.
    main_process_only : bool, optional
        If ``True``, only the main process logs the message, by default ``True``.

    Returns
    -------
    str
        The colorized message that was logged, or ``None`` if skipped.
    """
    if main_process_only and not is_main_process():
        return
    words = COLOR["current"].format(words)
    LOGGER.info(words)
    return words


def debugLogger(words: str):
    r"""Log a debug-level message.

    Parameters
    ----------
    words : str
        The message to log.

    Returns
    -------
    str
        The message that was logged.
    """
    LOGGER.debug(words)
    return words


def warnLogger(warn: str):
    r"""Log a warning-level message.

    Parameters
    ----------
    warn : str
        The warning message to log.

    Returns
    -------
    str
        The formatted warning message that was logged.
    """
    words = f"\033[1;31m[Warning] >>> {warn} \033[0m"
    LOGGER.info(words)
    return words


def timemeter(func):
    r"""Decorator that measures and logs the wall-clock time of a function.

    Parameters
    ----------
    func : Callable
        The function to wrap.

    Returns
    -------
    Callable
        The wrapped function that logs elapsed time after each call.
    """

    def wrapper(*args, **kwargs):
        r"""Wrap the function with timing."""
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        infoLogger(
            f"[Wall TIME] >>> {func.__qualname__} takes {end - start:.6f} seconds ...",
            False,
        )
        return results

    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    return wrapper


def mkdirs(*paths: str) -> None:
    r"""Create directories, ignoring those that already exist.

    Parameters
    ----------
    *paths : str
        One or more directory paths to create.
    """
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def activate_benchmark(benchmark: bool) -> None:
    r"""Activate or deactivate the cuDNN benchmark mode.

    Parameters
    ----------
    benchmark : bool
        If ``True``, enable ``cudnn.benchmark`` and disable deterministic mode;
        if ``False``, do the opposite.
    """
    from torch.backends import cudnn

    if benchmark:
        infoLogger(
            "[Benchmark] >>> cudnn.benchmark == True | cudnn.deterministic == False"
        )
        cudnn.benchmark, cudnn.deterministic = True, False
    else:
        infoLogger(
            "[Benchmark] >>> cudnn.benchmark == False | cudnn.deterministic == True"
        )
        cudnn.benchmark, cudnn.deterministic = False, True


def set_seed(seed: int) -> int:
    r"""Set the seed for all random number generators.

    Configures ``random``, :mod:`numpy`, and :mod:`torch` (CPU and CUDA).

    Parameters
    ----------
    seed : int
        The seed to set. If -1, a random seed between 0 and 2048 is generated.

    Returns
    -------
    int
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
