

from typing import Union, List

import torch
import os, argparse, time
from argparse import ArgumentParser

from .dict2obj import Config
from .ddp import is_distributed, main_process_only, is_main_process, all_gather
from .utils import (
    mkdirs, import_yaml,
    timemeter, set_color, set_seed, activate_benchmark, 
    set_logger, infoLogger, warnLogger
)


__all__ = ['Parser', 'CoreParser']


r"""
A configuration module for a Recommender System.

Templates:
----------
DATA_DIR : str
    Directory for saving historical metric information.
SUMMARY_DIR : str
    Directory for saving final summary information.
CHECKPOINT_PATH : str
    Path for saving training checkpoints (``checkpoint.tar`` only, used for ``--resume``).
LOG_PATH : str
    Path for saving all experiment outputs: logs, model weights (``model.pt``, ``best.pt``),
    metric data (``monitors.pkl``, ``best.pkl``), and summaries.
CORE_CHECKPOINT_PATH : str
    Path for tuning.
CORE_LOG_PATH : str
    Path for tuning logging.
TIME : str
    A string representation of time in the format of "%m%d%H%M%S".
CONFIG : Config
    A Config object containing general configurations for the recommender system.
CORE_CONFIG : Config
    A Config object containing configurations for tuning the recommender system.


Configurations:
---------------
SAVED_FILENAME : str
    The filename of saved model parameters (saved under ``LOG_PATH``).
BEST_FILENAME : str
    The filename of saved best model parameters (saved under ``LOG_PATH``).
CHECKPOINT_FREQ : int
    The frequency of saving checkpoints.
CHECKPOINT_MODULES : list
    A list of modules to save in a checkpoint.
CHECKPOINT_FILENAME : str
    The filename of the saved checkpoint.
SUMMARY_FILENAME : str
    The filename of the summary.
MONITOR_FILENAME : str
    The filename of the monitor.
MONITOR_BEST_FILENAME : str
    The filename of the best monitor.
description : str
    The description of the recommender system.
EXCLUSIVE : bool
    If True, run a command exclusively.
COMMAND : str
    The command to be run.
ENVS : dict
    A dictionary of environment variables.
PARAMS : dict
    A dictionary of parameters.
DEFAULTS : dict
    A dictionary of default parameters.
"""

DATA_DIR = 'data'
SUMMARY_DIR = 'summary'
CHECKPOINT_PATH = "./infos/{description}/{dataset}/{device}"
LOG_PATH = "./logs/{description}/{dataset}/{id}"
CORE_CHECKPOINT_PATH = "./infos/{description}/core"
CORE_LOG_PATH = "./logs/{description}/core"
TIME = "%m%d%H%M%S"

CONFIG = Config(
    # path|file
    SAVED_FILENAME = "model.pt",
    BEST_FILENAME = "best.pt",
    CHECKPOINT_FREQ = 1,
    CHECKPOINT_MODULES = ['model', 'optimizer', 'lr_scheduler'],
    CHECKPOINT_FILENAME = "checkpoint.tar",
    SUMMARY_FILENAME = "SUMMARY.md",
    MONITOR_FILENAME = "monitors.pkl",
    MONITOR_BEST_FILENAME = "best.pkl",

    # monitors
    monitors = [],
    which4best = "LOSS"
)

CORE_CONFIG = Config(
    MONITOR_BEST_FILENAME = CONFIG.MONITOR_BEST_FILENAME,
    CHECKPOINT_FILENAME = CONFIG.CHECKPOINT_FILENAME,
    EXCLUSIVE = False,
    COMMAND = None,
    ENVS = dict(),
    PARAMS = dict(),
    DEFAULTS = dict(),

    log2file = True,
    log2console = True
)


class Parser(Config):
    r"""Configuration parser for recommendation system training.

    Wraps :class:`argparse.ArgumentParser` and :class:`~Config` to manage
    command-line arguments, YAML configuration files, and runtime settings
    such as device selection, logging, and checkpoint paths.

    Attributes
    ----------
    SAVED_FILENAME : str
        Filename for saved model parameters.
    BEST_FILENAME : str
        Filename for saved best model parameters.
    CHECKPOINT_FREQ : int
        Frequency of saving checkpoints.
    CHECKPOINT_MODULES : list
        Modules to save in a checkpoint.
    CHECKPOINT_FILENAME : str
        Filename of the saved checkpoint.
    SUMMARY_FILENAME : str
        Filename of the summary.
    MONITOR_FILENAME : str
        Filename of the monitor.
    MONITOR_BEST_FILENAME : str
        Filename of the best monitor.
    DATA_DIR : str
        Directory for saving historical metric information.
    SUMMARY_DIR : str
        Directory for saving final summary information.
    CHECKPOINT_PATH : str
        Path for saving checkpoints.
    LOG_PATH : str
        Path for saving logging information.
    monitors : list
        List of metric names to monitor.
    which4best : str
        Metric name used for selecting the best model.
    """

    # FILE
    SAVED_FILENAME: str
    BEST_FILENAME: str
    CHECKPOINT_FREQ: int
    CHECKPOINT_MODULES: List[str]
    CHECKPOINT_FILENAME: str
    SUMMARY_FILENAME: str
    MONITOR_FILENAME: str
    MONITOR_BEST_FILENAME: str

    # PATH
    DATA_DIR: str
    SUMMARY_DIR: str
    CHECKPOINT_PATH: str
    LOG_PATH: str

    monitors: list
    which4best: str

    def __init__(self) -> None:
        r"""Initialize Parser with default CONFIG and parse arguments."""
        super().__init__(**CONFIG)
        self.parse()

    @main_process_only
    def readme(self, path: str, mode: str = "w") -> None:
        r"""Write a README.md file containing all configuration key-value pairs.

        Parameters
        ----------
        path : str
            The directory to write the README.md file into.
        mode : str, optional
            The file opening mode, by default ``"w"``.
        """
        time_ = time.strftime("%Y-%m-%d-%H:%M:%S")
        file_ = os.path.join(path, "README.md")
        s = "|  {key}  |   {val}    |\n"
        info = "\n## {0} \n\n\n".format(time_)
        info += "|  Attribute   |   Value   |\n"
        info += "| :-------------: | :-----------: |\n"
        for key in sorted(self.keys(), key=lambda x: x.upper()):
            info += s.format(key=key, val=self[key])
        with open(file_, mode, encoding="utf8") as fh:
            fh.write(info)

    def reset(self):
        r"""Reset all configurations to the defaults defined in CONFIG."""
        self.clear()
        for key, val in CONFIG.items():
            self[key] = val

    @timemeter
    def parse(self):
        r"""Register all command-line arguments to the internal :class:`argparse.ArgumentParser`.

        Registers arguments for data paths, model hyperparameters, training
        schedule, evaluation settings, logging, and reproducibility.
        """

        self.parser = argparse.ArgumentParser()

        self.add_argument("--root", type=str, default=".", help="data path")
        self.add_argument("--dataset", type=str, default="RecDataSet", help="useless if no need to automatically select a dataset")
        self.add_argument("--tasktag", type=str, choices=('MATCHING', 'NEXTITEM', 'PREDICTION'), default=None, help="to specify a tasktag for dataset")
        self.add_argument("--config", type=str, default=None, help="config.yaml")
        self.add_argument("--ranking", type=str, choices=('full', 'pool'), default='full', help="full: full ranking; pool: sampled-based ranking")
        self.add_argument("--retain-seen", action="store_true", default=False, help="True: retain seen candidates during evaluation")

        self.add_argument("--device", default=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu', help="device")
        self.add_argument("--ddp-backend", type=str, default='nccl', help="ddp backend")

        # model
        self.add_argument("--optimizer", type=str, default="adam", help="Optimizer: adam (default), sgd, ...")
        self.add_argument("--nesterov", action="store_true", default=False, help="nesterov for SGD")
        self.add_argument("-mom", "--momentum", type=float, default=0.9, help="the momentum used for SGD")
        self.add_argument("-beta1", "--beta1", type=float, default=0.9, help="the first beta argument for Adam")
        self.add_argument("-beta2", "--beta2", type=float, default=0.999, help="the second beta argument for Adam")
        self.add_argument("-wd", "--weight-decay", type=float, default=None, help="weight for 'l1|l2|...' regularzation")
        self.add_argument("-lr", "--lr", "--LR", "--learning-rate", type=float, default=None)
        self.add_argument("-b", "--batch-size", type=int, default=None)
        self.add_argument("-gas", "--gradient-accumulation-steps", type=int, default=1)
        self.add_argument("--epochs", type=int, default=None)

        # logging
        self.add_argument("--log2file", action="store_false", default=True, help="if True, save logs to a file")
        self.add_argument("--log2console", action="store_false", default=True, help="if True, display logs on the console")

        # eval
        self.add_argument("--eval-valid", action="store_false", default=True, help="if True, evaluate validset")
        self.add_argument("--eval-test", action="store_true", default=False, help="if True, evaluate testset")
        self.add_argument("--eval-freq", type=int, default=5, help="the evaluation frequency")
        self.add_argument("-esp", "--early-stop-patience", type=int, default=1e23, help="the steps for early stopping")

        self.add_argument("--num-workers", type=int, default=4)

        self.add_argument("--seed", type=int, default=1, help="calling --seed=-1 for a random seed")
        self.add_argument("--benchmark", action="store_true", default=False, help="cudnn.benchmark == True ?")
        self.add_argument("--resume", action="store_true", default=False, help="resume the training from the recent checkpoint")

        self.add_argument("--id", type=str, default=time.strftime(TIME))
        self.add_argument("-m", "--description", type=str, default="RecSys")

    def add_argument(self, *args: str, **kwargs):
        r"""Add an argument to the internal :class:`argparse.ArgumentParser`.

        Parameters
        ----------
        *args : str
            The name(s) of the argument.
        **kwargs
            Additional keyword arguments passed to
            :meth:`argparse.ArgumentParser.add_argument`.

        Notes
        -----
        Any underscore (``_``) in argument names will be replaced by a
        hyphen (``-``).
        """
        args = (arg.replace('_', '-') for arg in args) # user '-' instead of '_'
        action = self.parser.add_argument(*args, **kwargs)
        self[action.dest] = action.default

    def set_defaults(self, **kwargs):
        r"""Set default values for registered arguments.

        Parameters
        ----------
        **kwargs
            Keyword arguments mapping argument names to their default values.
        """
        self.parser.set_defaults(**kwargs)
        for key, val in kwargs.items():
            self[key] = val

    def set_device(self, device: Union[torch.device, str, int]):
        r"""Set the computation device and configure color output.

        Parameters
        ----------
        device : Union[:class:`torch.device`, str, int]
            The target device identifier.

        Returns
        -------
        device
            The resolved device identifier.
        """
        try:
            device = int(device)
        except ValueError:
            pass

        set_color(device)
        self.device = torch.device(device)
        try:
            torch.cuda.set_device(device)
        except ValueError:
            pass
        except AttributeError:
            pass
        return device

    def init_ddp(self):
        r"""Initialize Distributed Data Parallel (DDP) if running in distributed mode."""
        import torch.distributed as dist
        if is_distributed():
            dist.init_process_group(backend=self.ddp_backend, init_method="env://")
            self.device = int(os.environ["LOCAL_RANK"])
            # synchronize ids
            self.id = all_gather(self.id)[0]
            infoLogger(f"[DDP] >>> DDP is activated ...")

    def set_tasktag(self):
        r"""Convert the ``tasktag`` string to a :class:`~freerec.data.tags.TaskTags` enum."""
        from .data.tags import TaskTags
        if self.tasktag is not None:
            self['tasktag'] = TaskTags(self.tasktag.upper())
    
    @timemeter
    def load(self):
        r"""Load and merge settings from a YAML configuration file.

        Parses command-line arguments first, then overlays values from the
        YAML file specified by ``--config`` if provided.

        Returns
        -------
        :class:`argparse.Namespace`
            The parsed arguments with defaults overwritten by the config file.

        Raises
        ------
        KeyError
            If a parameter key in the config file is not recognized.
        """
        args = self.parser.parse_args()
        keys, _ = zip(*args._get_kwargs())
        if hasattr(args, 'config') and args.config:
            for key, val in import_yaml(args.config).items():
                if key in keys: # overwriting defaults
                    self.set_defaults(**{key: val})
                elif key.upper() in self:
                    self[key.upper()] = val
                elif key in self:
                    self[key] = val
                else:
                    self[key] = val
                    warnLogger(f"Find an undefined parameter `{key}' in `{args.config}' ...")
        return self.parser.parse_args()

    @timemeter
    def compile(self):
        r"""Compile the full runtime configuration.

        Performs the following steps in order:

        1. Load settings from a YAML file if ``--config`` is specified.
        2. Generate ``CHECKPOINT_PATH`` and ``LOG_PATH`` from templates.
        3. Initialize DDP if running in distributed mode.
        4. Configure device, logger, random seed, and benchmark mode.
        5. Write README.md under both ``CHECKPOINT_PATH`` and ``LOG_PATH``.
        """
        args = self.load()
        self.update(**dict(args._get_kwargs()))
        
        self['DATA_DIR'] = DATA_DIR
        self['SUMMARY_DIR'] = SUMMARY_DIR

        self.init_ddp()
        self['CHECKPOINT_PATH'] = CHECKPOINT_PATH.format(**self)
        self['LOG_PATH'] = LOG_PATH.format(**self)
        mkdirs(
            self.CHECKPOINT_PATH, self.LOG_PATH,
            os.path.join(self.LOG_PATH, self.DATA_DIR),
            os.path.join(self.LOG_PATH, self.SUMMARY_DIR)
        )

        self.set_device(self.device)
        set_logger(path=self.LOG_PATH, log2file=self.log2file, log2console=self.log2console)

        activate_benchmark(self.benchmark)
        self.seed = set_seed(self.seed)

        self.set_tasktag()

        self.readme(self.CHECKPOINT_PATH) # create README.md
        self.readme(self.LOG_PATH)

        if is_main_process():
            infoLogger(str(self))


class CoreParser(Config):
    r"""Configuration parser for hyperparameter grid search (tuning).

    Extends :class:`~Config` to manage environment variables, parameter
    grids, and default values for launching multiple training subprocesses.

    Attributes
    ----------
    ALL_ENVS : tuple
        Names of environment-level arguments that can be overridden from
        the command line.
    EXCLUSIVE : bool
        If ``True``, run grid search entries one by one.
    COMMAND : str
        The shell command template for each subprocess.
    ENVS : :class:`~Config`
        Environment variables shared across all subprocesses.
    PARAMS : :class:`~Config`
        Parameter grid for hyperparameter search.
    DEFAULTS : :class:`~Config`
        Default parameter values used when ``EXCLUSIVE`` is ``True``.
    """

    ALL_ENVS = (
        'description', 'root', 'device', 'num_workers'
    )

    def __init__(self) -> None:
        r"""Initialize CoreParser with default CORE_CONFIG."""
        super().__init__(**CORE_CONFIG)

    def check(self) -> None:
        r"""Validate the loaded configuration.

        Raises
        ------
        ValueError
            If ``COMMAND`` is not specified.
        KeyError
            If required environment variables (``root``, ``dataset``,
            ``device``) are missing from ``ENVS``.
        """
        template = """
        Please make sure the configuration file follows the template below:

        command: python xxx.py
        envs:
            root: ../../data
            dataset: Gowalla_m1
            device: 0,1,2,3
        params:
            optimizer: [adam, sgd]
            learning_rate: [1.e-3, 1.e-2, 1.e-1]
            weight_decay: [0, 1.e-4, 2.e-4, 5.e-4]
            batch_size: [128, 256, 512, 1024]
        defaults:
            optimizer: adam
            learning_rate: 1.e-3
            weight_decay: 0
            batch_size: 256
            epochs: 100
            seed: 1

        where 'command' is necessary but 'envs', 'params' and 'defaults' are optional ...

        Notes: when calling '--exclusive' for grid search one by one, 
            'defaults' is required for clear comparsions in tensorbaord.
        """
        if self.COMMAND is None:
            raise ValueError(template)
        for key in ('root', 'dataset', 'device'):
            if self.ENVS.get(key, None) is None:
                raise KeyError(f"No `{key}' is allocated, calling '--{key}' to specify it")

        self.ENVS = Config(self.ENVS)
        self.PARAMS = Config(self.PARAMS)

    @timemeter
    def load(self, args: ArgumentParser) -> None:
        r"""Load settings from a YAML configuration file.

        Parameters
        ----------
        args : :class:`argparse.ArgumentParser`
            The parsed command-line arguments containing ``config``,
            ``exclusive``, ``resume``, and environment overrides.
        """
        config = {key.upper(): vals for key, vals in import_yaml(args.config).items()}
        self.update(**config)
        self.EXCLUSIVE = args.exclusive
        self.resume = args.resume
        for key, val in args._get_kwargs():
            if key in self.ALL_ENVS and val is not None:
                self.ENVS[key] = val

    @timemeter
    def compile(self, args) -> None:
        r"""Compile the full tuning configuration.

        Parameters
        ----------
        args : :class:`argparse.Namespace`
            The parsed command-line arguments.

        Notes
        -----
        Performs the following steps:

        1. Load parameter grid from a YAML file.
        2. Validate the configuration via :meth:`check`.
        3. Generate ``CORE_CHECKPOINT_PATH`` and ``CORE_LOG_PATH``.
        4. Configure the logger.
        5. Write README.md under ``CORE_LOG_PATH``.
        """
        self.load(args)
        self.check()

        self['DATA_DIR'] = DATA_DIR
        self['SUMMARY_DIR'] = SUMMARY_DIR
        self['CHECKPOINT_PATH'] = CHECKPOINT_PATH
        self['LOG_PATH'] = LOG_PATH
        self['CORE_CHECKPOINT_PATH'] = CORE_CHECKPOINT_PATH.format(**self.ENVS)
        self['CORE_LOG_PATH'] = CORE_LOG_PATH.format(**self.ENVS)
        mkdirs(self.CORE_CHECKPOINT_PATH, self.CORE_LOG_PATH)
        set_logger(path=self.CORE_LOG_PATH, log2file=self.log2file, log2console=self.log2console)

        self.readme(self.CORE_LOG_PATH)

    def readme(self, path: str, mode: str = "w") -> None:
        r"""Write a README.md file with environment, parameter, and default settings.

        Parameters
        ----------
        path : str
            The directory to write the README.md file into.
        mode : str, optional
            The file opening mode, by default ``"w"``.
        """
        time_ = time.strftime("%Y-%m-%d-%H:%M:%S")
        file_ = os.path.join(path, "README.md")
        s = "|  {key}  |   {val}    |\n"
        info = "\n## {0} \n\n\n".format(time_)
        info += "|  Attribute   |   Value   |\n"
        info += "| :-------------: | :-----------: |\n"
        for key, val in self.ENVS.items():
            info += s.format(key=key, val=val)
        for key, val in self.PARAMS.items():
            info += s.format(key=key, val=val)
        for key, val in self.DEFAULTS.items():
            info += s.format(key=f"{key} (default)", val=val)
        with open(file_, mode, encoding="utf8") as fh:
            fh.write(info)