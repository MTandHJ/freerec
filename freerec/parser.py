

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
    Path for saving checkpoints.
LOG_PATH : str
    Path for saving logging information.
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
    The filename of saved model parameters.
BEST_FILENAME : str
    The filename of saved best model parameters.
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
    """ArgumentParser wrapper."""

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
        super().__init__(**CONFIG)
        self.parse()

    @main_process_only
    def readme(self, path: str, mode: str = "w") -> None:
        """Add README.md to the path."""
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
        self.clear()
        for key, val in CONFIG.items():
            self[key] = val

    @timemeter
    def parse(self):
        """Add command-line arguments to the parser."""

        self.parser = argparse.ArgumentParser()

        self.add_argument("--root", type=str, default=".", help="data path")
        self.add_argument("--dataset", type=str, default="RecDataSet", help="useless if no need to automatically select a dataset")
        self.add_argument("--tasktag", type=str, choices=('MATCHING', 'NEXTITEM'), default=None, help="to specify a tasktag for dataset")
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
        r"""
        Add an argument to the parser.

        Parameters:
        -----------
        *args : str
            The name(s) of the argument.
        **kwargs
            Additional keyword arguments to pass to `parser.add_argument`.
       
        Notes:
        ------
        Any character `_' will be replaced by `-'.
        """
        args = (arg.replace('_', '-') for arg in args) # user '-' instead of '_'
        action = self.parser.add_argument(*args, **kwargs)
        self[action.dest] = action.default

    def set_defaults(self, **kwargs):
        r"""
        Set the default values for the arguments.

        Parameters:
        -----------
        **kwargs
            The default values to set.
        """
        self.parser.set_defaults(**kwargs)
        for key, val in kwargs.items():
            self[key] = val

    def set_device(self, device: Union[torch.device, str, int]):
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
        import torch.distributed as dist
        if is_distributed():
            dist.init_process_group(backend=self.ddp_backend, init_method="env://")
            self.device = int(os.environ["LOCAL_RANK"])
            # synchronize ids
            self.id = all_gather(self.id)[0]
            infoLogger(f"[DDP] >>> DDP is activated ...")

    def set_tasktag(self):
        from .data.tags import TaskTags
        if self.tasktag is not None:
            self['tasktag'] = TaskTags(self.tasktag.upper())
    
    @timemeter
    def load(self):
        r"""
        Load config.yaml.

        Raises:
        -------
        KeyError
            If the parameter key is not recognized.
        
        Returns:
        --------
        ArgParser:
            Some of defaults are overwrited by the config file
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
        r"""
        Generate the configuration file according to the specified settings.

        Flows:
        ------
        1. If the `--config` flag has been specified, load settings from a .yaml file.
        2. Generate the paths for saving checkpoints and collecting training information:
            - CHECKPOINT_PATH: the path where checkpoints will be saved.
            - LOG_PATH: the path where training information will be collected.
        3. Configure the logger so that information can be logged by `info|debug|warnLogger`.
        4. Add a README.md file under CHECKPOINT_PATH and LOG_PATH.
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
    """CoreParser class to parse command-line arguments and configuration files."""

    ALL_ENVS = (
        'description', 'root', 'device', 'num_workers'
    )

    def __init__(self) -> None:
        super().__init__(**CORE_CONFIG)

    def check(self) -> None:
        """Check the validity of the given config."""
        template = """
        Please make sure the configuration file follows the template below:

        command: python xxx.py
        envs:
            root: ../../data
            dataset: Gowalla_m1
            device: 0,1,2,3
            eval_freq: 5
            num_workers: 0
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
        r"""
        Load configuration file.

        Parameters:
        -----------
        args : ArgumentParser
            The command line arguments.

        Returns:
        --------
        None
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
        r"""
        Generate config file according to settings.

        Flows:
        ------
        1. Load settings from xxx.yaml which provides parameters for grid searching.
        2. Load settings of the execution environment from parsed args (ArgumentParser).
        3. Generate basic paths:
            - CHECKPOINT_PATH: subprocess
            - LOG_PATH: subprocess
            - CORE_CHECKPOINT_PATH: saving checkpoints of the rest of params
            - CORE_LOG_PATH: saving best results of each subprocess for comparison
        4. Set Logger, and then you can log information by info|debug|warnLogger ...
        5. Finally, READMD.md will be added under CHECKPOINT_PATH and LOG_PATH both.
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
        r"""
        Add README.md to the given path.

        Parameters:
        -----------
        path: str 
            The path to add the README.md file.
        mode: str, optional 
            The file opening mode. Defaults to "w".

        Returns:
        --------
        None
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