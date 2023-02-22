

import torch
import os, argparse, time, yaml
from argparse import ArgumentParser

from .dict2obj import Config
from .utils import (
    mkdirs, timemeter, set_color, set_seed, activate_benchmark, 
    set_logger
)


"""
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
--------------
BENCHMARK : bool
    If True, activate cudnn.benchmark.
SEED : int
    The random seed used by PyTorch. If set to -1, uses a random seed.
EVAL_FREQ : int
    The frequency of evaluation.
EVAL_VALID : bool
    If True, evaluate on the validation set.
EVAL_TEST : bool
    If True, evaluate on the test set.
log2file : bool
    If True, save logs to a file.
log2console : bool
    If True, display logs on the console.
SAVED_FILENAME : str
    The filename of saved model parameters.
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
LOG_PATH = "./logs/{description}/{dataset}/{device}-{id}"
CORE_CHECKPOINT_PATH = "./infos/{description}/core"
CORE_LOG_PATH = "./logs/{description}/core"
TIME = "%m%d%H%M%S"

CONFIG = Config(
    BENCHMARK = True,
    SEED = 1,

    # evaluation
    EVAL_FREQ = 5,
    EVAL_VALID = True,
    EVAL_TEST = False,

    # logger
    log2file = True,
    log2console = True,

    # path|file
    SAVED_FILENAME = "model.pt",
    CHECKPOINT_FREQ = 1,
    CHECKPOINT_MODULES = ['model', 'optimizer', 'lr_scheduler'],
    CHECKPOINT_FILENAME = "checkpoint.tar",
    SUMMARY_FILENAME = "SUMMARY.md",
    MONITOR_FILENAME = "monitors.pickle",
    MONITOR_BEST_FILENAME = "best.pickle",
    description = "RecSys"
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
    log2console = True,
)


class Parser(Config):
    """ArgumentParser wrapper."""

    def __init__(self) -> None:
        super().__init__(**CONFIG)
        self.parse()

    def readme(self, path: str, mode: str = "w") -> None:
        """Add README.md to the path."""
        time_ = time.strftime("%Y-%m-%d-%H:%M:%S")
        file_ = os.path.join(path, "README.md")
        s = "|  {key}  |   {val}    |\n"
        info = "\n## {0} \n\n\n".format(time_)
        info += "|  Attribute   |   Value   |\n"
        info += "| :-------------: | :-----------: |\n"
        for key, val in self.items():
            info += s.format(key=key, val=val)
        with open(file_, mode, encoding="utf8") as fh:
            fh.write(info)

    def reset(self):
        self.clear()
        for key, val in CONFIG.items():
            self[key] = val

    @timemeter("Parser/parse")
    def parse(self):
        """Add command-line arguments to the parser."""

        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--root", type=str, default=".", help="data")
        self.parser.add_argument("--dataset", type=str, default="RecDataSet", help="useless if no need to automatically select a dataset")
        self.parser.add_argument("--config", type=str, default=None, help="config.yml")

        self.parser.add_argument("--device", default=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu', help="device")

        # model
        self.parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer: adam (default), sgd, ...")
        self.parser.add_argument("--nesterov", action="store_true", default=False, help="nesterov for SGD")
        self.parser.add_argument("-mom", "--momentum", type=float, default=0.9, help="the momentum used for SGD")
        self.parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="the first beta argument for Adam")
        self.parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="the second beta argument for Adam")
        self.parser.add_argument("-wd", "--weight-decay", type=float, default=None, help="weight for 'l1|l2|...' regularzation")
        self.parser.add_argument("-lr", "--lr", "--LR", "--learning-rate", type=float, default=None)
        self.parser.add_argument("-b", "--batch-size", type=int, default=None)
        self.parser.add_argument("--epochs", type=int, default=None)

        # eval
        self.parser.add_argument("--eval-valid", action="store_false", default=True, help="evaluate validset")
        self.parser.add_argument("--eval-test", action="store_true", default=False, help="evaluate testset")
        self.parser.add_argument("--eval-freq", type=int, default=CONFIG.EVAL_FREQ, help="the evaluation frequency")

        self.parser.add_argument("--num-workers", type=int, default=4)

        self.parser.add_argument("--seed", type=int, default=CONFIG.SEED, help="calling --seed=-1 for a random seed")
        self.parser.add_argument("--benchmark", action="store_false", default=True, help="cudnn.deterministic == True ?")
        self.parser.add_argument("--verbose", action="store_true", default=False, help="show the progress bar if true")
        self.parser.add_argument("--resume", action="store_true", default=False, help="resume the training from the recent checkpoint")

        self.parser.add_argument("--id", type=str, default=time.strftime(TIME))
        self.parser.add_argument("-m", "--description", type=str, default=CONFIG.description)

    def add_argument(self, *args: str, **kwargs):
        """
        Add an argument to the parser.

        Parameters
        ----------
        *args : str
            The name(s) of the argument.
        **kwargs
            Additional keyword arguments to pass to `parser.add_argument`.
        
        Notes:
        ------
        Any character `_' will be replaced by `-'.
        """
        args = (arg.replace('_', '-') for arg in args) # user '-' instead of '_'
        self.parser.add_argument(*args, **kwargs)

    def set_defaults(self, **kwargs):
        """Set the default values for the arguments.

        Parameters
        ----------
        **kwargs
            The default values to set.
        """
        self.parser.set_defaults(**kwargs)

    @timemeter("Parser/load")
    def load(self, args: ArgumentParser):
        """Load config.yaml.

        Parameters
        ----------
        args : argparse.ArgumentParser
            An instance of argparse.ArgumentParser containing the arguments
            passed to the script.

        Raises
        ------
        KeyError
            If the parameter key is not recognized.
        """
        if hasattr(args, 'config') and args.config:
            with open(args.config, encoding="UTF-8", mode='r') as f:
                for key, val in yaml.full_load(f).items():
                    if key.upper() in self:
                        self[key.upper()] = val
                    elif key in self:
                        self[key] = val
                    else:
                        KeyError(f"Unexpected parameter of {key} from {args.config} ...")

    @timemeter("Parser/compile")
    def compile(self):
        """
        Generate the configuration file according to the specified settings.

        Flows:
        ------
        1. Load the default settings from the parsed command-line arguments (ArgumentParser).
        2. If the `--config` flag has been specified, load settings from a .yaml file, which may override the 
        default settings.
        3. Generate the paths for saving checkpoints and collecting training information:
            - CHECKPOINT_PATH: the path where checkpoints will be saved.
            - LOG_PATH: the path where training information will be collected.
        4. Configure the logger so that information can be logged by `info|debug|warnLogger`.
        5. Add a README.md file under CHECKPOINT_PATH and LOG_PATH.
        """
        args = self.parser.parse_args()
        self.load(args) # loading config (.yaml) first ...
        for key, val in args._get_kwargs():
            if key.upper() in self:
                self[key.upper()] = val
            else:
                self[key] = val

        try:
            self.device = int(self.device)
        except ValueError:
            ...
        
        self['DATA_DIR'] = DATA_DIR
        self['SUMMARY_DIR'] = SUMMARY_DIR
        self['CHECKPOINT_PATH'] = CHECKPOINT_PATH.format(**self)
        self['LOG_PATH'] = LOG_PATH.format(**self)
        mkdirs(
            self.CHECKPOINT_PATH, self.LOG_PATH,
            os.path.join(self.LOG_PATH, self.DATA_DIR),
            os.path.join(self.LOG_PATH, self.SUMMARY_DIR)
        )
        set_color(self.device)
        set_logger(path=self.LOG_PATH, log2file=self.log2file, log2console=self.log2console)

        activate_benchmark(self.BENCHMARK)
        self.SEED = set_seed(self.SEED)

        self.readme(self.CHECKPOINT_PATH) # create README.md
        self.readme(self.LOG_PATH)


class CoreParser(Config):
    """CoreParser class to parse command-line arguments and configuration files."""

    ALL_ENVS = (
        'description', 'root', 'device', 'eval_freq', 'num_workers'
    )

    def __init__(self) -> None:
        super().__init__(**CORE_CONFIG)
        self.parse()

    @timemeter("CoreParser/parse")
    def parse(self):
        """Parse command-line arguments."""

        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("description", type=str, help="...")
        self.parser.add_argument("config", type=str, help="config.yml")
        self.parser.add_argument("--exclusive", action="store_true", default=False, help="one by one or one for all")

        self.parser.add_argument("--root", type=str, default=None, help="data")
        self.parser.add_argument("--dataset", type=str, default=None, help="useless if no need to automatically select a dataset")
        self.parser.add_argument("--device", type=str, default=None, help="device")

        self.parser.add_argument("--eval-freq", type=int, default=None, help="the evaluation frequency")

        self.parser.add_argument("--num-workers", type=int, default=None)

        self.parser.add_argument("--resume", action="store_true", default=False, help="resume the search from the recent checkpoint")


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
            learning_rate: [1e-3, 1e-2, 1e-1]
            weight_decay: [0, 1e-4, 2e-4, 5e-4]
            batch_size: [128, 256, 512, 1024]
        defaults:
            optimizer: adam
            learning_rate: 1e-3
            weight_decay: 0
            batch_size: 256
            epochs: 100
            seed: 1

        where 'command' is necessary but 'envs', 'params' and 'defaults' are optional ...

        Notes: when calling '--exclusive' for grid search one by one, 
            'defaults' is required for clear comparsions in tensorbaord.
        """
        if self.COMMAND is None:
            ValueError(template)
        for key in ('root', 'device'):
            if self.ENVS.get(key, None) is None:
                KeyError(f"No {key} is allocated, calling '--{key}' to specify it")

        if self.ENVS.get('dataset', None) is None:
            self.ENVS['dataset'] = "RecDataSet"
        self.ENVS = Config(self.ENVS)
        self.PARAMS = Config(self.PARAMS)

    @timemeter("Parser/load")
    def load(self, args: ArgumentParser) -> None:
        """
        Load configuration file.

        Parameters
        ----------
        args : ArgumentParser
            The command line arguments.

        Returns
        -------
        None
        """
        with open(args.config, encoding="UTF-8", mode='r') as f:
            config = {key.upper(): vals for key, vals in yaml.full_load(f).items()}
            self.update(**config)
        self.EXCLUSIVE = args.exclusive
        self.resume = args.resume
        for key, val in args._get_kwargs():
            if key in self.ALL_ENVS and val is not None:
                self.ENVS[key] = val

    @timemeter("CoreParser/compile")
    def compile(self) -> None:
        """
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
        args = self.parser.parse_args()
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
        """
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