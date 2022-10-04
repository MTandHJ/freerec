

import torch
import os, argparse, time, yaml
from argparse import ArgumentParser

from .dict2obj import Config
from .utils import mkdirs, set_logger, set_color, set_seed, activate_benchmark, timemeter, warnLogger


DATA_DIR = 'data'
SUMMARY_DIR = 'summary'
CHECKPOINT_PATH = "./infos/{description}/{dataset}/{device}"
LOG_PATH = "./logs/{description}/{dataset}/{device}-{id}"
CORE_CHECKPOINT_PATH = "./infos/{description}/core"
CORE_LOG_PATH = "./logs/{description}/core"
TIME = "%m%d%H%M%S"

CONFIG = Config(
    BENCHMARK = True, # activate cudnn.benchmark if True
    SEED = 1, # -1 for random

    # evaluation
    EVAL_FREQ = 5,
    EVAL_VALID = True,
    EVAL_TEST = False,

    # logger
    log2file = True,
    log2console = True,

    # path|file
    SAVED_FILENAME = "model.pt", # the filename of saved model paramters
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


def _root2dataset(root: str):
    return root.split('/')[-1]


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

    @timemeter("Parser/load")
    def load(self, args: ArgumentParser):
        """Load config.yaml."""
        if hasattr(args, 'config') and args.config:
            with open(args.config, encoding="UTF-8", mode='r') as f:
                for key, val in yaml.full_load(f).items():
                    if key.upper() in self:
                        self[key.upper()] = val
                    else:
                        self[key] = val

    def reset(self):
        self.clear()
        for key, val in CONFIG.items():
            self[key] = val

    @timemeter("Parser/parse")
    def parse(self):

        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--root", type=str, default=".", help="data")
        self.parser.add_argument("--dataset", type=str, default=None, help="useless if no need to automatically select a dataset")
        self.parser.add_argument("--config", type=str, default=None, help="config.yml")

        self.parser.add_argument("--device", default=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu', help="device")

        # model
        self.parser.add_argument("--optimizer", type=str, choices=("sgd", "adam"), default="adam")
        self.parser.add_argument("--nesterov", action="store_true", default=False, help="nesterov for SGD")
        self.parser.add_argument("-mom", "--momentum", type=float, default=0.9, help="the momentum used for SGD")
        self.parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="the first beta argument for Adam")
        self.parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="the second beta argument for Adam")
        self.parser.add_argument("-wd", "--weight-decay", type=float, default=1e-4, help="weight for 'l1|l2|...' regularzation")
        self.parser.add_argument("-lr", "--lr", "--LR", "--learning-rate", type=float, default=0.001)
        self.parser.add_argument("-b", "--batch-size", type=int, default=256)
        self.parser.add_argument("--epochs", type=int, default=1000)

        # eval
        self.parser.add_argument("--eval-valid", action="store_false", default=True, help="evaluate validset")
        self.parser.add_argument("--eval-test", action="store_true", default=False, help="evaluate testset")
        self.parser.add_argument("--eval-freq", type=int, default=CONFIG.EVAL_FREQ, help="the evaluation frequency")

        self.parser.add_argument("--num-workers", type=int, default=0)

        self.parser.add_argument("--seed", type=int, default=CONFIG.SEED, help="calling --seed=-1 for a random seed")
        self.parser.add_argument("--benchmark", action="store_false", default=True, help="cudnn.benchmark == True ?")
        self.parser.add_argument("--verbose", action="store_true", default=False, help="show the progress bar if true")
        self.parser.add_argument("--resume", action="store_true", default=False, help="resume the training from the recent checkpoint")

        self.parser.add_argument("--id", type=str, default=time.strftime(TIME))
        self.parser.add_argument("-m", "--description", type=str, default=CONFIG.description)

    def add_argument(self, *args: str, **kwargs):
        args = (arg.replace('_', '-') for arg in args) # user '-' instead of '_'
        self.parser.add_argument(*args, **kwargs)

    def set_defaults(self, **kwargs):
        self.parser.set_defaults(**kwargs)

    @timemeter("Parser/compile")
    def compile(self):
        """Generate config file according to settings.

        Flows:
        ---

        1. Load (default) settings from parsed args (ArgumentParser).
        2. Load settings from xxx.yaml if --config has been activated. 
        Note that this step might override the settings above.
        3. Generate basic paths:
            - CHECKPOINT_PATH: saving checkpoints
            - LOG_PATH: collecting training infomation

        4. Set Logger, and then you can log information by info|debug|warn|errorLogger ...
        5. Finally, READMD.md will be added under CHECKPOINT_PATH and LOG_PATH both.
        """
        args = self.parser.parse_args()
        for key, val in args._get_kwargs():
            if key.upper() in self:
                self[key.upper()] = val
            else:
                self[key] = val
        self.load(args) # loading config (.yaml) ...

        if self.dataset is None:
            self.dataset = _root2dataset(self.root)
        try:
            self.device = int(self.device)
        except ValueError:
            ...


        activate_benchmark(self.BENCHMARK)
        self.SEED = set_seed(self.SEED)
        
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

        self.readme(self.CHECKPOINT_PATH) # generate README.md
        self.readme(self.LOG_PATH)


class CoreParser(Config):
    ALL_ENVS = (
        'description', 'root', 'device', 'eval_freq', 'num_workers'
    )

    def __init__(self) -> None:
        super().__init__(**CORE_CONFIG)
        self.parse()

    @timemeter("CoreParser/parse")
    def parse(self):

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


    def check(self):
        """Check the validity of the given config."""
        template = """
        Please make sure the configuration file follows the template below:

        command: python xxx.py
        envs:
            root: ../../data
            dataset: MovieLens1M
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
            raise NotImplementedError(warnLogger(template))
        for key in ('root', 'device'):
            if self.ENVS.get(key, None) is None:
                raise NotImplementedError(
                    warnLogger(f"No device is allocated, calling '--{key}' to specify it")
                )
        if self.ENVS.get('dataset', None) is None:
            self.ENVS['dataset'] = _root2dataset(self.ENVS['root'])
        self.ENVS = Config(self.ENVS)
        self.PARAMS = Config(self.PARAMS)

    @timemeter("Parser/load")
    def load(self, args: ArgumentParser):
        """Load config."""
        with open(args.config, encoding="UTF-8", mode='r') as f:
            config = {key.upper(): vals for key, vals in yaml.full_load(f).items()}
            self.update(**config)
        self.EXCLUSIVE = args.exclusive
        self.resume = args.resume
        for key, val in args._get_kwargs():
            if key in self.ALL_ENVS and val is not None:
                self.ENVS[key] = val

    @timemeter("CoreParser/compile")
    def compile(self):
        """Generate config file according to settings.

        Flows:
        ---

        1. Load settings from xxx.yaml which provides parameters for grid searching.
        2. Load settings of the execution environment from parsed args (ArgumentParser).
        3. Generate basic paths:
            - CHECKPOINT_PATH: subprocess
            - LOG_PATH: subprocess
            - CORE_CHECKPOINT_PATH: saving checkpoints of the rest of params
            - CORE_LOG_PATH: saving best results of each subprocess for comparison

        4. Set Logger, and then you can log information by info|debug|warn|errorLogger ...
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
        """Add README.md to the path."""
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