

import os
import argparse
import torch
import time
import yaml
from argparse import ArgumentParser

from .dict2obj import Config
from .utils import mkdirs, set_logger, set_seed, activate_benchmark, timemeter


INFO_PATH = "./infos/{description}/{id}"
LOG_PATH = "./logs/{description}/{id}"
CORE_PATH = "./logs/{description}/core"
TIME = "%m%d%H%M"

CONFIG = Config(
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu",
    BENCHMARK = True, # activate cudnn.benchmark if True
    SEED = -1, # -1 for random

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


class Parser(Config):

    def __init__(self) -> None:
        super().__init__(**CONFIG)
        self.parse()

    def readme(self, path: str, mode: str = "w") -> None:
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
        self.parser.add_argument("--config", type=str, default=None, help="config.yml")

        self.parser.add_argument("--device", type=str, default=CONFIG.DEVICE, help="device")

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
        self.parser.add_argument("--eval-freq", type=int, default=5, help="the evaluation frequency")

        self.parser.add_argument("--num-workers", type=int, default=0)
        self.parser.add_argument("--buffer-size", type=int, default=None, help="buffer size for datapipe")

        self.parser.add_argument("--seed", type=int, default=1, help="calling --seed=-1 for a random seed")
        self.parser.add_argument("--benchmark", action="store_false", default=True, help="cudnn.benchmark == True ?")
        self.parser.add_argument("--verbose", action="store_true", default=False, help="show the progress bar if true")
        self.parser.add_argument("--resume", action="store_true", default=False, help="resume the training from the recent checkpoint")

        self.parser.add_argument("--id", type=str, default=time.strftime(TIME))
        self.parser.add_argument("-m", "--description", type=str, default="RecSys")

    def add_argument(self, *args: str, **kwargs):
        args = (arg.replace('_', '-') for arg in args) # user '-' instead of '_'
        self.parser.add_argument(*args, **kwargs)

    def set_defaults(self, **kwargs):
        self.parser.set_defaults(**kwargs)

    @timemeter("Parser/compile")
    def compile(self):
        args = self.parser.parse_args()
        for key, val in args._get_kwargs():
            if key.upper() in self:
                self[key.upper()] = val
            else:
                self[key] = val
        self.load(args) # loading config (.yaml) ...
        
        self['INFO_PATH'] = INFO_PATH.format(**self)
        self['LOG_PATH'] = LOG_PATH.format(**self)
        mkdirs(self.INFO_PATH, self.LOG_PATH)
        set_logger(path=self.LOG_PATH, log2file=self.log2file, log2console=self.log2console)

        self.DEVICE = torch.device(self.DEVICE)
        activate_benchmark(self.BENCHMARK)
        set_seed(self.SEED)

        self.readme(self.INFO_PATH) # generate README.md
        self.readme(self.LOG_PATH)


class CoreParser(Parser):

    @timemeter("CoreParser/parse")
    def parse(self):

        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("config", type=str, help="config.yml")
        self.parser.add_argument("--exclusive", action="store_true", default=False, help="one by one or one for all")

        self.parser.add_argument("--root", type=str, default=None, help="data")
        self.parser.add_argument("--device", type=str, default=None, help="device")

        self.parser.add_argument("--eval-freq", type=int, default=None, help="the evaluation frequency")

        self.parser.add_argument("--num-workers", type=int, default=None)
        self.parser.add_argument("--buffer-size", type=int, default=None, help="buffer size for datapipe")

        self.parser.add_argument("--seed", type=int, default=None, help="calling --seed=-1 for a random seed")
        self.parser.add_argument("-m", "--description", type=str, default=None)


    @timemeter("Parser/load")
    def load(self, args: ArgumentParser):
        with open(args.config, encoding="UTF-8", mode='r') as f:
            config = yaml.full_load(f)
        envs = config.get('envs', None)
        envs = envs if envs else dict()
        params = config.get('params', dict())
        for key, val in args._get_kwargs():
            if key in ('config', 'exclusive') or val is None:
                continue
            else:
                envs[key] = val
        self['envs'], self['params'] = envs, params
        if 'command' in config:
            self['command'] = config['command']

    @timemeter("CoreParser/compile")
    def compile(self):
        args = self.parser.parse_args()
        for key, val in args._get_kwargs():
            if key.upper() in self:
                self[key.upper()] = val
            else:
                self[key] = val
        self.load(args)
        
        self['INFO_PATH'] = INFO_PATH
        self['LOG_PATH'] = LOG_PATH
        self['CORE_PATH'] = CORE_PATH.format(**self)
        mkdirs(self.CORE_PATH)
        set_logger(path=self.CORE_PATH, log2file=self.log2file, log2console=self.log2console)

        self.readme(self.CORE_PATH)

