



import argparse
import torch
import time
import yaml
from argparse import ArgumentParser

from .dict2obj import Config
from .utils import mkdirs, set_logger, set_seed, activate_benchmark, timemeter


INFO_PATH = "./infos/{description}"
LOG_PATH = "./logs/{description}-{TIME}"
TIME = "%m%d%H"

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
    description = "RecSys"

)


class Parser(Config):

    def __init__(self) -> None:
        super().__init__(**CONFIG)
        self.parse()

    def readme(self, path: str, mode: str = "w") -> None:
        time_ = time.strftime("%Y-%m-%d-%H:%M:%S")
        filename = path + "/README.md"
        s = "|  {key}  |   {val}    |\n"
        info = "\n## {0} \n\n\n".format(time_)
        info += "|  Attribute   |   Value   |\n"
        info += "| :-------------: | :-----------: |\n"
        for key, val in self.items():
            info += s.format(key=key, val=val)
        with open(filename, mode, encoding="utf8") as fh:
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
        self.parser.add_argument("--config", type=str, default=None, help=".yml")

        # model
        self.parser.add_argument("--optimizer", type=str, choices=("sgd", "adam"), default="sgd")
        self.parser.add_argument("--nesterov", action="store_true", default=False, help="nesterov for SGD")
        self.parser.add_argument("-mom", "--momentum", type=float, default=0.9, help="the momentum used for SGD")
        self.parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="the first beta argument for Adam")
        self.parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="the second beta argument for Adam")
        self.parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4, help="weight decay")
        self.parser.add_argument("-lr", "--lr", "--LR", "--learning_rate", type=float, default=0.1)
        self.parser.add_argument("-b", "--batch_size", type=int, default=128)
        self.parser.add_argument("--epochs", type=int, default=10)

        # eval
        self.parser.add_argument("--eval-valid", action="store_false", default=True, help="evaluate validset")
        self.parser.add_argument("--eval-test", action="store_true", default=False, help="evaluate testset")
        self.parser.add_argument("--eval-freq", type=int, default=5, help="the evaluation frequency")

        self.parser.add_argument("--num-workers", type=int, default=0)
        self.parser.add_argument("--buffer-size", type=int, default=6400, help="buffer size for datapipe")

        self.parser.add_argument("--seed", type=int, default=1, help="calling --seed=-1 for a random seed")
        self.parser.add_argument("--benchmark", action="store_false", default=True, help="cudnn.benchmark == True ?")
        self.parser.add_argument("--progress", action="store_true", default=False, help="show the progress if true")
        self.parser.add_argument("--resume", action="store_true", default=False, help="resume the training from the recent checkpoint")

        self.parser.add_argument("--fmt", type=str, default="{description}={optimizer}-{lr}-{weight_decay}={seed}")
        self.parser.add_argument("-m", "--description", type=str, default="RecSys")

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def set_defaults(self, **kwargs):
        self.parser.set_defaults(**kwargs)

    @timemeter("Parser/compile")
    def compile(self):
        args = self.parser.parse_args()
        args.description = args.fmt.format(**args.__dict__)
        for key, val in args._get_kwargs():
            if key.upper() in self:
                self[key.upper()] = val
            else:
                self[key] = val
        self.load(args) # loading config (.yaml) ...
        
        self['TIME'] = time.strftime(TIME)
        self['INFO_PATH'] = INFO_PATH.format(**self)
        self['LOG_PATH'] = LOG_PATH.format(**self)
        mkdirs(self.INFO_PATH, self.LOG_PATH)
        set_logger(path=self.LOG_PATH, log2file=self.log2file, log2console=self.log2console)

        self.DEVICE = torch.device(self.DEVICE)
        activate_benchmark(self.BENCHMARK)
        set_seed(self.SEED)

        self.readme(self.INFO_PATH) # generate README.md
        self.readme(self.LOG_PATH)