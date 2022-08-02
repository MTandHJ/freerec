





# Here are some basic settings.
# It could be overwritten if you want to specify
# some configs. However, please check the correspoding
# codes in loadopts.py.



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

    # checkpoint
    SAVE_FREQ = 1,
    EVAL_FREQ = 5,
    EVAL_TRAIN = False,
    EVAL_VALID = True,

    # logger
    log2file = True, 
    log2console = True,

    # PATH
    SAVED_FILENAME = "paras.pt", # the filename of saved model paramters
    description = "RecSys"
)


class Parser(Config):

    def __init__(self) -> None:
        super().__init__(**CONFIG)

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

    @timemeter("Parser/compile")
    def compile(self, args: ArgumentParser):
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
