

from typing import Any, Callable, Iterable, List, Dict, Optional, Tuple, Union

import torch
import pandas as pd
import os, subprocess, shlex, time, sys
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from itertools import product
from collections import defaultdict
from tqdm import tqdm
from freeplot.utils import import_pickle, export_pickle

from .data.datasets.base import BaseSet
from .data.fields import Field, Fielder
from .data.dataloader import DataLoader
from .dict2obj import Config
from .utils import AverageMeter, Monitor, timemeter, infoLogger, warnLogger, errorLogger
from .metrics import *
from .parser import TIME


__all__ = ['Coach', 'Adapter']


DEFAULT_METRICS = {
    'LOSS': None,
    #############
    'MSE': mean_squared_error,
    'MAE': mean_abs_error,
    'RMSE': root_mse,
    #############
    'PRECISION': precision,
    'RECALL': recall,
    'HITRATE': hit_rate,
    #############
    'NDCG': normalized_dcg,
    'MRR': mean_reciprocal_rank,
    'MAP': mean_average_precision
}

DEFAULT_FMTS = {
    'LOSS': ".5f",
    #############
    'MSE': ".4f",
    'MAE': ".4f",
    'RMSE': ".4f",
    #############
    'PRECISION': ".4f",
    'RECALL': ".4f",
    'HITRATE': ".4f",
    #############
    'NDCG': ".4f",
    'MRR': ".4f",
    'MAP': ".4f",
}

DEFAULT_BEST_CASTER = {
    'LOSS': min,
    #############
    'MSE': min,
    'MAE': min,
    'RMSE': min,
    #############
    'PRECISION': max,
    'RECALL': max,
    'HITRATE': max,
    #############
    'NDCG': max,
    'MRR': max,
    'MAP': max,
}

class _DummyModule(torch.nn.Module):

    def forward(self, *args, **kwargs):
        errorLogger("No model available for Coach ...", NotImplementedError)

    def step(self, *args, **kwargs):
        errorLogger("No optimizer or lr scheduler available for Coach ...", NotImplementedError)

    def backward(self, *args, **kwargs):
        errorLogger("No optimizer available for Coach ...", NotImplementedError)


class Coach:

    """The framework for training."""
    
    def __init__(
        self, 
        dataset: BaseSet,
        criterion: Callable,
        model: Optional[torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer],
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: Union[torch.device, str, int]
    ):
        self.dataset = dataset
        self.fields: Fielder[Field] = self.dataset.fields
        self.device = torch.device(device)
        self.criterion = criterion
        self.model = model.to(self.device) if model else _DummyModule()
        self.optimizer = optimizer if optimizer else _DummyModule()
        self.lr_scheduler = lr_scheduler if lr_scheduler else _DummyModule()

      
    def save(self) -> None:
        """Save the model"""
        torch.save(self.model.state_dict(), os.path.join(self.cfg.LOG_PATH, self.cfg.SAVED_FILENAME))

    def save_checkpoint(self, epoch: int) -> None:
        """Save current checkpoint at epoch X."""
        path = os.path.join(self.cfg.CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = dict()
        checkpoint['epoch'] = epoch
        for module in self.cfg.CHECKPOINT_MODULES:
            checkpoint[module] = getattr(self, module).state_dict()
        checkpoint['monitors'] = self.monitors.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self) -> int:
        """Load last saved checkpoint.

        Returns:
            The epoch.
        """
        path = os.path.join(self.cfg.CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = torch.load(path)
        for module in self.cfg.CHECKPOINT_MODULES:
            getattr(self, module).load_state_dict(checkpoint[module])
        self.monitors.load_state_dict(checkpoint['monitors'])
        return checkpoint['epoch']

    def save_best(self, path: str, prefix: str): ...

    def check_best(self, val: Optional[float]): ...

    @timemeter("Coach/resume")
    def resume(self):
        """Resume from last checkpoint."""
        start_epoch = self.load_checkpoint() if self.cfg.resume else 0
        infoLogger(f"[Coach] >>> Load the recent checkpoint and train from epoch: {start_epoch}")
        return start_epoch

    @property
    def monitors(self) -> Monitor:
        """Return Dict[str, Dict[str, List]], specifically,
        Monitor[prefix: str, Dict[metric: str, meters: List]].
        """
        return self.__monitors

    @timemeter("Coach/compile")
    def compile(
        self, cfg: Config, monitors: List[str]
    ):
        """Load config and set monitors."""
        self.cfg = cfg
        # meters for train|valid|test
        self.__monitors = Monitor()
        self.__monitors['train'] = {
            'LOSS': [
                AverageMeter(
                    name='LOSS', 
                    metric=DEFAULT_METRICS['LOSS'], 
                    fmt=DEFAULT_FMTS['LOSS']
                )
            ]
        }
        self.__monitors['valid'] = defaultdict(list)
        self.__monitors['test'] = defaultdict(list)

        for name in monitors:
            name = name.upper()
            if '@' in name:
                metric, K = name.split('@')
                for prefix in ('valid', 'test'):
                    self.__monitors[prefix][metric].append(
                        AverageMeter(
                            name=name,
                            metric=partial(DEFAULT_METRICS[metric], k=int(K)),
                            fmt=DEFAULT_FMTS[metric]
                        )
                    )
            else:
                metric = name
                for prefix in ('valid', 'test'):
                    self.__monitors[prefix][metric].append(
                        AverageMeter(
                            name=name,
                            metric=DEFAULT_METRICS[metric],
                            fmt=DEFAULT_FMTS[metric]
                        )
                    )

        # dataloader
        self.load_dataloader()

    def load_dataloader(self):
        self.__dataloader = DataLoader(
            datapipe=self.dataset, num_workers=self.cfg.num_workers
        )
    
    @property
    def dataloader(self):
        if self.cfg.verbose:
            return tqdm(
                self.__dataloader,
                leave=False, desc="վ'ᴗ' ի-"
            )
        else:
            return self.__dataloader

    def monitor(
        self, *values,
        n: int = 1, mode: str = 'mean', 
        prefix: str = 'train', pool: Optional[List] = None
    ):
        """Log some values to given monitors.

        Args:
            values: data
        Kwargs:
            n: batch size in general
            mode: 'sum'|'mean' (default)
            prefix: the mode those values belonging to
            pool: given monitors
        """
        metrics: Dict[List] = self.monitors[prefix]
        for metric in pool:
            for meter in metrics.get(metric, []):
                meter(*values, n=n, mode=mode)

    def step(self, epoch: int):
        """Print information and reset them."""
        for prefix, metrics in self.monitors.items():
            metrics: Dict[str, List[AverageMeter]]
            infos = [f"[Coach] >>> {prefix.upper():5} @Epoch: {epoch:<4d} >>> "]
            for meters in metrics.values():
                infos += [meter.step() for meter in meters if meter.active]
            infoLogger(' || '.join(infos))

    @timemeter("Coach/summary")
    def summary(self):
        """Summary the whole training process.
        The following information will be saved:
            1. historical evaluation saved in monitors;
            2. best historical results;
            3. curves of historical results.
        """
        file_ = os.path.join(self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR, self.cfg.SUMMARY_FILENAME)
        s = "|  {prefix}  |   {metric}   |   {val}   |   {epoch}   |   {img}   |\n"
        info = ""
        info += "|  Prefix  |   Metric   |   Best   |   @Epoch   |   Img   |\n"
        info += "| :-------: | :-------: | :-------: | :-------: | :-------: |\n"
        data = []
        best = defaultdict(dict)

        for prefix, metrics in self.monitors.items():
            metrics: defaultdict[str, List[AverageMeter]]
            freq = 1 if prefix == 'train' else self.cfg.EVAL_FREQ
            for METRIC, meters in metrics.items():
                for meter in meters:
                    meter.plot(freq=freq)
                    imgname = meter.save(path=os.path.join(self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR), prefix=prefix)
                    epoch, val = meter.argbest(DEFAULT_BEST_CASTER[METRIC], freq)
                    info += s.format(
                        prefix=prefix, metric=meter.name,
                        val=val, epoch=epoch, img=f"![]({imgname})"
                    )
                    data.append([prefix, meter.name, val, epoch])
                    best[prefix][meter.name] = val

        with open(file_, "w", encoding="utf8") as fh:
            fh.write(info) # Summary.md

        df = pd.DataFrame(data, columns=['Prefix', 'Metric', 'Best', '@Epoch'])
        infoLogger(str(df)) # print final metrics
        # save corresponding data for next analysis
        self.monitors.write(os.path.join(self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR)) # tensorboard
        self.monitors.save(os.path.join(self.cfg.LOG_PATH, self.cfg.DATA_DIR), self.cfg.MONITOR_FILENAME)
        export_pickle(best, os.path.join(self.cfg.LOG_PATH, self.cfg.DATA_DIR, self.cfg.MONITOR_BEST_FILENAME))

    def train_per_epoch(self):
        errorLogger("train_per_epoch should be specified ...", NotImplementedError)

    def evaluate(self, prefix: str = 'valid'):
        errorLogger("evaluate should be specified ...", NotImplementedError)


    @timemeter("Coach/train")
    def train(self):
        self.dataset.train()
        self.model.train()
        return self.train_per_epoch()

    @timemeter("Coach/valid")
    @torch.no_grad()
    def valid(self):
        self.dataset.valid() # TODO: multiprocessing pitfall ???
        self.model.eval()
        return self.evaluate(prefix='valid')

    @timemeter("Coach/test")
    @torch.no_grad()
    def test(self):
        self.dataset.test()
        self.model.eval()
        return self.evaluate(prefix='test')

    @timemeter("Coach/fit")
    def fit(self):
        """Start the training and log some useful information."""
        start_epoch = self.resume()
        for epoch in range(start_epoch, self.cfg.epochs):
            if epoch % self.cfg.CHECKPOINT_FREQ == 0:
                self.save_checkpoint(epoch)
            if epoch % self.cfg.EVAL_FREQ == 0:
                if self.cfg.EVAL_VALID:
                    self.check_best(self.valid())
                if self.cfg.EVAL_TEST:
                    self.test()
            self.train()

            self.step(epoch)
        self.save()

        # last epoch
        self.check_best(self.valid())
        self.test()
        self.step(self.cfg.epochs)

        self.summary()


class CoachForCTR(Coach): ...

class CoachForMatching(Coach): ...


class Adapter:
    """ Params tuner.

    It proceeds as follows:
    1. compile -> get config: command, envs and params for training;
    2. allocate devices for various params
        - register id, logPath, device first
        - run it
        - collect information from logPath and output to tensorbaord
        - save checkpoint
        - release corresponding device

    Examples:
    ---

    >>> cfg = {'command': 'python xxx.py', 'params': {'optimizer': ['sgd', 'adam']}}
    >>> tuner = Adapter()
    >>> tuner.compile(cfg)
    >>> tuner.fit()
    """

    def __init__(self) -> None:
        self.params = []
        self.values = []
        self.devices = []

    @property
    def COMMAND(self):
        return self.cfg.COMMAND

    def register(self, device: str) -> Tuple[str, str]:
        self.cfg.ENVS['id'] = time.strftime(TIME)
        self.cfg.ENVS['device'] = device
        command = self.COMMAND + self.get_option('id', self.cfg.ENVS.id)
        command += self.get_option('device', self.cfg.ENVS.device)
        return command, self.cfg.ENVS.id, self.cfg.LOG_PATH.format(**self.cfg.ENVS)

    @timemeter("Adapter/compile")
    def compile(self, cfg: Config):
        def safe_cast(vals):
            for caster in (int, float, str):
                try:
                    return list(map(caster, vals))
                except ValueError:
                    continue
        self.cfg = cfg
        for key, val in self.cfg.ENVS.items():
            if key == 'device':
                self.devices = list(val.split(','))
            else:
                self.cfg.COMMAND += self.get_option(key, val)
        for key, vals in self.cfg.PARAMS.items():
            if isinstance(vals, (str, int, float)):
                vals = (vals, )
            self.deploy_params(key, safe_cast(vals))
        for key, val in self.cfg.DEFAULTS.items():
            self.cfg.DEFAULTS[key] = safe_cast([val])[0]

    def deploy_params(self, key: str, vals: Iterable):
        self.params.append(key)
        self.values.append(vals)

    def get_option(self, key: str, val: Any):
        """Convert (key, val) to '--key=val'."""
        return f" --{key.replace('_', '-')}={val}"

    def load_best(self, logPath: str):
        """load best.pickle from logPath of corresponding """
        file_ = os.path.join(logPath, self.cfg.DATA_DIR, self.cfg.MONITOR_BEST_FILENAME)
        return import_pickle(file_)

    def write(self, id_: str, logPath: str, params: Dict):
        """Write results to tensorboard"""
        data = self.load_best(logPath)
        path = os.path.join(self.cfg.CORE_LOG_PATH, id_)
        with SummaryWriter(log_dir=path) as writer:
            metrics = dict()
            for prefix, best in data.items():
                for metric, val in best.items():
                    metrics['/'.join([prefix, metric])] = val
            writer.add_hparams(
                params, metrics,
            )

    def each_grid(self):
        """Grid search for each kind of param"""
        for key, vals in zip(self.params, self.values):
            for val in vals:
                yield self.cfg.DEFAULTS | {key: val}

    def product_grid(self):
        """Grid search across all combination of params"""
        for vals in product(*self.values):
            yield self.cfg.DEFAULTS | {option:val for option, val in zip(self.params, vals)}

    def save_checkpoint(self, source: List) -> None:
        """Save the rest of params"""
        path = os.path.join(self.cfg.CORE_CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = dict()
        checkpoint['source'] = source
        torch.save(checkpoint, path)

    def load_checkpoint(self) -> int:
        """Load the rest of params"""
        path = os.path.join(self.cfg.CORE_CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = torch.load(path)
        return checkpoint['source']

    @timemeter("Coach/resume")
    def resume(self):
        """Resume from last checkpoint"""
        source = self.each_grid() if self.cfg.EXCLUSIVE else self.product_grid()
        source = list(source)[::-1]
        source = self.load_checkpoint() if self.cfg.resume else source
        infoLogger(f"[Coach] >>> Load the recent checkpoint ...")
        return source

    def run(self, command: str, params: Dict):
        """Start a new subprocess"""
        for option, val in params.items():
            command += self.get_option(option, val)
        warnLogger(command)
        return subprocess.Popen(shlex.split(command))

    def wait(self, tasks: Dict):
        """Wait util all processes terminate"""
        for process_, id_, logPath, params in tasks.values():
            process_.wait()
            self.write(id_, logPath, params)

    def poll(self, tasks: Dict):
        """Wait util any one of processes terminate"""
        buffer_source = []
        time.sleep(1) # for unique id
        while len(self.devices) == 0:
            time.sleep(7)
            for device, (process_, id_, logPath, params) in tasks.items():
                if process_.poll() is not None:
                    self.write(id_, logPath, params)
                    self.devices.append(device)
                else:
                    buffer_source.append(params)
        self.save_checkpoint(self.source + buffer_source)

    @timemeter("Adapter/fit")
    def fit(self):
        """Grid search."""
        try:
            self.source = self.resume()
            tasks = dict()
            while self.source:
                self.poll(tasks)
                params = self.source.pop()
                device = self.devices.pop()
                command, id_, logPath = self.register(device)
                process_ = self.run(command, params)
                tasks[device] = (process_, id_, logPath, params)
            self.wait(tasks)
        except Exception as e:
            print(e)
        finally:
            sys.exit()