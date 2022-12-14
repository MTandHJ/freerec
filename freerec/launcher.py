

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
from .models import RecSysArch
from .criterions import BaseCriterion
from .dict2obj import Config
from .utils import AverageMeter, Monitor, timemeter, infoLogger, errorLogger
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
    'F1': f1_score,
    'AUC': auroc,
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
    'F1': ".4f",
    'AUC': ".4f",
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
    'F1': max,
    'AUC': max,
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
        model: Union[RecSysArch, torch.nn.Module, None],
        criterion: Union[BaseCriterion, Callable],
        optimizer: Optional[torch.optim.Optimizer],
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: Union[torch.device, str, int]
    ):
        """
        Parameters:
        ---

        dataset: RecDataSet for training, validation and test
        criterion: Callable
            Loss funcion.
        model: nn.Module or None
            - `None`: Using _DummyModule instead, by which `forward` should not be called.
        
        optimizer: torch.optim.Optimizer or None
            - `None`: Using _DummyModule instead, by which `step` and `backward` should not be called.

        lr_scheduler: Callable or None
            - `None`: Using _DummyModule instead, by which `step` should not be called.

        device: torch.device, str or int
            - `torch.device`
            - `str`: Like `cpu`, `cuda:0`.
            - `int`: Using cuda:`int`.
        """
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
        ---

        epoch: int
        """
        path = os.path.join(self.cfg.CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = torch.load(path)
        for module in self.cfg.CHECKPOINT_MODULES:
            getattr(self, module).load_state_dict(checkpoint[module])
        self.monitors.load_state_dict(checkpoint['monitors'])
        return checkpoint['epoch']

    def save_best(self, path: str, prefix: str): ...

    def check_best(self, val: Optional[float]): ...

    def resume(self):
        """Resume from last checkpoint."""
        start_epoch: int = 0
        if self.cfg.resume:
            start_epoch = self.load_checkpoint() 
            infoLogger(f"[Coach] >>> Load the recent checkpoint and train from epoch: {start_epoch}")
        return start_epoch
    
    def register_metric(
        self, name: str, metric: Optional[str] = None, 
        func: Optional[Callable] = None, fmt: str = '.4f', best_caster: Callable = max,
        prefix: str = 'train'
    ):
        """Register a metric.

        Parameters:
        ---

        name: str
            The complete name of this metric, such as `NDCG@20`.
        metric: str
            The name of this metric, like `NDCG` for `NDCG@20`.
        func: Callable
            The function to process data.
        fmt: str
            The format to print.
        best_caster: `min` or `max`
            Which one is better.
        prefix: str
            The group to which this metric belongs.

        Examples
        ---

        >>> coach = Coach(...)
        >>> coach.compile(cfg, monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'])
        >>> coach.register_metric('loss2', func=None, fmt='.5f', prefix='train')
        >>> from freerec.metrics import normalized_dcg
        >>> from functools import partial
        >>> coach.register_metric(
        ...    name='NDCG@50',
        ...    metric='NDCG',
        ...    func=partial(normalized_dcg, k=50),
        ...   fmt='.4f',
        ...    prefix='test'
        ... )

        Raises:
        ---

        AttributeError: when calling `register_metric' before `compile'.
        
        """
        try:
            metric = name if metric is None else metric
            self.__monitors[prefix][metric.upper()].append(
                AverageMeter(
                    name=name,
                    metric=func,
                    fmt=fmt,
                    best_caster=best_caster
                )
            )
        except AttributeError:
            errorLogger(
                "'register_metric' should be called after 'compile' ...",
                AttributeError
            )

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
        """Load config and set monitors.
        
        Parameters:
        ---

        cfg: Config
        monitors: List[str]
            The metrics (for 'valid' and 'test' only) of interest.
        
        Examples:
        ---

        >>> coach = Coach(None)
        >>> coach.compile(cfg, monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'])
        """
        self.cfg = cfg
        # meters for train|valid|test
        self.__monitors = Monitor()
        self.__monitors['train'] = defaultdict(list)
        self.__monitors['valid'] = defaultdict(list)
        self.__monitors['test'] = defaultdict(list)

        self.register_metric(
            name='LOSS',
            metric='LOSS',
            func=DEFAULT_METRICS['LOSS'],
            fmt=DEFAULT_FMTS['LOSS'],
            best_caster=DEFAULT_BEST_CASTER['LOSS'],
            prefix='train'
        )

        for name in monitors:
            name = name.upper()
            if '@' in name:
                metric, K = name.split('@')
                for prefix in ('valid', 'test'):
                    self.register_metric(
                        name=name,
                        metric=metric,
                        func=partial(DEFAULT_METRICS[metric], k=int(K)),
                        fmt=DEFAULT_FMTS[metric],
                        best_caster=DEFAULT_BEST_CASTER[metric],
                        prefix=prefix
                    )
            else:
                metric = name
                for prefix in ('valid', 'test'):
                    self.register_metric(
                        name=name,
                        metric=metric,
                        func=DEFAULT_METRICS[metric],
                        fmt=DEFAULT_FMTS[metric],
                        best_caster=DEFAULT_BEST_CASTER[metric],
                        prefix=prefix
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
                leave=False, desc="??'???' ??-"
            )
        else:
            return self.__dataloader

    @torch.no_grad()
    def monitor(
        self, *values,
        n: int = 1, mode: str = 'mean', 
        prefix: str = 'train', pool: Optional[List] = None
    ):
        """Log some values to given monitors.

        Parameters:
        ---

        *values: data
        n: int
            Batch size in general
        mode: 'sum'|'mean' (default)
        prefix: str, 'train'|'test'|'valid'
            The mode values belonging to.
        pool: List[str]
            Given metrics.
            - `None`: `pool` will be set for all metrics.
        """
        metrics: Dict[List] = self.monitors[prefix]
        pool = metrics if pool is None else pool
        for metric in pool:
            for meter in metrics.get(metric.upper(), []):
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
            for _, meters in metrics.items():
                for meter in meters:
                    meter.plot(freq=freq)
                    imgname = meter.save(path=os.path.join(self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR), prefix=prefix)
                    epoch, val = meter.argbest(freq)
                    info += s.format(
                        prefix=prefix, metric=meter.name,
                        val=val, epoch=epoch, img=f"![]({imgname})"
                    )
                    data.append([prefix, meter.name, val, epoch])
                    if val != -1: # Only save available data.
                        best[prefix][meter.name] = val

        with open(file_, "w", encoding="utf8") as fh:
            fh.write(info) # Summary.md

        df = pd.DataFrame(data, columns=['Prefix', 'Metric', 'Best', '@Epoch'])
        infoLogger(str(df)) # print final metrics
        infoLogger(f"[LoG_PaTH] >>> {self.cfg.LOG_PATH}")
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

    Flows:
    ---

    1. compile: get command, envs and params for training;
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
        """
        Parameters:
        ---

        cfg: Config
            Including command, envs, params, defaults.
        
        Flows:
        ---

        1. Add environmental parameters to basic `command`;
        2. Register all available devices;
        3. Convert all parameters from `cfg.PARAMS`;
        4. Convert all defaults from `cfg.DEFAULTS`.
        """
        self.cfg = cfg
        piece = "\t{key}: {vals} \n"
        envs, params, defaults = "", "", ""
        for key, val in self.cfg.ENVS.items():
            if key == 'device':
                self.devices = val.split(',')
            else:
                self.cfg.COMMAND += self.get_option(key, val)
            envs += piece.format(key=key, vals=val)
        for key, vals in self.cfg.PARAMS.items():
            if isinstance(vals, (str, int, float)):
                vals = [vals]
            self.deploy_params(key, vals)
            params += piece.format(key=key, vals=vals)
        for key, val in self.cfg.DEFAULTS.items():
            self.cfg.DEFAULTS[key] = val
            defaults += piece.format(key=key, vals=val)

        cfg_infos = f"command: {self.cfg.COMMAND} \nenvs: \n{envs}params: \n{params}defaults: \n{defaults}"
        infoLogger(f"\033[0;31;47m{cfg_infos}\033[0m")
        

    def deploy_params(self, key: str, vals: Iterable):
        self.params.append(key)
        self.values.append(vals)

    @staticmethod
    def get_option(key: str, val: Any):
        """Convert (key, val) to '--key=val'.

        Parameters:
        ---

        key: str
        val: Any

        Notes:
        ---

        All '_' in key will be replaced by '-'.
        
        Examples:
        ---

        >>> Adapter.get_option('lr', '1e-3')
        --lr=1e-3
        >>> Adapter.get_option('learning_rate', '1e-3')
        --learning-rate=1e-3
        """
        return f" --{key.replace('_', '-')}={val}"

    def load_best(self, logPath: str):
        """Load best.pickle from logPath of corresponding."""
        file_ = os.path.join(logPath, self.cfg.DATA_DIR, self.cfg.MONITOR_BEST_FILENAME)
        return import_pickle(file_)

    def write(self, id_: str, logPath: str, params: Dict):
        """Write results to tensorboard.
        
        Parameters:
        ---

        id_: str
            The experiment id.
        logPath: str
            The log path of this experiment.
        params: Dict
            Parameter config of this experiemnt.
        
        Flows:
        ---

        1. Load best data from the `logPath`.
        2. Write best data to tensorboard with `params`.

        Notes:
        ---

        If you find `-1` appears in the tensorboard,
        it must be the data therein is of `str` type,
        which will raise error if we sent it to tensorboard directly !
        """
        try:
            data = self.load_best(logPath)
            path = os.path.join(self.cfg.CORE_LOG_PATH, id_)
            with SummaryWriter(log_dir=path) as writer:
                metrics = dict()
                for prefix, best in data.items():
                    for metric, val in best.items():
                        val = val if isinstance(val, (int, float)) else -1
                        metrics['/'.join([prefix, metric])] = val
                writer.add_hparams(
                    params, metrics,
                )
        except Exception:
            infoLogger(
                f"\033[0;31;47m[Adapter] >>> Unknown errors happen. This is mainly due to abnormal exits of child processes.\033[0m"
            )


    def each_grid(self):
        """Grid search for each kind of param."""
        for key, vals in zip(self.params, self.values):
            for val in vals:
                yield self.cfg.DEFAULTS | {key: val}

    def product_grid(self):
        """Grid search across all combination of params"""
        for vals in product(*self.values):
            yield self.cfg.DEFAULTS | {option:val for option, val in zip(self.params, vals)}

    def save_checkpoint(self, source: List) -> None:
        """Save the rest of params."""
        path = os.path.join(self.cfg.CORE_CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = dict()
        checkpoint['source'] = source
        torch.save(checkpoint, path)

    def load_checkpoint(self) -> int:
        """Load the rest of params."""
        path = os.path.join(self.cfg.CORE_CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = torch.load(path)
        return checkpoint['source']

    @timemeter("Coach/resume")
    def resume(self):
        """Resume from the recent checkpoint."""
        source = self.each_grid() if self.cfg.EXCLUSIVE else self.product_grid()
        source = list(source)[::-1]
        source = self.load_checkpoint() if self.cfg.resume else source
        infoLogger(f"[Coach] >>> Load the recent checkpoint ...")
        return source

    def run(self, command: str, params: Dict):
        """Start a new subprocess"""
        for option, val in params.items():
            command += self.get_option(option, val)
        infoLogger(f"\033[0;31;47m{command}\033[0m")
        return subprocess.Popen(shlex.split(command))

    def wait(self, tasks: Dict):
        """Wait util all processes terminate."""
        for process_, id_, logPath, params in tasks.values():
            process_.wait()
            self.write(id_, logPath, params)

    def poll(self, tasks: Dict):
        """Wait util any process terminates."""
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

        self.source = self.resume()
        tasks = dict()
        try:
            while self.source:
                self.poll(tasks)
                params = self.source.pop()
                device = self.devices.pop()
                command, id_, logPath = self.register(device)
                process_ = self.run(command, params)
                tasks[device] = (process_, id_, logPath, params)
        except Exception as e:
            print(e)
        finally:
            self.wait(tasks)
            sys.exit()