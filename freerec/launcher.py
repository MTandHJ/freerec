

from typing import Any, Literal, Callable, Iterable, List, Dict, Optional, Tuple

import torch, abc, os, time, sys, signal
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from itertools import product
from collections import defaultdict

from .data.datasets import RecDataSet
from .data.postprocessing import PostProcessor
from .data.fields import Field, FieldTuple
from .data.tags import USER, ITEM, ID, UNSEEN, SEEN
from .models import RecSysArch
from .dict2obj import Config
from .utils import AverageMeter, Monitor, timemeter, infoLogger, import_pickle, export_pickle
from .metrics import *
from .parser import TIME, Parser
from .ddp import is_main_process, main_process_only, is_distributed, synchronize


__all__ = [
    'ChiefCoach', 'Coach', 'Adapter'
]


DEFAULT_METRICS = {
    'LOSS': lambda x: x,
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
    """This is a dummy module that serves as a placeholder for a real model."""
    def forward(self, *args, **kwargs):
        """Dummy forward method that raises a `NotImplementedError`."""
        raise NotImplementedError("No model available for Coach ...")

    def step(self, *args, **kwargs):
        """Dummy step method that raises a `NotImplementedError`."""
        raise NotImplementedError("No optimizer or lr scheduler available for Coach ...")

    def backward(self, *args, **kwargs):
        """Dummy backward method that raises a `NotImplementedError`."""
        raise NotImplementedError("No optimizer available for Coach ...")


class ChiefCoach(metaclass=abc.ABCMeta):
    r""" 
    The `ChiefCoach` class is the top-level class for running the training and evaluation loops.

    Parameters:
    -----------
    dataset: RecDataSet,
        The original dataset.
    trainpipe : IterDataPipe
        Iterable data pipeline for training data.
    validpipe : IterDataPipe, optional
        Iterable data pipeline for validation data.
    testpipe : IterDataPipe, optional
        Iterable data pipeline for testing data.
        If `None`, use `validpipe` instead.
    model : RecSysArch
        Model for training and evaluating. 
    cfg: Parser
        Configuration file.
    """

    def __init__(
        self, *,
        dataset: RecDataSet,
        trainpipe: PostProcessor, validpipe: PostProcessor, testpipe: Optional[PostProcessor], 
        model: RecSysArch, cfg: Parser
    ):

        self.cfg = cfg
        self.__mode = 'train'

        self.set_device(self.cfg.device)
        self.set_dataset(dataset)
        self.set_datapipe(
            trainpipe, validpipe, testpipe
        )
        self.set_dataloader()

        self.set_model(model)
        self.set_optimizer()
        self.set_lr_scheduler()
        self.reset_monitors(self.cfg.monitors, self.cfg.which4best)

        # Other setup can be placed here
        self.set_other()

    def set_device(self, device):
        self.device = torch.device(device)

    def set_dataset(self, dataset: RecDataSet):
        self.dataset = dataset
        self.fields: FieldTuple[Field] = FieldTuple(dataset.fields)
        self.User = self.fields[USER, ID]
        self.Item = self.fields[ITEM, ID]
        self.ISeen = self.Item.fork(SEEN)
        self.IUnseen = self.Item.fork(UNSEEN)

    def set_datapipe(
        self,
        trainpipe,
        validpipe,
        testpipe=None,
    ):
        self.trainpipe = trainpipe
        self.validpipe = validpipe
        self.testpipe = self.validpipe if testpipe is None else testpipe

    def set_model(
        self, model: RecSysArch
    ):
        self.model = model.to(self.device)
        if is_distributed():
            self.model = torch.nn.parallel.DistributedDataParallel(model)
    
    def set_optimizer(self):
        if self.cfg.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.cfg.lr, 
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Unexpected optimizer {self.cfg.optimizer} ..."
            )

    def set_lr_scheduler(self):
        self.lr_scheduler = _DummyModule()

    def set_dataloader(self) -> None:
        from torchdata.dataloader2 import (
            DataLoader2, 
            MultiProcessingReadingService, DistributedReadingService, 
            SequentialReadingService
        )

        def get_reading_servie():
            if is_distributed():
                rs = SequentialReadingService(
                    DistributedReadingService(),
                    MultiProcessingReadingService(self.cfg.num_workers)
                )
            else:
                rs = MultiProcessingReadingService(self.cfg.num_workers)
            return rs

        self.trainloader = DataLoader2(
            datapipe=self.trainpipe, 
            reading_service=get_reading_servie()
        )
        self.validloader = DataLoader2(
            datapipe=self.validpipe, 
            reading_service=get_reading_servie()
        )
        self.testloader = DataLoader2(
            datapipe=self.testpipe, 
            reading_service=get_reading_servie()
        )

    def set_other(self):
        ...
    
    def get_res_sys_arch(self) -> RecSysArch:
        model = self.model
        if isinstance(self.model, torch.nn.parallel.DataParallel):
            model = self.model.module
        elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        assert isinstance(model, RecSysArch), "No RecSysArch found ..."
        return model

    @property
    def fields(self) -> FieldTuple[Field]:
        return self.__fields

    @fields.setter
    def fields(self, fields):
        self.__fields = FieldTuple(fields)

    @property
    def mode(self):
        """Get the current mode of the chief coach."""
        return self.__mode

    @timemeter
    def train(self, epoch: int):
        """Start training and return the training loss."""
        self.__mode = 'train'
        self.model.train()
        return self.train_per_epoch(epoch)

    @timemeter
    @torch.no_grad()
    def valid(self, epoch: int):
        """Start validation and return the validation metrics."""
        self.__mode = 'valid'
        self.model.eval()
        return self.evaluate(epoch=epoch, mode='valid')

    @timemeter
    @torch.no_grad()
    def test(self, epoch: int):
        """Start testing and return the test metrics."""
        self.__mode = 'test'
        self.model.eval()
        return self.evaluate(epoch=epoch, mode='test')

    @property
    def dataloader(self):
        if self.mode == 'train':
            return self.trainloader
        elif self.mode == 'valid':
            return self.validloader
        else:
            return self.testloader

    def shutdown(self):
        self.trainloader.shutdown()
        self.validloader.shutdown()
        self.testloader.shutdown()

    @abc.abstractmethod
    def train_per_epoch(self, epoch: int):
        raise NotImplementedError(
            f"{self.__class__.__name__}.train_per_epoch() should be implemented ..."
        )

    @abc.abstractmethod
    def evaluate(self, epoch: int, mode: str = 'valid'):
        raise NotImplementedError(
            f"{self.__class__.__name__}.evaluate() should be implemented ..."
        )

    def register_metric(
        self, name: str, func: Callable, 
        fmt: str = '.4f', best_caster: Callable = max
    ) -> None:
        r"""
        Register a metric.

        Parameters
        ----------
        name : str
            The complete name of the metric, such as `LOSS2`.
            The notation `@` should not be included, i.e., 'LOSS@2' is invalid.
        func : Callable
            The function to process the data for the metric.
        fmt : str, optional
            The format to use when printing the metric, defaults to `'.4f'`.
        best_caster : Callable, optional
            A function used to cast the best value of the metric, defaults to `max`.
            
        Returns
        -------
        None
        
        Raises
        ------
        AssertionError
            When `name` has already been registered or contains the notation `@`.
        """

        name = name.upper()
        assert DEFAULT_METRICS.get(name, None) is None, f"The metric {name} already exists ..."
        assert '@' not in name, f"The metric name has invalid notation of `@' ..."
        DEFAULT_METRICS[name] = func
        DEFAULT_FMTS[name] = fmt
        DEFAULT_BEST_CASTER[name] = best_caster

        for mode in ('train', 'valid', 'test'):
            self._set_monitor(
                name=name,
                lastname=name,
                mode=mode
            )

    def _set_monitor(
        self, name: str, lastname: str, mode: str = 'train', **kwargs
    ):
        """Add a monitor for the specified metric."""
        try:
            meter = AverageMeter(
                    name=name,
                    metric=partial(DEFAULT_METRICS[lastname], **kwargs),
                    fmt=DEFAULT_FMTS[lastname],
                    best_caster=DEFAULT_BEST_CASTER[lastname]
                )
            self.__monitors[mode][lastname].append(meter)
        except KeyError:
            raise KeyError(
                f"The metric of {lastname} is not included. "
                f"You can register by calling `register_metric(...)' ..."
            )
        return meter

    @property
    def monitors(self) -> Monitor:
        """Return the monitor dictionary for the different modes ('train', 'valid', 'test')."""
        return self.__monitors

    def reset_monitors(
        self, monitors: List[str], which4best: str = 'LOSS'
    ):
        r"""
        Set up monitors for training.

        Parameters
        ----------
        monitors : List[str]
            A list of metric names to be monitored during training.
        which4best : str, defaults `LOSS'
            The metric used for selecting the best checkpoint.

        Examples
        --------
        >>> coach: Coach
        >>> coach.compile(monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'])
        """
        assert isinstance(monitors, List), f"'monitors' should be a list but {type(monitors)} received ..."
        assert isinstance(which4best, str), f"'which4best' should be a str but {type(which4best)} received ..."

        # meters for train|valid|test
        self.__monitors = Monitor()
        self.__monitors['train'] = defaultdict(list)
        self.__monitors['valid'] = defaultdict(list)
        self.__monitors['test'] = defaultdict(list)

        # UPPER
        which4best = which4best.upper()
        monitors = ['LOSS'] + [name.upper() for name in monitors] + [which4best]
        monitors = sorted(set(monitors), key=monitors.index)

        for name in monitors:
            for mode in ('train', 'valid', 'test'):
                if '@' in name:
                    lastname, K = name.split('@')
                    meter = self._set_monitor(
                        name=name,
                        lastname=lastname,
                        mode=mode,
                        k=int(K)
                    )
                else:
                    lastname = name
                    meter = self._set_monitor(
                        name=name,
                        lastname=lastname,
                        mode=mode
                    )
                if mode == 'valid' and name == which4best:
                    self.meter4best = meter
                    self._best = -float('inf') if meter.caster is max else float('inf')
                    self._best_epoch = 0


class Coach(ChiefCoach):


    @property
    def meter4best(self):
        return self.__best_meter

    @meter4best.setter
    def meter4best(self, meter: AverageMeter):
        self.__best_meter = meter
        infoLogger(f"[Coach] >>> Set best meter: {meter.name} ")

    @property
    def remove_seen(self):
        # remove seen if
        # 1) retain_seen is not activated and
        # 2) dataset has no duplicates
        return not (self.cfg.retain_seen or self.dataset.has_duplicates())

    def save(self, filename: Optional[str] = None) -> None:
        r"""
        Save the model to `LOG_PATH` with a given filename.

        Parameters:
        -----------
        filename: str, optional
            `None`: Use `SAVED_FILENAME`
        """
        if is_main_process():
            filename = self.cfg.SAVED_FILENAME if filename is None else filename
            torch.save(self.model.state_dict(), os.path.join(self.cfg.LOG_PATH, filename))

        synchronize()
        return 

    def load(self, path: str, filename: Optional[str] = None) -> None:
        filename = self.cfg.SAVED_FILENAME if filename is None else filename
        self.model.load_state_dict(
            torch.load(os.path.join(path, filename), map_location=self.device)
        )

        synchronize()
        return

    def save_checkpoint(self, epoch: int) -> None:
        r"""
        Save current checkpoint at epoch.

        Parameters:
        -----------
            epoch :int Current epoch number.

        Returns:
        --------
            None
        """
        path = os.path.join(self.cfg.CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = dict()
        checkpoint['epoch'] = epoch
        for module in self.cfg.CHECKPOINT_MODULES:
            checkpoint[module] = getattr(self, module).state_dict()
        checkpoint['monitors'] = self.monitors.state_dict()
        torch.save(checkpoint, path)

        synchronize()
        return

    def load_checkpoint(self) -> int:
        r"""
        Load last saved checkpoint.

        Returns:
        --------
        epoch: int 
            The epoch number loaded from the checkpoint.
        """
        path = os.path.join(self.cfg.CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = torch.load(path)
        for module in self.cfg.CHECKPOINT_MODULES:
            getattr(self, module).load_state_dict(checkpoint[module])
        self.monitors.load_state_dict(checkpoint['monitors'])

        synchronize()
        return checkpoint['epoch']

    def save_best(self) -> None:
        self.save(self.cfg.BEST_FILENAME)

    def load_best(self) -> None:
        infoLogger(f"[Coach] >>> Load best model @Epoch {self._best_epoch:<4d} ")
        self.model.load_state_dict(torch.load(os.path.join(self.cfg.LOG_PATH, self.cfg.BEST_FILENAME)))

        synchronize()
        return

    def check_best(self, epoch: int) -> None:
        """Update best value."""
        if self.meter4best.active:
            best_ = self.meter4best.which_is_better(self._best)
            if best_ != self._best:
                self._best = best_
                self._best_epoch = epoch
                infoLogger(f"[Coach] >>> Better ***{self.meter4best.name}*** of ***{self._best:.4f}*** ")
                self.save_best()

    def eval_at_best(self):
        try:
            self.load_best()
            self.valid(self._best_epoch)
            self.test(self._best_epoch)
            self.step(self._best_epoch)
            self.load(self.cfg.LOG_PATH, self.cfg.SAVED_FILENAME)
        except FileNotFoundError:
            infoLogger(f"[Coach] >>> No best model was recorded. Skip it ...")

    @main_process_only
    def easy_record_best(self, best: defaultdict):
        r"""
        Record the best results on test set.
        It make easy to watch on tensorboard.
        """

        for lastname, meters in self.monitors['test'].items():
            for meter in meters:
                # Skip those meters never activated.
                if len(meter.history) == 0:
                    continue
                # Note that meter.history[-1] is the result at the best checkpoint.
                val = meter.history[-1]
                best['best'][meter.name] = val

        export_pickle(best, os.path.join(self.cfg.LOG_PATH, self.cfg.DATA_DIR, self.cfg.MONITOR_BEST_FILENAME))

    def resume(self) -> int:
        r"""
        Resume training from the last checkpoint.

        Returns:
        --------
        start_epoch: int
            The epoch number to resume training from.
        """
        start_epoch: int = 0
        if self.cfg.resume:
            start_epoch = self.load_checkpoint()
            infoLogger(f"[Coach] >>> Load last checkpoint and train from epoch: {start_epoch}")
        return start_epoch

    @torch.no_grad()
    def monitor(
        self, *values,
        n: int = 1, reduction: str = 'mean', 
        mode: Literal['train', 'valid', 'test'] = 'train', 
        pool: Optional[Iterable] = None
    ):

        r"""
        Log data values to specific monitors.

        Parameters:
        -----------
        *values : data
            The data values to be logged.
        n : int
            The batch size in general.
        reduction : str, optional
            The reduction to compute the metric. Can be 'sum' or 'mean' (default).
        mode : str, optional
            The mode string indicating which mode the values belong to. Can be 'train', 'test' or 'valid'.
        pool : List[str], optional
            A list of metric names to log. If None, all metrics in the pool of `mode` will be logged.
        """

        metrics: Dict[List] = self.monitors[mode]
        pool = metrics if pool is None else pool
        for lastname in pool:
            for meter in metrics.get(lastname.upper(), []):
                meter(*values, n=n, reduction=reduction)

    def step(self, epoch: int):
        r"""
        Prints training status and evaluation results for each epoch, 
        and resets the corresponding `AverageMeter` instances.

        Parameters:
        -----------
        epoch : int
            The epoch number.

        Returns:
        --------
        None
        """
        metrics: Dict[str, List[AverageMeter]]
        for mode, metrics in self.monitors.items():
            infos = [f"[Coach] >>> {mode.upper():5} @Epoch: {epoch:<4d} >>> "]
            for meters in metrics.values():
                infos += [meter.step() for meter in meters if meter.active]
            infoLogger(' || '.join(infos))

    @main_process_only
    @timemeter
    def summary(self):
        r"""
        Summary the whole training process.

        Generate a summary of the entire training process, including the historical evaluation results, the best
        historical results, and the curves of historical results. The resulting summary is saved to a Markdown file named
        "Summary.md" in the `self.cfg.LOG_PATH` directory.

        Additionally, the best historical results are saved to a binary file named `self.cfg.MONITOR_BEST_FILENAME`.
        """
        import pandas as pd

        s = "|  {mode}  |   {name}   |   {val}   |   {epoch}   |   {img}   |\n"
        info = ""
        info += "|  Mode  |   Metric   |   Best   |   @Epoch   |   Img   |\n"
        info += "| :-------: | :-------: | :-------: | :-------: | :-------: |\n"
        data = []
        best = defaultdict(dict)

        for mode, metrics in self.monitors.items():
            metrics: defaultdict[str, List[AverageMeter]]
            freq = 1 if mode == 'train' else self.cfg.eval_freq
            for lastname, meters in metrics.items():
                for meter in meters:
                    # Skip those meters never activated.
                    if len(meter.history) == 0:
                        continue
                    meter.plot(freq=freq)
                    imgname = meter.save(path=os.path.join(self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR), mode=mode)
                    epoch, val = meter.argbest(freq)
                    info += s.format(
                        mode=mode, name=meter.name,
                        val=val, epoch=epoch, img=f"![]({imgname})"
                    )
                    data.append([mode, meter.name, val, epoch])
                    if val != -1: # Only save available data.
                        best[mode][meter.name] = val

        file_ = os.path.join(self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR, self.cfg.SUMMARY_FILENAME)
        with open(file_, "w", encoding="utf8") as fh:
            fh.write(info)

        df = pd.DataFrame(data, columns=['Mode', 'Metric', 'Best', '@Epoch'])
        infoLogger(str(df))
        infoLogger(f"[LoG_PaTH] >>> {self.cfg.LOG_PATH}")

        self.monitors.write(os.path.join(self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR)) # tensorboard
        self.monitors.save(os.path.join(self.cfg.LOG_PATH, self.cfg.DATA_DIR), self.cfg.MONITOR_FILENAME)

        return best

    def dict_to_device(self, data: Dict[Field, Any]) -> Dict[Field, Any]:
        return {field: value.to(self.device) if isinstance(value, torch.Tensor) else value for field, value in data.items()}

    def evaluate(self, epoch: int, mode: str = 'valid'):
        self.get_res_sys_arch().reset_ranking_buffers()
        for data in self.dataloader:
            data = self.dict_to_device(data)
            users = data[self.User]
            if self.cfg.ranking == 'full':
                scores = self.model(data, ranking='full')
                if self.remove_seen:
                    seen = self.Item.to_csr(data[self.ISeen]).to(self.device).to_dense().bool()
                    scores[seen] = -1e23
                targets = self.Item.to_csr(data[self.IUnseen]).to(self.device).to_dense()
            elif self.cfg.ranking == 'pool':
                scores = self.model(data, ranking='pool')
                targets = torch.zeros_like(scores)
                targets[:, 0].fill_(1)
            else:
                raise NotImplementedError(
                    f"`ranking` should be 'full' or 'pool' but {self.cfg.ranking} received ..."
                )

            self.monitor(
                scores, targets,
                n=len(users), reduction="mean", mode=mode,
                pool=['HITRATE', 'PRECISION', 'RECALL', 'NDCG', 'MRR']
            )
            
    @timemeter
    def fit(self):

        start_epoch = self.resume()
        for epoch in range(start_epoch, self.cfg.epochs):
            if epoch % self.cfg.CHECKPOINT_FREQ == 0:
                self.save_checkpoint(epoch)
            if epoch % self.cfg.eval_freq == 0:
                if self.cfg.eval_valid:
                    self.valid(epoch)
                if self.cfg.eval_test:
                    self.test(epoch)
            self.check_best(epoch)
            self.step(epoch)
            self.train(epoch)

        self.save()

        # last epoch
        self.valid(self.cfg.epochs)
        self.test(self.cfg.epochs)

        self.check_best(self.cfg.epochs)
        self.step(self.cfg.epochs)

        best = self.summary()

        self.eval_at_best()
        self.easy_record_best(best)

        self.shutdown()


class Adapter:
    r"""
    Params tuner.

    Flows:
    ------
    1. compile: configure the command, environments, and parameters for training.
    2. allocate devices for various parameters:
        - register the ID, log path, and device first
        - execute the command
        - collect information from the log path and output to TensorBoard
        - save the checkpoint
        - release the corresponding device

    Examples:
    ---------
    >>> cfg = {'command': 'python xxx.py', 'params': {'optimizer': ['sgd', 'adam']}}
    >>> tuner = Adapter()
    >>> tuner.compile(cfg)
    >>> tuner.fit()
    """

    def __init__(self) -> None:
        self.params = []
        self.values = []
        self.devices = tuple()

    @property
    def COMMAND(self):
        return self.cfg.COMMAND

    def register(self, device: str) -> Tuple[str, str]:
        self.cfg.ENVS['id'] = time.strftime(TIME)
        self.cfg.ENVS['device'] = device
        command = self.COMMAND + self.get_option('id', self.cfg.ENVS.id)
        command += self.get_option('device', self.cfg.ENVS.device)
        return command, self.cfg.ENVS.id, self.cfg.LOG_PATH.format(**self.cfg.ENVS)

    @timemeter
    def compile(self, cfg: Config) -> None:
        r"""
        Configure the command, environments, and parameters for training.

        Parameters:
        -----------
        cfg : Config
            An object that contains the command, environments, parameters, and defaults.

        Flows:
        ------
        1. Add environmental parameters to the basic `command`.
        2. Register all available devices.
        3. Convert all parameters from `cfg.PARAMS`.
        4. Convert all defaults from `cfg.DEFAULTS`.

        Returns:
        --------
        None
        """
        self.cfg = cfg
        piece = "\t{key}: {vals} \n"
        envs, params, defaults = "", "", ""
        for key, val in self.cfg.ENVS.items():
            if key == 'device':
                self.devices = tuple(val.split(','))
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
        r"""
        Convert (key, val) to '--key=val'.

        Parameters:
        -----------
        key : str
            The key of the parameter.
        val : Any
            The value of the parameter.

        Notes:
        ------
        All '_' in `key` will be replaced by '-'.

        Returns:
        --------
        str
            The parameter with format '--key=val'.

        Examples:
        ---------
        >>> Adapter.get_option('lr', '1e-3')
        '--lr=1e-3'
        >>> Adapter.get_option('learning_rate', '1e-3')
        '--learning-rate=1e-3'
        """
        return f" --{key.replace('_', '-')}={val}"

    def load_best(self, logPath: str):
        """Load best.pkl from logPath of corresponding."""
        file_ = os.path.join(logPath, self.cfg.DATA_DIR, self.cfg.MONITOR_BEST_FILENAME)
        return import_pickle(file_)

    def write(self, id_: str, logPath: str, params: Dict):
        r"""
        Write experiment results to tensorboard.

        Parameters:
        -----------
        id_: str
            Experiment ID.
        logPath: str
            Path to the experiment logs.
        params: Dict
            Configuration parameters of the experiment.

        Flows:
        ------
        1. Load the best data from `logPath`.
        2. Write the best data to tensorboard with `params`.

        Notes:
        ------
        If you find `-1` appearing in the tensorboard,
        it could mean that the data is of `str` type,
        which will cause an error if it is sent to tensorboard directly!
        """
        try:
            data = self.load_best(logPath)
            path = os.path.join(self.cfg.CORE_LOG_PATH, id_)
            with SummaryWriter(log_dir=path) as writer:
                metrics = dict()
                for mode, best in data.items():
                    for metric, val in best.items():
                        val = val if isinstance(val, (int, float)) else -1
                        metrics['/'.join([mode, metric])] = val
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

    @timemeter
    def load_checkpoint(self) -> int:
        """Load the rest of params."""
        infoLogger(f"[Coach] >>> Load the recent checkpoint ...")
        path = os.path.join(self.cfg.CORE_CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = torch.load(path)
        return checkpoint['source']

    def resume(self):
        """Resume from the recent checkpoint."""
        source = self.each_grid() if self.cfg.EXCLUSIVE else self.product_grid()
        source = list(source)[::-1]
        source = self.load_checkpoint() if self.cfg.resume else source
        return source

    def run(self, command: str, params: Dict):
        """Start a new subprocess"""
        import subprocess, shlex
        for option, val in params.items():
            command += self.get_option(option, val)
        infoLogger(f"\033[0;31;47m{command}\033[0m")
        return subprocess.Popen(shlex.split(command))

    def wait(self, tasks: List):
        """Wait util all processes terminate."""
        tasks = [task for task in tasks if task is not None]
        for process_, id_, logPath, params in tasks:
            process_.wait()
            self.write(id_, logPath, params)

    def poll(self, tasks: List):
        """Wait util any process terminates."""
        def is_null(task):
            return task is None
        buffer_source = [task[-1] for task in tasks if task is not None]
        time.sleep(1) # for unique id
        while not any(map(is_null, tasks)):
            time.sleep(7)
            buffer_source = []
            for i, (process_, id_, logPath, params) in enumerate(tasks):
                if process_.poll() is not None:
                    self.write(id_, logPath, params)
                    tasks[i] = None
                else:
                    buffer_source.append(params)
        self.save_checkpoint(self.source + buffer_source)
        return tasks.index(None)

    def terminate(self, tasks: List):
        tasks = [task for task in tasks if task is not None]
        for process_, _, _, _ in tasks:
            if process_.poll() is None:
                process_.terminate()
        time.sleep(3)
        for process_, _, _, _ in tasks:
            if process_.poll() is None:
                process_.kill()
        sys.exit()

    @timemeter
    def fit(self):
        """Grid search."""
        self.source = self.resume()
        tasks = [None for _ in range(len(self.devices))]

        def signal_handler(sig, frame):
            infoLogger(f"\033[0;31;47m===============================TERMINATE ALL SUBPROCESSES===============================\033[0m")
            self.terminate(tasks)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            while self.source:
                index = self.poll(tasks)
                device = self.devices[index]
                params = self.source.pop()
                command, id_, logPath = self.register(device)
                process_ = self.run(command, params)
                tasks[index] = (process_, id_, logPath, params)
        finally:
            self.wait(tasks)
            self.terminate(tasks)