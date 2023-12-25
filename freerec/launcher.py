

from typing import Any, Callable, Iterable, List, Dict, Optional, Tuple, Union

import torch, abc, os, time, sys, signal, psutil, atexit
import torch.distributed as dist
from torchdata.datapipes.iter import IterDataPipe
from torch.utils.data.graph_settings import get_all_graph_pipes
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from itertools import product
from collections import defaultdict
from freeplot.utils import import_pickle, export_pickle

from .data.fields import FieldModule, FieldTuple
from .data.dataloader import DataLoader
from .models import RecSysArch
from .criterions import BaseCriterion
from .dict2obj import Config
from .utils import AverageMeter, Monitor, timemeter, infoLogger
from .metrics import *
from .parser import TIME, Parser
from .ddp import main_process_only, is_distributed, shared_random_seed, all_gather, synchronize


__all__ = [
    'ChiefCoach', 'Coach', 'Adapter',
    'GenCoach', 'SeqCoach', 'SessCoach'
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
    trainpipe : IterDataPipe
        Iterable data pipeline for training data.
    validpipe : IterDataPipe, optional
        Iterable data pipeline for validation data.
        If `None`, use `trainpipe` instead.
    testpipe : IterDataPipe, optional
        Iterable data pipeline for testing data.
        If `None`, use `validpipe` instead.
    fields : Iterable[FieldModule]
        Tuple of `FieldModule`s for dataset fields.
    model : Union[RecSysArch, torch.nn.Module, None]
        Model for training and evaluating. 
        If `None`, use _DummyModule instead, which should not call `forward`.
    criterion : Union[BaseCriterion, Callable]
        Callable for computing the loss function.
    optimizer : torch.optim.Optimizer, optional
        Optimizer for updating model parameters. 
        If `None`, use _DummyModule instead, which should not call `step` and `backward`.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler. If `None`, use _DummyModule instead, 
        which should not call `step`.
    device : Union[torch.device, str, int]
        Device on which to run the computation. 
            - `torch.device`
            - `str`: Like `cpu`, `cuda:0`.
            - `int`: Using cuda:`int`.
    """


    def __init__(
        self, *,
        trainpipe: IterDataPipe, validpipe: Optional[IterDataPipe], testpipe: Optional[IterDataPipe], fields: Iterable[FieldModule],
        model: Union[RecSysArch, torch.nn.Module, None], criterion: Union[BaseCriterion, Callable], 
        optimizer: Optional[torch.optim.Optimizer], lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: Union[torch.device, str, int]
    ):

        self.fields: FieldTuple[FieldModule] = FieldTuple(fields)
        self.device = torch.device(device)

        self._set_datapipe(trainpipe, validpipe, testpipe)
        self._set_other(model, criterion, optimizer, lr_scheduler)

        self.__mode = 'train'

        def clean():
            if is_distributed(): # clean up DDP
                dist.destroy_process_group()
            parent = psutil.Process(os.getpid())
            children = parent.children(recursive=True)
            for process in children:
                process.send_signal(signal.SIGTERM)
            psutil.wait_procs(children, timeout=5)

        atexit.register(clean)

    def _set_datapipe(
        self,
        trainpipe,
        validpipe=None,
        testpipe=None,
    ):
        """Set the data pipe for training, validation and test."""
        self.trainpipe = trainpipe
        self.validpipe = self.trainpipe if validpipe is None else validpipe
        self.testpipe = self.validpipe if testpipe is None else testpipe

    def _set_other(
        self,
        model=None, criterion=None, optimizer=None, lr_scheduler=None,
    ):
        """Set the other necessary components."""
        self.criterion = criterion
        self.model = model.to(self.device) if model else _DummyModule()
        self.optimizer = optimizer if optimizer else _DummyModule()
        self.lr_scheduler = lr_scheduler if lr_scheduler else _DummyModule()

    def get_res_sys_arch(self) -> RecSysArch:
        model = self.model
        if isinstance(self.model, torch.nn.parallel.DataParallel):
            model = self.model.module
        elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        assert isinstance(model, RecSysArch), "No RecSysArch found ..."
        return model

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
        return self.evaluate(epoch=epoch, prefix='valid')

    @timemeter
    @torch.no_grad()
    def test(self, epoch: int):
        """Start testing and return the test metrics."""
        self.__mode = 'test'
        self.model.eval()
        return self.evaluate(epoch=epoch, prefix='test')

    @abc.abstractmethod
    def train_per_epoch(self, epoch: int):
        raise NotImplementedError(
            f"{self.__class__.__name__}.train_per_epoch() should be implemented ..."
        )

    @abc.abstractmethod
    def evaluate(self, epoch: int, prefix: str = 'valid'):
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


class Coach(ChiefCoach):
    """The framework for training."""

    def seed_worker(self):
        """Set seed to keep consistent across differents ranks."""
        if is_distributed() and dist.is_initialized():
            datapipe = self.trainpipe
            if isinstance(datapipe, IterDataPipe):
                graph = torch.utils.data.graph.traverse(datapipe, only_datapipe=True)
                for pipe in get_all_graph_pipes(graph):
                    if hasattr(pipe, "set_seed"):
                        pipe.set_seed(
                            shared_random_seed()
                        )

    def prepare_dataloader(self) -> None:
        """Prepare data loaders for training, validation, and testing data."""
        self.trainloader = DataLoader(
            datapipe=self.trainpipe, 
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        self.validloader = DataLoader(
            datapipe=self.validpipe, 
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        self.testloader = DataLoader(
            datapipe=self.testpipe, 
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
    
    @property
    def dataloader(self):
        """Return the corresponding data loader depending on the current mode."""
        if self.mode == 'train':
            return self.trainloader
        elif self.mode == 'valid':
            return self.validloader
        else:
            return self.testloader

    @property
    def monitors(self) -> Monitor:
        """Return the monitor dictionary for the different modes ('train', 'valid', 'test')."""
        return self.__monitors

    @property
    def meter4best(self):
        return self.__best_meter

    @meter4best.setter
    def meter4best(self, meter: AverageMeter):
        self.__best_meter = meter
        infoLogger(f"[Coach] >>> Set best meter: {meter.name} ")

    @timemeter
    def compile(
        self, cfg: Parser, monitors: List[str], 
        which4best: str = 'LOSS'
    ):
        r"""
        Load the configuration and set up monitors for training.

        Parameters
        ----------
        cfg : Config
            A configuration object with the training details.
        monitors : List[str]
            A list of metric names to be monitored during training.
        which4best : str, defaults `LOSS'
            The metric used for selecting the best checkpoint.

        Examples
        --------
        >>> coach: Coach
        >>> coach.compile(cfg, monitors=['loss', 'recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'])
        """
        self.cfg = cfg
        # meters for train|valid|test
        self.__monitors = Monitor()
        self.__monitors['train'] = defaultdict(list)
        self.__monitors['valid'] = defaultdict(list)
        self.__monitors['test'] = defaultdict(list)

        def set_monitor(
            name: str, lastname: str, prefix: str = 'train', **kwargs
        ):
            """Add a monitor for the specified metric."""
            try:
                meter = AverageMeter(
                        name=name,
                        metric=partial(DEFAULT_METRICS[lastname], **kwargs),
                        fmt=DEFAULT_FMTS[lastname],
                        best_caster=DEFAULT_BEST_CASTER[lastname]
                    )
                self.__monitors[prefix][lastname].append(meter)
            except KeyError:
                raise KeyError(
                    f"The metric of {lastname} is not included. "
                    f"You can register by calling `register_metric(...)' ..."
                )
            return meter

        # UPPER
        which4best = which4best.upper()
        monitors = ['LOSS'] + [name.upper() for name in monitors] + [which4best]
        monitors = sorted(set(monitors), key=monitors.index)

        for name in monitors:
            for prefix in ('train', 'valid', 'test'):
                if '@' in name:
                    lastname, K = name.split('@')
                    meter = set_monitor(
                        name=name,
                        lastname=lastname,
                        prefix=prefix,
                        k=int(K)
                    )
                else:
                    lastname = name
                    meter = set_monitor(
                        name=name,
                        lastname=lastname,
                        prefix=prefix
                    )
                if prefix == 'valid' and name == which4best:
                    self.meter4best = meter
                    self._best = -float('inf') if meter.caster is max else float('inf')
                    self._best_epoch = 0

        # Prepare data loaders
        self.prepare_dataloader()

    @main_process_only
    def save(self) -> None:
        """Save the model"""
        torch.save(self.model.state_dict(), os.path.join(self.cfg.LOG_PATH, self.cfg.SAVED_FILENAME))

    def load(self, path: str, filename: Optional[str] = None, **kwargs) -> None:
        filename = self.cfg.SAVED_FILENAME if filename is None else filename
        self.model.load_state_dict(torch.load(os.path.join(path, filename), **kwargs))

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

    @main_process_only
    def save_best(self) -> None:
        torch.save(self.model.state_dict(), os.path.join(self.cfg.LOG_PATH, self.cfg.BEST_FILENAME))

    def load_best(self) -> None:
        infoLogger(f"[Coach] >>> Load best model @Epoch {self._best_epoch:<4d} ")
        self.model.load_state_dict(torch.load(os.path.join(self.cfg.LOG_PATH, self.cfg.BEST_FILENAME)))

        synchronize()
        return

    @main_process_only
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

    @main_process_only
    @torch.no_grad()
    def monitor(
        self, *values,
        n: int = 1, mode: str = 'mean', 
        prefix: str = 'train', pool: Optional[Iterable] = None
    ):

        r"""
        Log data values to specific monitors.

        Parameters:
        -----------
        *values : data
            The data values to be logged.
        n : int
            The batch size in general.
        mode : str, optional
            The mode to compute the metric. Can be 'sum' or 'mean' (default).
        prefix : str, optional
            The prefix string indicating which mode the values belong to. Can be 'train', 'test' or 'valid'.
        pool : List[str], optional
            A list of metric names to log. If None, all metrics in the pool of `prefix` will be logged.
        """

        metrics: Dict[List] = self.monitors[prefix]
        pool = metrics if pool is None else pool
        for lastname in pool:
            for meter in metrics.get(lastname.upper(), []):
                meter(*values, n=n, mode=mode)

    @main_process_only
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
        for prefix, metrics in self.monitors.items():
            infos = [f"[Coach] >>> {prefix.upper():5} @Epoch: {epoch:<4d} >>> "]
            for meters in metrics.values():
                infos += [meter.step() for meter in meters if meter.active]
            infoLogger(' || '.join(infos))

    @timemeter
    @main_process_only
    def summary(self):
        r"""
        Summary the whole training process.

        Generate a summary of the entire training process, including the historical evaluation results, the best
        historical results, and the curves of historical results. The resulting summary is saved to a Markdown file named
        "Summary.md" in the `self.cfg.LOG_PATH` directory.

        Additionally, the best historical results are saved to a binary file named `self.cfg.MONITOR_BEST_FILENAME`.
        """
        import pandas as pd

        s = "|  {prefix}  |   {name}   |   {val}   |   {epoch}   |   {img}   |\n"
        info = ""
        info += "|  Prefix  |   Metric   |   Best   |   @Epoch   |   Img   |\n"
        info += "| :-------: | :-------: | :-------: | :-------: | :-------: |\n"
        data = []
        best = defaultdict(dict)

        for prefix, metrics in self.monitors.items():
            metrics: defaultdict[str, List[AverageMeter]]
            freq = 1 if prefix == 'train' else self.cfg.eval_freq
            for lastname, meters in metrics.items():
                for meter in meters:
                    # Skip those meters never activated.
                    if len(meter.history) == 0:
                        continue
                    meter.plot(freq=freq)
                    imgname = meter.save(path=os.path.join(self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR), prefix=prefix)
                    epoch, val = meter.argbest(freq)
                    info += s.format(
                        prefix=prefix, name=meter.name,
                        val=val, epoch=epoch, img=f"![]({imgname})"
                    )
                    data.append([prefix, meter.name, val, epoch])
                    if val != -1: # Only save available data.
                        best[prefix][meter.name] = val

        file_ = os.path.join(self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR, self.cfg.SUMMARY_FILENAME)
        with open(file_, "w", encoding="utf8") as fh:
            fh.write(info)

        df = pd.DataFrame(data, columns=['Prefix', 'Metric', 'Best', '@Epoch'])
        infoLogger(str(df))
        infoLogger(f"[LoG_PaTH] >>> {self.cfg.LOG_PATH}")

        self.monitors.write(os.path.join(self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR)) # tensorboard
        self.monitors.save(os.path.join(self.cfg.LOG_PATH, self.cfg.DATA_DIR), self.cfg.MONITOR_FILENAME)

        return best
            
    @timemeter
    def fit(self):

        def signal_handler(sig, frame):
            infoLogger(f"\033[0;31;47m===============================TERMINATE CURRENT PROCESS===============================\033[0m")
            sys.exit()
        signal.signal(signal.SIGINT, signal_handler)

        start_epoch = self.resume()
        for epoch in range(start_epoch, self.cfg.epochs):
            self.seed_worker()
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


class GenCoach(Coach):

    def evaluate(self, epoch: int, prefix: str = 'valid'):
        r"""
        Evaluate recommender by sampled-based or full ranking
        according to the form of `data`:

        1. (users, pool):
            users: torch.Tensor, (B, 1)
            pool: torch.Tensor, (B, 101)
        2. (users, unseen, seen):
            users: BufferField
            unseen: BufferField
            seen: BufferField
        """
        model = self.get_res_sys_arch()
        userFeats, itemFeats = model.recommend()
        for data in self.dataloader:
            if len(data) == 2:
                users, pool = [col.to(self.device) for col in data]
                users = userFeats[users] # (B, 1, D)
                items = itemFeats[pool] # (B, K, D)
                scores = users.mul(items).sum(-1)
                targets = torch.zeros_like(scores)
                targets[:, 0].fill_(1)
            elif len(data) == 3:
                users, unseen, seen = data
                users = users.to(self.device).data
                seen = seen.to_csr().to(self.device).to_dense().bool()
                targets = unseen.to_csr().to(self.device).to_dense()
                users = userFeats[users].flatten(1) # B x D
                items = itemFeats.flatten(1) # N x D
                scores = users.matmul(items.T) # B x N
                scores[seen] = -1e23
            else:
                raise NotImplementedError(
                    f"GenCoach's `evaluate` expects the `data` to be the length of 2 or 3, but {len(data)} received ..."
                )

            scores = torch.cat(
                all_gather(scores), dim=0
            )
            targets = torch.cat(
                all_gather(targets), dim=0
            )

            self.monitor(
                scores, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['HITRATE', 'PRECISION', 'RECALL', 'NDCG', 'MRR']
            )


class SeqCoach(Coach):

    def evaluate(self, epoch: int, prefix: str = 'valid'):
        r"""
        Evaluate recommender by sampled-based or full ranking
        according to the form of `data`:

        1. (users, seqs, pool):
            users: torch.Tensor, (B, 1)
            seqs: torch.Tensor, (B, S)
            pool: torch.Tensor, (B, 101)
        2. (users, seqs, unseen, seen):
            users: BufferField
            seqs: BufferField
            unseen: BufferField
            seen: BufferField
        """
        model = self.get_res_sys_arch()
        for data in self.dataloader:
            if len(data) == 3:
                users, seqs, pool = [col.to(self.device) for col in data]
                scores = model.recommend(users=users, seqs=seqs, pool=pool)
                targets = torch.zeros_like(scores)
                targets[:, 0].fill_(1)
            elif len(data) == 4:
                users, seqs, unseen, seen = data
                users = users.to(self.device).data
                seqs = seqs.to(self.device).data
                scores = model.recommend(users=users, seqs=seqs)
                seen = seen.to_csr().to(self.device).to_dense().bool()
                scores[seen] = -1e23
                targets = unseen.to_csr().to(self.device).to_dense()
            else:
                raise NotImplementedError(
                    f"SeqCoach's `evaluate` expects the `data` to be the length of 3 or 4, but {len(data)} received ..."
                )

            scores = torch.cat(
                all_gather(scores), dim=0
            )
            targets = torch.cat(
                all_gather(targets), dim=0
            )

            self.monitor(
                scores, targets,
                n=len(users), mode="mean", prefix=prefix,
                pool=['HITRATE', 'PRECISION', 'RECALL', 'NDCG', 'MRR']
            )


class SessCoach(Coach):

    def evaluate(self, epoch: int, prefix: str = 'valid'):
        r"""
        Evaluate recommender by sampled-based or full ranking
        according to the form of `data`:

        1. (users, seqs, pool):
            users: torch.Tensor, (B, 1)
            seqs: torch.Tensor, (B, S)
            pool: torch.Tensor, (B, 101)
        2. (users, seqs, unseen, seen):
            users: BufferField
            seqs: BufferField
            unseen: BufferField
            seen: BufferField
        """
        model = self.get_res_sys_arch()
        for data in self.dataloader:
            if len(data) == 3:
                sesses, seqs, pool = [col.to(self.device) for col in data]
                scores = model.recommend(sesses=sesses, seqs=seqs, pool=pool)
                targets = torch.zeros_like(scores)
                targets[:, 0].fill_(1)
            elif len(data) == 4:
                sesses, seqs, unseen, seen = data
                sesses = sesses.data
                seqs = seqs.to(self.device).data
                scores = model.recommend(sesses=sesses, seqs=seqs)
                # Don't remove seens for session
                targets = unseen.to_csr().to(self.device).to_dense()
            else:
                raise NotImplementedError(
                    f"SessCoach's `evaluate` expects the `data` to be the length of 3 or 4, but {len(data)} received ..."
                )

            scores = torch.cat(
                all_gather(scores), dim=0
            )
            targets = torch.cat(
                all_gather(targets), dim=0
            )

            self.monitor(
                scores, targets,
                n=len(sesses), mode="mean", prefix=prefix,
                pool=['HITRATE', 'PRECISION', 'RECALL', 'NDCG', 'MRR']
            )


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
        """Load best.pickle from logPath of corresponding."""
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