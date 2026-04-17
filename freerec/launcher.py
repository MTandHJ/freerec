import abc
import os
import signal
import sys
import time
from collections import defaultdict
from functools import partial
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from .data.datasets import RecDataSet
from .data.fields import Field, FieldTuple
from .data.postprocessing import PostProcessor
from .data.tags import ID, ITEM, LABEL, SEEN, SIZE, UNSEEN, USER
from .ddp import is_distributed, is_main_process, main_process_only, synchronize
from .dict2obj import Config
from .metrics import *
from .models import RecSysArch
from .parser import TIME, Parser
from .utils import (
    AverageMeter,
    Monitor,
    export_pickle,
    import_pickle,
    infoLogger,
    timemeter,
)

__all__ = ["ChiefCoach", "Coach", "Adapter"]


DEFAULT_METRICS = {
    "LOSS": lambda x: x,
    #############
    "MSE": mean_squared_error,
    "MAE": mean_abs_error,
    "RMSE": root_mse,
    #############
    "PRECISION": precision,
    "RECALL": recall,
    "F1": f1_score,
    "HITRATE": hit_rate,
    #############
    "NDCG": normalized_dcg,
    "MRR": mean_reciprocal_rank,
    "MAP": mean_average_precision,
    #############
    "AUC": auroc,
    "GAUC": group_auroc,
    "LOGLOSS": log_loss,
}

DEFAULT_FMTS = {
    "LOSS": ".5f",
    #############
    "MSE": ".4f",
    "MAE": ".4f",
    "RMSE": ".4f",
    #############
    "PRECISION": ".4f",
    "RECALL": ".4f",
    "F1": ".4f",
    "HITRATE": ".4f",
    #############
    "NDCG": ".4f",
    "MRR": ".4f",
    "MAP": ".4f",
    #############
    "AUC": ".4f",
    "GAUC": ".4f",
    "LOGLOSS": ".5f",
}

DEFAULT_BEST_CASTER = {
    "LOSS": min,
    #############
    "MSE": min,
    "MAE": min,
    "RMSE": min,
    #############
    "PRECISION": max,
    "RECALL": max,
    "F1": max,
    "HITRATE": max,
    #############
    "NDCG": max,
    "MRR": max,
    "MAP": max,
    #############
    "AUC": max,
    "GAUC": max,
    "LOGLOSS": min,
}


class EarlyStopError(Exception):
    r"""Raised when early stopping criteria are met."""


class _DummyModule(torch.nn.Module):
    r"""Placeholder module used before a real model is assigned."""

    def forward(self, *args, **kwargs):
        r"""Raise :class:`NotImplementedError`."""
        raise NotImplementedError("No model available for Coach ...")

    def step(self, *args, **kwargs):
        r"""Raise :class:`NotImplementedError`."""
        raise NotImplementedError(
            "No optimizer or lr scheduler available for Coach ..."
        )

    def backward(self, *args, **kwargs):
        r"""Raise :class:`NotImplementedError`."""
        raise NotImplementedError("No optimizer available for Coach ...")


class ChiefCoach(metaclass=abc.ABCMeta):
    r"""Top-level class for running training and evaluation loops.

    Parameters
    ----------
    dataset : :class:`~freerec.data.datasets.RecDataSet`
        The original dataset.
    trainpipe : :class:`~freerec.data.postprocessing.PostProcessor`
        Data pipeline for training data.
    validpipe : :class:`~freerec.data.postprocessing.PostProcessor`
        Data pipeline for validation data.
    testpipe : :class:`~freerec.data.postprocessing.PostProcessor`, optional
        Data pipeline for testing data.
        If ``None``, ``validpipe`` is used instead.
    model : :class:`~freerec.models.RecSysArch`
        Model for training and evaluation.
    cfg : :class:`~freerec.parser.Parser`
        Runtime configuration.
    """

    def __init__(
        self,
        *,
        dataset: RecDataSet,
        trainpipe: PostProcessor,
        validpipe: PostProcessor,
        testpipe: Optional[PostProcessor],
        model: RecSysArch,
        cfg: Parser,
    ):
        r"""Initialize ChiefCoach."""

        self.cfg = cfg
        self.__mode = "train"

        self.set_device(self.cfg.device)
        self.set_dataset(dataset)
        self.set_datapipe(trainpipe, validpipe, testpipe)
        self.set_dataloader()

        self.set_model(model)
        self.set_optimizer()
        self.set_lr_scheduler()
        self.set_monitors(self.cfg.monitors)

        # Other setup can be placed here
        self.set_other()

    def set_device(self, device):
        r"""Set the computation device."""
        self.device = torch.device(device)

    def set_dataset(self, dataset: RecDataSet):
        r"""Set the dataset and extract common field references.

        Parameters
        ----------
        dataset : :class:`~freerec.data.datasets.RecDataSet`
            The recommendation dataset.
        """
        self.dataset = dataset
        self.fields: FieldTuple[Field] = FieldTuple(dataset.fields)
        self.User = self.fields[USER, ID]
        self.Item = self.fields[ITEM, ID]
        self.Label = self.fields[LABEL]
        self.Size = Field(SIZE.name, SIZE)
        if self.Item is not None:
            self.ISeen = self.Item.fork(SEEN)
            self.IUnseen = self.Item.fork(UNSEEN)

    def set_datapipe(
        self,
        trainpipe,
        validpipe,
        testpipe=None,
    ):
        r"""Set the train, validation, and test data pipelines."""
        self.trainpipe = trainpipe
        self.validpipe = validpipe
        self.testpipe = self.validpipe if testpipe is None else testpipe

    def set_model(self, model: RecSysArch):
        r"""Move the model to the target device and wrap with DDP if distributed."""
        self.model = model.to(self.device)
        if is_distributed():
            self.model = torch.nn.parallel.DistributedDataParallel(model)

    def set_optimizer(self):
        r"""Create the optimizer based on ``cfg.optimizer``.

        Raises
        ------
        NotImplementedError
            If the optimizer name is not one of ``'sgd'``, ``'adam'``,
            or ``'adamw'``.
        """
        if self.cfg.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise NotImplementedError(f"Unexpected optimizer {self.cfg.optimizer} ...")

    def set_lr_scheduler(self):
        r"""Set the learning rate scheduler. Override to use a real scheduler."""
        self.lr_scheduler = _DummyModule()

    def set_dataloader(self) -> None:
        r"""Create :class:`torchdata.dataloader2.DataLoader2` instances for train, valid, and test."""
        from torchdata.dataloader2 import (
            DataLoader2,
            DistributedReadingService,
            MultiProcessingReadingService,
            SequentialReadingService,
        )

        def get_reading_service():
            if is_distributed():
                rs = SequentialReadingService(
                    DistributedReadingService(),
                    MultiProcessingReadingService(self.cfg.num_workers),
                )
            else:
                rs = MultiProcessingReadingService(self.cfg.num_workers)
            return rs

        self.trainloader = DataLoader2(
            datapipe=self.trainpipe, reading_service=get_reading_service()
        )
        self.validloader = DataLoader2(
            datapipe=self.validpipe, reading_service=get_reading_service()
        )
        self.testloader = DataLoader2(
            datapipe=self.testpipe, reading_service=get_reading_service()
        )

    def set_other(self):
        r"""Hook for additional setup. Override in subclasses."""
        ...

    def get_res_sys_arch(self) -> RecSysArch:
        r"""Unwrap the underlying :class:`~freerec.models.RecSysArch` from DDP or DataParallel.

        Returns
        -------
        :class:`~freerec.models.RecSysArch`
            The unwrapped model.
        """
        model = self.model
        if isinstance(self.model, torch.nn.parallel.DataParallel):
            model = self.model.module
        elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        assert isinstance(model, RecSysArch), "No RecSysArch found ..."
        return model

    @property
    def fields(self) -> FieldTuple[Field]:
        r"""The dataset fields."""
        return self.__fields

    @fields.setter
    def fields(self, fields):
        r"""Set the dataset fields."""
        self.__fields = FieldTuple(fields)

    @property
    def mode(self):
        r"""The current mode (``'train'``, ``'valid'``, or ``'test'``)."""
        return self.__mode

    @mode.setter
    def mode(self, mode: str):
        r"""Set the current mode and toggle model train/eval accordingly."""
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()
        self.__mode = mode

    @property
    def dataloader(self):
        r"""The dataloader for the current mode."""
        if self.mode == "train":
            return self.trainloader
        elif self.mode == "valid":
            return self.validloader
        else:
            return self.testloader

    def shutdown(self):
        r"""Shutdown all dataloaders."""
        self.trainloader.shutdown()
        self.validloader.shutdown()
        self.testloader.shutdown()

    @abc.abstractmethod
    def train_per_epoch(self, epoch: int):
        r"""Run one training epoch. Must be implemented by subclasses.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.train_per_epoch() should be implemented ..."
        )

    @abc.abstractmethod
    def evaluate(self, epoch: int, step: int = -1, mode: str = "valid"):
        r"""Run evaluation. Must be implemented by subclasses.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        step : int, optional
            The step number within the epoch, by default -1.
        mode : str, optional
            One of ``'valid'`` or ``'test'``, by default ``'valid'``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.evaluate() should be implemented ..."
        )

    def register_metric(
        self,
        name: str,
        func: Callable,
        fmt: str = ".4f",
        best_caster: Callable = max,
        **kwargs,
    ) -> None:
        r"""Register a custom metric.

        Parameters
        ----------
        name : str
            The full metric name, e.g. ``'NDCG@10'``.
            The ``@`` separator indicates ``metric@k``; in this case
            ``func`` must accept a ``k`` argument.
        func : callable
            The metric function.
        fmt : str, optional
            The display format string, by default ``'.4f'``.
        best_caster : callable, optional
            Either ``max`` or ``min``, indicating whether higher or lower
            values are better, by default ``max``.
        **kwargs
            Additional keyword arguments forwarded to ``func``.
        """
        name = name.upper()
        if "@" in name:
            lastname, K = name.split("@")
            kwargs["k"] = int(K)
        else:
            lastname = name

        DEFAULT_METRICS[lastname] = func
        DEFAULT_FMTS[lastname] = fmt
        DEFAULT_BEST_CASTER[lastname] = best_caster

        for mode in ("train", "valid", "test"):
            self._set_monitor(name=name, lastname=lastname, mode=mode, **kwargs)

    def _set_monitor(self, name: str, lastname: str, mode: str = "train", **kwargs):
        r"""Create and register an :class:`~freerec.utils.AverageMeter` for a metric.

        Parameters
        ----------
        name : str
            The full metric name (e.g. ``'NDCG@10'``).
        lastname : str
            The base metric name without ``@k`` suffix.
        mode : str, optional
            One of ``'train'``, ``'valid'``, or ``'test'``, by default ``'train'``.
        **kwargs
            Additional keyword arguments forwarded to the metric function.

        Returns
        -------
        :class:`~freerec.utils.AverageMeter`
            The created meter.

        Raises
        ------
        KeyError
            If ``lastname`` is not found in ``DEFAULT_METRICS``.
        """
        name, lastname = name.upper(), lastname.upper()
        try:
            meter = AverageMeter(
                name=name,
                metric=partial(DEFAULT_METRICS[lastname], **kwargs),
                fmt=DEFAULT_FMTS[lastname],
                best_caster=DEFAULT_BEST_CASTER[lastname],
            )
            self.__monitors[mode][lastname].append(meter)

            if mode == "valid" and name == self.cfg.which4best.upper():
                self.meter4best = meter
                self._best = -float("inf") if meter.caster is max else float("inf")
                self._best_epoch: int = 0
                self._best_step: int = -1
                self._stopping_steps: int = 0
                self._early_stop_patience: int = self.cfg.early_stop_patience

        except KeyError:
            raise KeyError(
                f"The metric of {lastname} is not included. "
                f"You can register by calling `register_metric(...)' ..."
            )
        return meter

    @property
    def monitors(self) -> Monitor:
        r"""The :class:`~freerec.utils.Monitor` dictionary keyed by mode."""
        return self.__monitors

    def set_monitors(self, monitors: List[str]):
        r"""Initialize monitors for all modes.

        Parameters
        ----------
        monitors : list of str
            Metric names to monitor (e.g. ``['RECALL@10', 'NDCG@20']``).
            ``'LOSS'`` is always included automatically.

        Examples
        --------
        >>> coach.set_monitors(monitors=['recall@10', 'recall@20', 'ndcg@10', 'ndcg@20'])
        """
        # meters for train|valid|test
        self.__monitors = Monitor()
        self.__monitors["train"] = defaultdict(list)
        self.__monitors["valid"] = defaultdict(list)
        self.__monitors["test"] = defaultdict(list)

        # UPPER
        monitors = ["LOSS"] + [name.upper() for name in monitors]
        monitors = sorted(set(monitors), key=monitors.index)

        for name in monitors:
            for mode in ("train", "valid", "test"):
                if "@" in name:
                    lastname, K = name.split("@")
                    self._set_monitor(name=name, lastname=lastname, mode=mode, k=int(K))
                else:
                    lastname = name
                    self._set_monitor(name=name, lastname=lastname, mode=mode)


class Coach(ChiefCoach):
    r"""Concrete training coach with checkpointing, early stopping, and evaluation.

    Inherits from :class:`ChiefCoach` and implements the full training loop
    including model saving/loading, checkpoint management, early stopping,
    and summary generation.
    """

    @property
    def meter4best(self):
        r"""The :class:`~freerec.utils.AverageMeter` used for best-model selection."""
        return self.__best_meter

    @meter4best.setter
    def meter4best(self, meter: AverageMeter):
        r"""Set the meter used for best-model selection."""
        self.__best_meter = meter
        infoLogger(f"[Coach] >>> Set best meter: {meter.name} ")

    @property
    def remove_seen(self):
        r"""Whether to remove seen items during full-ranking evaluation."""
        return not (self.cfg.retain_seen or self.dataset.has_duplicates())

    def save(self, filename: Optional[str] = None) -> None:
        r"""Save the model state dict to ``LOG_PATH``.

        Parameters
        ----------
        filename : str, optional
            Target filename. If ``None``, uses ``cfg.SAVED_FILENAME``.
        """
        if is_main_process():
            filename = self.cfg.SAVED_FILENAME if filename is None else filename
            torch.save(
                self.model.state_dict(), os.path.join(self.cfg.LOG_PATH, filename)
            )

        synchronize()
        return

    def load(self, path: str, filename: Optional[str] = None) -> None:
        r"""Load model state dict from disk.

        Parameters
        ----------
        path : str
            Directory containing the model file.
        filename : str, optional
            Model filename. If ``None``, uses ``cfg.SAVED_FILENAME``.
        """
        filename = self.cfg.SAVED_FILENAME if filename is None else filename
        self.model.load_state_dict(
            torch.load(
                os.path.join(path, filename),
                map_location=self.device,
                weights_only=True,
            )
        )

        synchronize()
        return

    def save_checkpoint(self, epoch: int) -> None:
        r"""Save a training checkpoint at the given epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        path = os.path.join(self.cfg.CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = dict()
        checkpoint["epoch"] = epoch
        for module in self.cfg.CHECKPOINT_MODULES:
            checkpoint[module] = getattr(self, module).state_dict()
        checkpoint["monitors"] = self.monitors.state_dict()
        torch.save(checkpoint, path)

        synchronize()
        return

    def load_checkpoint(self) -> int:
        r"""Load the last saved checkpoint.

        Returns
        -------
        int
            The epoch number stored in the checkpoint.
        """
        path = os.path.join(self.cfg.CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = torch.load(path, weights_only=False)
        for module in self.cfg.CHECKPOINT_MODULES:
            getattr(self, module).load_state_dict(checkpoint[module])
        self.monitors.load_state_dict(checkpoint["monitors"])

        synchronize()
        return checkpoint["epoch"]

    def resume(self) -> int:
        r"""Resume training from the last checkpoint if ``cfg.resume`` is set.

        Returns
        -------
        int
            The epoch number to resume from (0 if not resuming).
        """
        start_epoch: int = 0
        if self.cfg.resume:
            start_epoch = self.load_checkpoint()
            infoLogger(
                f"[Coach] >>> Load last checkpoint and train from epoch: {start_epoch}"
            )
        return start_epoch

    def save_best(self) -> None:
        r"""Save the current model as the best model."""
        self.save(self.cfg.BEST_FILENAME)

    def load_best(self) -> None:
        r"""Load the best saved model."""
        infoLogger(
            f"[Coach] >>> Load best model @Epoch: {self._best_epoch} ({self._best_step}) "
        )
        self.load(self.cfg.LOG_PATH, self.cfg.BEST_FILENAME)

    def check_best(self, epoch: int, step: int = -1) -> None:
        r"""Check if the current metric is the best and update accordingly.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        step : int, optional
            The current step number, by default -1.

        Raises
        ------
        :class:`EarlyStopError`
            If the early stopping patience is exceeded.
        """
        if self.meter4best.active:
            best_ = self.meter4best.which_is_better(self._best)
            if best_ != self._best:
                self._best = best_
                self._best_epoch = epoch
                self._best_step = step
                self._stopping_steps = 0
                infoLogger(
                    f"[Coach] >>> Better ***{self.meter4best.name}*** of ***{self._best:.4f}*** "
                )
                self.save_best()
            else:
                if self._stopping_steps >= self._early_stop_patience:
                    self.step(epoch, step)
                    raise EarlyStopError
                else:
                    self._stopping_steps += 1

    def eval_at_best(self):
        r"""Load the best model and run validation and test evaluation."""
        try:
            self.load_best()
            self.valid(self._best_epoch, self._best_step)
            self.test(self._best_epoch, self._best_step)
            self.load(self.cfg.LOG_PATH, self.cfg.SAVED_FILENAME)
        except FileNotFoundError:
            infoLogger("[Coach] >>> No best model was recorded. Skip it ...")

    @main_process_only
    def easy_record_best(self, best: defaultdict):
        r"""Record the best test-set results to a pickle file for TensorBoard.

        Parameters
        ----------
        best : :class:`~collections.defaultdict`
            Dictionary collecting the best metric values by mode.
        """

        for lastname, meters in self.monitors["test"].items():
            for meter in meters:
                # Skip those meters never activated.
                if len(meter.history) == 0:
                    continue
                # Note that meter.history[-1] is the result at the best checkpoint.
                val = meter.history[-1]
                best["best"][meter.name] = val

        export_pickle(
            best,
            os.path.join(
                self.cfg.LOG_PATH, self.cfg.DATA_DIR, self.cfg.MONITOR_BEST_FILENAME
            ),
        )

    @torch.no_grad()
    def monitor(
        self,
        *values,
        n: int = 1,
        reduction: str = "mean",
        mode: Literal["train", "valid", "test"] = "train",
        pool: Optional[Iterable] = None,
        refresh: bool = False,
    ):
        r"""Feed data values into the metric monitors.

        Parameters
        ----------
        *values
            Data values passed to each metric function.
        n : int, optional
            The batch size, by default 1.
        reduction : str, optional
            Reduction method: ``'sum'`` or ``'mean'``, by default ``'mean'``.
        mode : str, optional
            One of ``'train'``, ``'valid'``, or ``'test'``, by default ``'train'``.
        pool : iterable of str, optional
            Metric names to update. If ``None``, all metrics in ``mode`` are updated.
        refresh : bool, optional
            If ``True``, immediately step the meter (results will not be
            printed), by default ``False``.
        """

        metrics: Dict[List] = self.monitors[mode]
        pool = metrics if pool is None else pool
        for lastname in pool:
            for meter in metrics.get(lastname.upper(), []):
                meter(*values, n=n, reduction=reduction)
                if refresh:
                    meter.step()

    def step(self, epoch: int, step: int = -1):
        r"""Print metrics and reset all active meters.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        step : int, optional
            The step number within the epoch, by default -1.
        """
        metrics: Dict[str, List[AverageMeter]]
        for mode, metrics in self.monitors.items():
            if step == -1:
                infos = [f"[Coach] >>> {mode.upper():5} @Epoch: {epoch} >>> "]
            else:
                infos = [f"[Coach] >>> {mode.upper():5} @Epoch: {epoch} ({step}) >>> "]
            for meters in metrics.values():
                infos += [meter.step() for meter in meters if meter.active]
            if len(infos) == 1:
                continue
            infoLogger(" || ".join(infos))

    @main_process_only
    @timemeter
    def summary(self):
        r"""Generate a summary of the entire training process.

        Produces a Markdown summary table, metric curve plots, and saves
        the best results to a pickle file. All outputs are written under
        ``cfg.LOG_PATH/SUMMARY_DIR``.

        Returns
        -------
        :class:`~collections.defaultdict`
            Nested dict mapping ``mode -> metric_name -> best_value``.
        """
        import pandas as pd

        s = "|  {mode}  |   {name}   |   {val}   |   {step}   |   {img}   |\n"
        info = ""
        info += "|  Mode  |   Metric   |   Best   |   @Step   |   Img   |\n"
        info += "| :-------: | :-------: | :-------: | :-------: | :-------: |\n"
        data = []
        best = defaultdict(dict)

        for mode, metrics in self.monitors.items():
            metrics: defaultdict[str, List[AverageMeter]]
            freq = 1 if mode == "train" else self.cfg.eval_freq
            for lastname, meters in metrics.items():
                for meter in meters:
                    # Skip those meters never activated.
                    if len(meter.history) == 0:
                        continue
                    meter.plot(freq=freq)
                    imgname = meter.save(
                        path=os.path.join(self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR),
                        mode=mode,
                    )
                    step, val = meter.argbest(freq)
                    info += s.format(
                        mode=mode,
                        name=meter.name,
                        val=val,
                        step=step,
                        img=f"![]({imgname})",
                    )
                    data.append([mode, meter.name, val, step])
                    if val != -1:  # Only save available data.
                        best[mode][meter.name] = val

        file_ = os.path.join(
            self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR, self.cfg.SUMMARY_FILENAME
        )
        with open(file_, "w", encoding="utf8") as fh:
            fh.write(info)

        df = pd.DataFrame(data, columns=["Mode", "Metric", "Best", "@Step"])
        infoLogger(str(df))
        infoLogger(f"[LoG_PaTH] >>> {self.cfg.LOG_PATH}")

        self.monitors.write(
            os.path.join(self.cfg.LOG_PATH, self.cfg.SUMMARY_DIR)
        )  # tensorboard
        self.monitors.save(
            os.path.join(self.cfg.LOG_PATH, self.cfg.DATA_DIR),
            self.cfg.MONITOR_FILENAME,
        )

        return best

    def dict_to_device(self, data: Dict[Field, Any]) -> Dict[Field, Any]:
        r"""Move all :class:`torch.Tensor` values in a dict to the target device."""
        return {
            field: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for field, value in data.items()
        }

    def evaluate(self, epoch: int, step: int = -1, mode: str = "valid"):
        r"""Run evaluation over the current dataloader and update monitors.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        step : int, optional
            The step number, by default -1.
        mode : str, optional
            One of ``'valid'`` or ``'test'``, by default ``'valid'``.
        """
        self.get_res_sys_arch().reset_ranking_buffers()
        y_pred = []
        y_true = []
        groups = []
        for data in self.dataloader:
            bsz = data[self.Size]
            if self.User in data:
                users = data[self.User].flatten().tolist()
            else:  # No User field
                users = [0] * bsz

            data = self.dict_to_device(data)
            if self.cfg.ranking == "full":
                scores = self.model(data, ranking="full")
                if self.remove_seen:
                    seen = (
                        self.Item.to_csr(data[self.ISeen])
                        .to(self.device)
                        .to_dense()
                        .bool()
                    )
                    scores[seen] = -1e23
                targets = (
                    self.Item.to_csr(data[self.IUnseen]).to(self.device).to_dense()
                )
            elif self.cfg.ranking == "pool":
                scores = self.model(data, ranking="pool")
                if self.Label in data:
                    targets = data[self.Label]
                else:
                    targets = torch.zeros_like(scores)
                    targets[:, 0].fill_(1)
            else:
                raise NotImplementedError(
                    f"`ranking` should be 'full' or 'pool' but {self.cfg.ranking} received ..."
                )

            if self.Label in data:
                groups.extend(users)
                y_pred.extend(scores.flatten().tolist())
                y_true.extend(targets.flatten().tolist())

            self.monitor(
                scores,
                targets,
                n=bsz,
                reduction="mean",
                mode=mode,
                pool=["HITRATE", "PRECISION", "RECALL", "NDCG", "MRR"],
            )

        # TODO: Multi GPUs Support
        self.monitor(
            y_pred, y_true, n=1, reduction="mean", mode=mode, pool=["LOGLOSS", "AUC"]
        )
        self.monitor(
            y_pred, y_true, groups, n=1, reduction="mean", mode=mode, pool=["GAUC"]
        )

    @timemeter
    def train(self, epoch: int):
        r"""Run one training epoch and print metrics.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        self.mode = "train"
        # self.dataloader.seed(epoch)
        self.train_per_epoch(epoch)
        self.step(epoch)

    @timemeter
    @torch.no_grad()
    def valid(self, epoch: int, step: int = -1):
        r"""Run validation, check for best model, and print metrics."""
        mode_ = self.mode

        self.mode = "valid"
        self.evaluate(epoch=epoch, step=step, mode="valid")
        self.check_best(epoch, step)
        self.step(epoch, step)

        self.mode = mode_

    @timemeter
    @torch.no_grad()
    def test(self, epoch: int, step: int = -1):
        r"""Run test evaluation and print metrics."""
        mode_ = self.mode

        self.mode = "test"
        self.evaluate(epoch=epoch, step=step, mode="test")
        self.step(epoch, step)

        self.mode = mode_

    @timemeter
    def fit(self):
        r"""Run the full training loop with evaluation, early stopping, and summary."""
        start_epoch = self.resume()
        epoch = 0
        try:
            for epoch in range(start_epoch, self.cfg.epochs):
                if epoch % self.cfg.CHECKPOINT_FREQ == 0:
                    self.save_checkpoint(epoch)
                if epoch % self.cfg.eval_freq == 0:
                    if self.cfg.eval_valid:
                        self.valid(epoch)
                    if self.cfg.eval_test:
                        self.test(epoch)
                self.train(epoch + 1)

            self.save()

            # last epoch
            self.valid(self.cfg.epochs)
            self.test(self.cfg.epochs)
        except EarlyStopError:
            infoLogger(f"[Coach] >>> Early Stop @Epoch: {epoch}")
            self.save()
        self._stopping_steps = -1
        best = self.summary()
        self.eval_at_best()
        self.easy_record_best(best)
        self.shutdown()


class Adapter:
    r"""Hyperparameter grid-search tuner.

    Manages multi-device subprocess-based grid search over hyperparameters,
    writing results to TensorBoard for comparison.

    Examples
    --------
    >>> tuner = Adapter()
    >>> tuner.compile(cfg)
    >>> tuner.fit()
    """

    def __init__(self) -> None:
        r"""Initialize Adapter."""
        self.params = []
        self.values = []
        self.devices = tuple()

    @property
    def COMMAND(self):
        r"""The base shell command template."""
        return self.cfg.COMMAND

    def register(self, device: str) -> Tuple[str, str]:
        r"""Register a new experiment run on the given device.

        Parameters
        ----------
        device : str
            The device identifier for this run.

        Returns
        -------
        tuple of (str, str, str)
            ``(command, id, log_path)`` for the registered run.
        """
        self.cfg.ENVS["id"] = time.strftime(TIME)
        self.cfg.ENVS["device"] = device
        command = self.COMMAND + self.get_option("id", self.cfg.ENVS.id)
        command += self.get_option("device", self.cfg.ENVS.device)
        return command, self.cfg.ENVS.id, self.cfg.LOG_PATH.format(**self.cfg.ENVS)

    @timemeter
    def compile(self, cfg: Config) -> None:
        r"""Configure the command, environments, and parameter grid.

        Parameters
        ----------
        cfg : :class:`~freerec.dict2obj.Config`
            Configuration containing ``COMMAND``, ``ENVS``, ``PARAMS``,
            and ``DEFAULTS``.
        """
        self.cfg = cfg
        piece = "\t{key}: {vals} \n"
        envs, params, defaults = "", "", ""
        for key, val in self.cfg.ENVS.items():
            if key == "device":
                self.devices = tuple(val.split(","))
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
        r"""Register a parameter and its candidate values for grid search."""
        self.params.append(key)
        self.values.append(vals)

    @staticmethod
    def get_option(key: str, val: Any):
        r"""Convert a key-value pair to a CLI option string.

        Parameters
        ----------
        key : str
            The parameter name. Underscores are replaced by hyphens.
        val
            The parameter value.

        Returns
        -------
        str
            A string in the form ``' --key=val'``.

        Examples
        --------
        >>> Adapter.get_option('lr', '1e-3')
        ' --lr=1e-3'
        >>> Adapter.get_option('learning_rate', '1e-3')
        ' --learning-rate=1e-3'
        """
        return f" --{key.replace('_', '-')}={val}"

    def load_best(self, logPath: str):
        r"""Load the best results pickle from a subprocess log directory."""
        file_ = os.path.join(logPath, self.cfg.DATA_DIR, self.cfg.MONITOR_BEST_FILENAME)
        return import_pickle(file_)

    def write(self, id_: str, logPath: str, params: Dict):
        r"""Write experiment results to TensorBoard.

        Parameters
        ----------
        id_ : str
            Experiment ID.
        logPath : str
            Path to the experiment logs.
        params : dict
            Hyperparameter configuration of the experiment.

        Notes
        -----
        A value of ``-1`` in TensorBoard indicates a non-numeric metric
        value that could not be recorded.
        """
        try:
            data = self.load_best(logPath)
            path = os.path.join(self.cfg.CORE_LOG_PATH, id_)
            with SummaryWriter(log_dir=path) as writer:
                metrics = dict()
                for mode, best in data.items():
                    for metric, val in best.items():
                        val = val if isinstance(val, (int, float)) else -1
                        metrics["/".join([mode, metric])] = val
                writer.add_hparams(
                    params,
                    metrics,
                )
        except Exception:
            infoLogger(
                "\033[0;31;47m[Adapter] >>> Unknown errors happen. This is mainly due to abnormal exits of child processes.\033[0m"
            )

    def each_grid(self):
        r"""Yield parameter dicts for exclusive (one-at-a-time) grid search."""
        for key, vals in zip(self.params, self.values):
            for val in vals:
                yield self.cfg.DEFAULTS | {key: val}

    def product_grid(self):
        r"""Yield parameter dicts for full Cartesian-product grid search."""
        for vals in product(*self.values):
            yield self.cfg.DEFAULTS | {
                option: val for option, val in zip(self.params, vals)
            }

    def save_checkpoint(self, source: List) -> None:
        r"""Save remaining parameter combinations to a checkpoint."""
        path = os.path.join(self.cfg.CORE_CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = dict()
        checkpoint["source"] = source
        torch.save(checkpoint, path)

    @timemeter
    def load_checkpoint(self) -> int:
        r"""Load remaining parameter combinations from a checkpoint."""
        infoLogger("[Coach] >>> Load the recent checkpoint ...")
        path = os.path.join(self.cfg.CORE_CHECKPOINT_PATH, self.cfg.CHECKPOINT_FILENAME)
        checkpoint = torch.load(path, weights_only=True)
        return checkpoint["source"]

    def resume(self):
        r"""Resume grid search from the last checkpoint or start fresh."""
        source = self.each_grid() if self.cfg.EXCLUSIVE else self.product_grid()
        source = list(source)[::-1]
        source = self.load_checkpoint() if self.cfg.resume else source
        return source

    def run(self, command: str, params: Dict):
        r"""Launch a training subprocess with the given parameters."""
        import shlex
        import subprocess

        for option, val in params.items():
            command += self.get_option(option, val)
        infoLogger(f"\033[0;31;47m{command}\033[0m")
        return subprocess.Popen(shlex.split(command))

    def wait(self, tasks: List):
        r"""Block until all running subprocesses terminate."""
        tasks = [task for task in tasks if task is not None]
        for process_, id_, logPath, params in tasks:
            process_.wait()
            self.write(id_, logPath, params)

    def poll(self, tasks: List):
        r"""Poll until any subprocess terminates, then return its slot index."""

        def is_null(task):
            return task is None

        buffer_source = [task[-1] for task in tasks if task is not None]
        time.sleep(1)  # for unique id
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
        r"""Terminate all running subprocesses and exit."""
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
        r"""Run the full grid search loop across all devices."""
        self.source = self.resume()
        tasks = [None for _ in range(len(self.devices))]

        def signal_handler(sig, frame):
            infoLogger(
                "\033[0;31;47m===============================TERMINATE ALL SUBPROCESSES===============================\033[0m"
            )
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
