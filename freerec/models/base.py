import math
from typing import Dict, Iterable, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from freerec.criterions import BaseCriterion
from freerec.data.datasets.base import RecDataSet
from freerec.data.fields import Field, FieldModule, FieldModuleList
from freerec.data.postprocessing import PostProcessor
from freerec.data.tags import (
    ID,
    ITEM,
    LABEL,
    NEGATIVE,
    POSITIVE,
    SEEN,
    SEQUENCE,
    SIZE,
    UNSEEN,
    USER,
)

__all__ = ["RecSysArch", "GenRecArch", "SeqRecArch", "PredRecArch"]


class RecSysArch(nn.Module):
    r"""A base PyTorch module for recommendation system architectures.

    This module provides common utilities for recommendation models,
    including tensor broadcasting, parameter initialization, and
    evaluation pipelines.

    Parameters
    ----------
    dataset : :class:`~RecDataSet`
        The recommendation dataset that provides field definitions.

    Attributes
    ----------
    criterion : :class:`~BaseCriterion`
        The loss criterion used for training.
    User : :class:`~FieldModule`
        The user ID field module.
    Item : :class:`~FieldModule`
        The item ID field module.
    Label : :class:`~FieldModule`
        The label field module.
    """

    criterion: BaseCriterion

    def __init__(self, dataset: RecDataSet) -> None:
        r"""Initialize RecSysArch with the given dataset."""
        super().__init__()

        self.dataset = dataset
        self.fields = dataset.fields

        self.User: FieldModule = self.fields[USER, ID]
        self.Item: FieldModule = self.fields[ITEM, ID]
        self.Label: FieldModule = self.fields[LABEL]
        self.Size: Field = Field(SIZE.name, SIZE)
        if self.Item:
            self.ISeq: FieldModule = self.Item.fork(SEQUENCE)
            self.IPos: FieldModule = self.Item.fork(POSITIVE)
            self.INeg: FieldModule = self.Item.fork(NEGATIVE)
            self.IUnseen: FieldModule = self.Item.fork(UNSEEN)
            self.ISeen: FieldModule = self.Item.fork(SEEN)

    @property
    def fields(self) -> FieldModuleList:
        r"""Return the list of field modules."""
        return self.__fields

    @fields.setter
    def fields(self, fields: Iterable[Union[Field, FieldModule]]):
        r"""Set and convert fields to a :class:`~FieldModuleList`."""
        self.__fields = []
        for field in fields:
            if isinstance(field, FieldModule):
                self.__fields.append(field)
            elif isinstance(field, Field):
                self.__fields.append(field.to_module())
            else:
                raise ValueError(f"{field} is not a Field ...")
        self.__fields = FieldModuleList(self.__fields)

    def to(
        self,
        device: Optional[Union[int, torch.device]] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        non_blocking: bool = False,
    ):
        r"""Move and/or cast the parameters and buffers.

        Parameters
        ----------
        device : int or :class:`torch.device`, optional
            The destination device of the parameters and buffers.
        dtype : :class:`torch.dtype` or str, optional
            The desired data type of the parameters and buffers.
        non_blocking : bool, optional
            Whether the copy should be asynchronous, by default ``False``.

        Returns
        -------
        :class:`~RecSysArch`
            The module with parameters and buffers moved and/or cast.
        """
        if device:
            self.device = device
        return super().to(device, dtype, non_blocking)

    def reset_parameters(self):
        r"""Reset all learnable parameters using Xavier/truncated-normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=math.sqrt(1.0 / m.weight.size(1)))
            elif isinstance(m, nn.GRU):
                nn.init.xavier_uniform_(m.weight_hh_l0)
                nn.init.xavier_uniform_(m.weight_ih_l0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def sure_trainpipe(self) -> PostProcessor:
        r"""Return the training data pipeline.

        Returns
        -------
        :class:`~PostProcessor`
            The configured training post-processor.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    def sure_validpipe(self) -> PostProcessor:
        r"""Return the validation data pipeline.

        Returns
        -------
        :class:`~PostProcessor`
            The configured validation post-processor.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    def sure_testpipe(self) -> PostProcessor:
        r"""Return the test data pipeline.

        Returns
        -------
        :class:`~PostProcessor`
            The configured test post-processor.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    def reset_ranking_buffers(self):
        r"""Reset ranking buffers before evaluation."""
        self.ranking_buffer = dict()

    def recommend_from_full(self, data: Dict[Field, torch.Tensor]) -> torch.Tensor:
        r"""Generate recommendations based on full ranking.

        Parameters
        ----------
        data : dict of :class:`~Field` to :class:`torch.Tensor`
            A mapping from fields to their tensor representations.

        Returns
        -------
        :class:`torch.Tensor`
            Predicted scores for all items.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    def recommend_from_pool(self, data: Dict[Field, torch.Tensor]) -> torch.Tensor:
        r"""Generate recommendations based on a candidate pool.

        Parameters
        ----------
        data : dict of :class:`~Field` to :class:`torch.Tensor`
            A mapping from fields to their tensor representations.

        Returns
        -------
        :class:`torch.Tensor`
            Predicted scores for candidate items.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    def encode(
        self, data: Dict[Field, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Encode user and item embeddings.

        Parameters
        ----------
        data : dict of :class:`~Field` to :class:`torch.Tensor`
            A mapping from fields to their tensor representations.

        Returns
        -------
        userEmbds : :class:`torch.Tensor`
            The encoded user embeddings.
        itemEmbds : :class:`torch.Tensor`
            The encoded item embeddings.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    def fit(
        self, data: Dict[Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        r"""Compute training losses from the given data.

        Parameters
        ----------
        data : dict of :class:`~Field` to :class:`torch.Tensor`
            A mapping from fields to their tensor representations.

        Returns
        -------
        :class:`torch.Tensor` or tuple of :class:`torch.Tensor`
            The computed loss value(s).

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method.
        """
        raise NotImplementedError

    def forward(
        self, data: Dict[Field, torch.Tensor], ranking: Literal["full", "pool"] = "full"
    ):
        r"""Run a forward pass for training or inference.

        Parameters
        ----------
        data : dict of :class:`~Field` to :class:`torch.Tensor`
            A mapping from fields to their tensor representations.
        ranking : ``'full'`` or ``'pool'``, optional
            The ranking strategy used during inference, by default ``'full'``.

        Returns
        -------
        :class:`torch.Tensor` or tuple of :class:`torch.Tensor`
            Training losses (if training) or predicted scores (if evaluating).
        """
        if self.training:
            return self.fit(data)
        else:
            if ranking == "full":
                return self.recommend_from_full(data)
            else:
                return self.recommend_from_pool(data)


class GenRecArch(RecSysArch):
    r"""Base architecture for general (non-sequential) recommendation models.

    This class provides default validation and test pipelines using
    ordered user ID sources.
    """

    def sure_validpipe(
        self, ranking: str = "full", batch_size: int = 512
    ) -> PostProcessor:
        r"""Return the validation data pipeline.

        Parameters
        ----------
        ranking : str, optional
            The ranking strategy, by default ``'full'``.
        batch_size : int, optional
            The batch size, by default ``512``.

        Returns
        -------
        :class:`~PostProcessor`
            The configured validation post-processor.
        """
        return (
            self.dataset.valid()
            .ordered_user_ids_source()
            .valid_sampling_(ranking)
            .batch_(batch_size)
            .tensor_()
        )

    def sure_testpipe(
        self, ranking: str = "full", batch_size: int = 512
    ) -> PostProcessor:
        r"""Return the test data pipeline.

        Parameters
        ----------
        ranking : str, optional
            The ranking strategy, by default ``'full'``.
        batch_size : int, optional
            The batch size, by default ``512``.

        Returns
        -------
        :class:`~PostProcessor`
            The configured test post-processor.
        """
        return (
            self.dataset.test()
            .ordered_user_ids_source()
            .test_sampling_(ranking)
            .batch_(batch_size)
            .tensor_()
        )


class SeqRecArch(RecSysArch):
    r"""Base architecture for sequential recommendation models.

    This class provides default validation and test pipelines with
    sequence pruning and padding.

    Attributes
    ----------
    NUM_PADS : int
        Number of padding tokens, by default ``1``.
    PADDING_VALUE : int
        Value used for padding, by default ``0``.
    """

    NUM_PADS: int = 1
    PADDING_VALUE: int = 0

    def sure_validpipe(
        self,
        maxlen: int,
        ranking: str = "full",
        batch_size: int = 512,
    ) -> PostProcessor:
        r"""Return the validation data pipeline with sequence handling.

        Parameters
        ----------
        maxlen : int
            Maximum sequence length.
        ranking : str, optional
            The ranking strategy, by default ``'full'``.
        batch_size : int, optional
            The batch size, by default ``512``.

        Returns
        -------
        :class:`~PostProcessor`
            The configured validation post-processor.
        """
        return (
            self.dataset.valid()
            .ordered_user_ids_source()
            .valid_sampling_(ranking)
            .lprune_(maxlen, modified_fields=(self.ISeq,))
            .add_(offset=self.NUM_PADS, modified_fields=(self.ISeq,))
            .lpad_(
                maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
            )
            .batch_(batch_size)
            .tensor_()
        )

    def sure_testpipe(
        self,
        maxlen: int,
        ranking: str = "full",
        batch_size: int = 512,
    ) -> PostProcessor:
        r"""Return the test data pipeline with sequence handling.

        Parameters
        ----------
        maxlen : int
            Maximum sequence length.
        ranking : str, optional
            The ranking strategy, by default ``'full'``.
        batch_size : int, optional
            The batch size, by default ``512``.

        Returns
        -------
        :class:`~PostProcessor`
            The configured test post-processor.
        """
        return (
            self.dataset.test()
            .ordered_user_ids_source()
            .test_sampling_(ranking)
            .lprune_(maxlen, modified_fields=(self.ISeq,))
            .add_(offset=self.NUM_PADS, modified_fields=(self.ISeq,))
            .lpad_(
                maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
            )
            .batch_(batch_size)
            .tensor_()
        )


class PredRecArch(RecSysArch):
    r"""Base architecture for prediction-based recommendation models.

    This class provides default validation and test pipelines using
    ordered interaction sources.
    """

    def sure_validpipe(
        self,
        batch_size: int = 4096,
    ) -> PostProcessor:
        r"""Return the validation data pipeline.

        Parameters
        ----------
        batch_size : int, optional
            The batch size, by default ``4096``.

        Returns
        -------
        :class:`~PostProcessor`
            The configured validation post-processor.
        """
        return self.dataset.valid().ordered_inter_source().batch_(batch_size).tensor_()

    def sure_testpipe(
        self,
        batch_size: int = 4096,
    ) -> PostProcessor:
        r"""Return the test data pipeline.

        Parameters
        ----------
        batch_size : int, optional
            The batch size, by default ``4096``.

        Returns
        -------
        :class:`~PostProcessor`
            The configured test post-processor.
        """
        return self.dataset.test().ordered_inter_source().batch_(batch_size).tensor_()
