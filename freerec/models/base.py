

from typing import Literal, Optional, Union, Iterable, Tuple, Dict

import torch
import torch.nn as nn

from ..data.datasets.base import RecDataSet
from ..data.postprocessing import PostProcessor
from ..data.fields import Field, FieldModule, FieldModuleList
from ..data.tags import USER, ITEM, ID, SEQUENCE, UNSEEN, SEEN, POSITIVE, NEGATIVE
from ..criterions import BaseCriterion

__all__ = ['RecSysArch', 'GenRecArch', 'SeqRecArch']


class RecSysArch(nn.Module):
    r"""
    A PyTorch Module for recommendation system architecture.
    This module contains methods for broadcasting tensors and initializing the module parameters.
    """

    criterion: BaseCriterion

    def __init__(self, dataset: RecDataSet) -> None:
        super().__init__()

        self.dataset = dataset
        self.fields = dataset.fields

        self.User: FieldModule = self.fields[USER, ID]
        self.Item: FieldModule = self.fields[ITEM, ID]
        self.ISeq: FieldModule = self.Item.fork(SEQUENCE)
        self.IPos: FieldModule = self.Item.fork(POSITIVE)
        self.INeg: FieldModule = self.Item.fork(NEGATIVE)
        self.IUnseen: FieldModule = self.Item.fork(UNSEEN)
        self.ISeen: FieldModule = self.Item.fork(SEEN)
    
    @property
    def fields(self):
        return self.__fields
    
    @fields.setter
    def fields(self, fields: Iterable[Union[Field, FieldModule]]) -> FieldModuleList:
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
        self, device: Optional[Union[int, torch.device]] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        non_blocking: bool = False
    ):
        r"""
        Moves and/or casts the parameters and buffers.

        Parameters:
        -----------
        device : Union[int, torch.device], optional
            The destination device of the parameters and buffers, by default None.
        dtype : Union[torch.dtype, str], optional
            The desired data type of the parameters and buffers, by default None.
        non_blocking : bool, optional
            Whether the copy should be asynchronous or not, by default False.

        Returns:
        --------
        nn.Module
            The module with parameters and buffers moved and/or cast.
        """
        if device:
            self.device = device
        return super().to(device, dtype, non_blocking)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.GRU):
                nn.init.xavier_uniform_(m.weight_hh_l0)
                nn.init.xavier_uniform_(m.weight_ih_l0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def sure_trainpipe(self) -> PostProcessor:
        raise NotImplementedError

    def sure_validpipe(self) -> PostProcessor:
        raise NotImplementedError

    def sure_testpipe(self) -> PostProcessor:
        raise NotImplementedError
    
    def reset_ranking_buffers(self):
        """This method will be runed before evaluation."""
        self.ranking_buffer = dict()

    def recommend_from_full(self, data: Dict[Field, torch.Tensor]) -> torch.Tensor:
        r"""
        Recommendation based on full ranking.

        Returns:
        --------
        scores: torch.Tensor
        """
        raise NotImplementedError

    def recommend_from_pool(self, data: Dict[Field, torch.Tensor]) -> torch.Tensor:
        r"""
        Recommendation based on full ranking.

        Returns:
        --------
        scores: torch.Tensor
        """
        raise NotImplementedError

    def encode(self, data: Dict[Field, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Encoding user/item embeddings.

        Returns:
        --------
        userEmbds: torch.Tensor
        itemEmbds: torch.Tensor
        """
        raise NotImplementedError

    def fit(self, data: Dict[Field, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        r"""
        Fit data.

        Returns:
        --------
        losses: torch.Tensor, Tuple[torch.Tensor]
        """
        raise NotImplementedError

    def forward(self, data: Dict[Field, torch.Tensor], ranking: Literal['full', 'pool'] = 'full'):
        if self.training:
            return self.fit(data)
        else:
            if ranking == 'full':
                return self.recommend_from_full(data)
            else:
                return self.recommend_from_pool(data)


class GenRecArch(RecSysArch):

    def sure_validpipe(
        self, ranking: str = 'full', batch_size: int = 256
    ):
        return self.dataset.valid().ordered_user_ids_source(
        ).sharding_filter().valid_sampling_(
            ranking
        ).batch_(batch_size).tensor_()

    def sure_testpipe(
        self, ranking: str = 'full', batch_size: int = 256
    ):
        return self.dataset.test().ordered_user_ids_source(
        ).sharding_filter().test_sampling_(
            ranking
        ).batch_(batch_size).tensor_()
        

class SeqRecArch(RecSysArch):

    NUM_PADS: int = 1
    PADDING_VALUE: int = 0

    def sure_validpipe(
        self, maxlen: int, ranking: str = 'full', batch_size: int = 256,
    ):
        return self.dataset.valid().ordered_user_ids_source(
        ).sharding_filter().valid_sampling_(ranking).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def sure_testpipe(
        self, maxlen: int, ranking: str = 'full', batch_size: int = 256,
    ):
        return self.dataset.test().ordered_user_ids_source(
        ).sharding_filter().test_sampling_(ranking).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()