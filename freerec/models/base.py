

from typing import Optional, Union, overload, Tuple, Dict

import torch, inspect
import torch.nn as nn
import torchdata.datapipes as dp

from ..data.datasets.base import RecDataSet
from ..data.tags import USER, ITEM, ID, SEQUENCE, UNSEEN, SEEN

__all__ = ['RecSysArch']


class RecSysArch(nn.Module):
    r"""
    A PyTorch Module for recommendation system architecture.
    This module contains methods for broadcasting tensors and initializing the module parameters.
    """

    def __init__(self) -> None:
        super().__init__()
        self._check()

    @classmethod
    def _check(cls):
        signature = inspect.signature(cls.recommend_from_pool)
        if 'pool' not in signature.parameters:
            raise NotImplementedError(
                f"`pool` must be a argument of `recommend_from_pool` ..."
            )
        signature = inspect.signature(cls.recommend_from_full)
        if 'pool' in signature.parameters:
            raise NotImplementedError(
                f"`pool` must not be a argument of `recommend_from_full` ..."
            )

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
    
    def reset_ranking_buffers(self):
        self.ranking_buffer = dict()

    def recommend_from_full(self, **kwargs):
        raise NotImplementedError()

    def recommend_from_pool(self, *, pool: torch.Tensor, **kwargs):
        raise NotImplementedError()

    def _kwargs4full(self, kwargs: Dict):
        parameters = inspect.signature(self.recommend_from_full).parameters
        if any(param.kind == param.VAR_KEYWORD for param in parameters.values()):
            # recommen_from_full(..., **kwargs)
            return kwargs
        return {key: kwargs[key] for key in parameters}

    def _kwargs4pool(self, kwargs: Dict):
        parameters = inspect.signature(self.recommend_from_pool).parameters
        if any(param.kind == param.VAR_KEYWORD for param in parameters.values()):
            # recommen_from_pool(..., **kwargs)
            return kwargs
        return {key: kwargs[key] for key in parameters}

    def recommend(self, **kwargs):
        if kwargs.get('pool', None) is None:
            return self.recommend_from_full(**self._kwargs4full(kwargs))
        else:
            return self.recommend_from_pool(**self._kwargs4pool(kwargs))

    @overload
    def encode(self, users: torch.Tensor) -> torch.Tensor:
        r"""
        User encoding.

        users: torch.Tensor, (B, 1)
        """

    @overload
    def encode(self, seqs: torch.Tensor) -> torch.Tensor:
        r"""
        Seq encoding.

        seqs: torch.Tensor, (B, S)
        """

    @overload
    def predict(self, users: torch.Tensor) -> torch.Tensor:
        r"""
        User-Item scoring.

        users: torch.Tensor, (B, 1)
        """

    @overload
    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        r"""
        User-Item scoring.

        users: torch.Tensor, (B, 1)
        items: torch.Tensor, (B, K + 1)
        """

    @overload
    def predict(self, users: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> Tuple[torch.Tensor]:
        r"""
        User-Item scoring.

        users: torch.Tensor, (B, 1)
        positives: torch.Tensor, (B, 1)
        negatives: torch.Tensor, (B, K)
        """

    def predict(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.predict(*args, **kwargs)
        else:
            return self.recommend(**kwargs)


class GenRecArch(RecSysArch):

    def match_datapipe(
        self,
        dataset: RecDataSet, 
        mode: str = 'valid', ranking: str = 'full', batch_size: int = 256
    ) -> dp.iter.IterDataPipe:
        r"""
        Match a valid/test datapipe.

        Parameters:
        -----------
        dataset: RecDataSet
        mode: str, default to 'valid'
        ranking: str, ('full' or 'pool')
        batch_size: int

        Flows:
        ------
        UserID -> [UserID, Sequence, Unseen, Seen]
        """
        from ..data.postprocessing.source import OrderedIDs
        User = dataset.fields[USER, ID]
        Item = dataset.fields[ITEM, ID]

        datapipe = OrderedIDs(
            field=User
        ).sharding_filter()

        if mode == 'valid':
            datapipe = datapipe.valid_sampling_(
                dataset=dataset, ranking=ranking
            )
        elif mode == 'test':
            datapipe = datapipe.test_sampling_(
                dataset=dataset, ranking=ranking
            )
        else:
            raise NotImplementedError

        datapipe = dataset.batch(batch_size).column_().tensor_().field(
            User.buffer(), Item.buffer(SEQUENCE), Item.buffer(UNSEEN), Item.buffer(SEEN)
        )

        return datapipe
        

class SeqRecArch(RecSysArch):

    def match_datapipe(
        self,
        dataset: RecDataSet, 
        maxlen: int, left_padding: bool = True,
        NUM_PADS: int = 1, padding_value: int = 0,
        mode: str = 'valid', ranking: str = 'full', batch_size: int = 256,
    ) -> dp.iter.IterDataPipe:
        r"""
        Match a valid/test datapipe.

        Parameters:
        -----------
        dataset: RecDataSet
        maxlen: int
        left_padding: bool
            `True`: padding from left
            `False`: padding from right
        NUM_PADS: int
        padding_value: int
        mode: str, 'valid' or 'test'
        ranking: str
            `full`: datapipe for full ranking
            `pool`: datapipe for sampled-based ranking
        batch_size: int

        Flows:
        ------
        UserID -> [UserID, Sequence, Unseen, Seen]
        """
        from ..data.postprocessing.source import OrderedIDs
        User = dataset.fields[USER, ID]
        Item = dataset.fields[ITEM, ID]

        datapipe = OrderedIDs(
            field=User
        ).sharding_filter()

        if mode == 'valid':
            datapipe = datapipe.valid_sampling_(
                dataset=dataset, ranking=ranking
            )
        elif mode == 'test':
            datapipe = datapipe.test_sampling_(
                dataset=dataset, ranking=ranking
            )
        else:
            raise NotImplementedError

        datapipe = datapipe.lprune_(indices=[1], maxlen=maxlen)

        if ranking == 'full':
            datapipe = datapipe.add_(indicies=[1], offset=NUM_PADS)
        elif ranking == 'pool':
            datapipe = datapipe.add_(indicies=[1, 2], offset=NUM_PADS)
        else:
            raise NotImplementedError

        if left_padding:
            datapipe = datapipe.lpad_(
                indices=[1], maxlen=maxlen, padding_value=padding_value
            )
        else:
            datapipe = datapipe.rpad_(
                indices=[1], maxlen=maxlen, padding_value=padding_value
            )

        datapipe = datapipe.batch(batch_size).column_().tensor_().field_(
            User.buffer(), Item.buffer(SEQUENCE), Item.buffer(UNSEEN), Item.buffer(SEEN)
        )

        return datapipe