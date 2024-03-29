

from typing import Optional, Union, overload, Tuple, Dict

import torch, inspect
import torch.nn as nn


__all__ = ['RecSysArch']


class RecSysArch(nn.Module):
    r"""
    A PyTorch Module for recommendation system architecture.
    This module contains methods for broadcasting tensors and initializing the module parameters.

    Parameters:
    -----------
    nn.Module : PyTorch Module
        The base class of this module.

    Methods:
    --------
    to(device, dtype, non_blocking)
        Moves and/or casts the parameters and buffers.

    initialize()
        Initializes the module parameters.

    broadcast(*tensors)
        Broadcasts the given tensors according to broadcasting semantics.

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

    @staticmethod
    def broadcast(*tensors: torch.Tensor):
        r"""
        Broadcasts the given tensors according to Broadcasting semantics.
        See [here](https://pytorch.org/docs/stable/generated/torch.broadcast_tensors.html#torch.broadcast_tensors) for details.

        Parameters:
        -----------
        tensors : torch.Tensor
            The tensors to broadcast.

        Returns:
        --------
        Tuple[torch.Tensor, ...]
            The broadcasted tensors.

        Examples:
        ---------
        >>> users = torch.rand(4, 1, 4)
        >>> items = torch.rand(4, 2, 1)
        >>> users, items = RecSysArch.broadcast(users, items)
        >>> users.size()
        torch.Size(4, 2, 4)
        >>> items.size()
        torch.Size(4, 2, 4)
        """
        return torch.broadcast_tensors(*tensors)

    @overload
    def recommend_from_full(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Returns:
        --------
        user features: torch.Tensor
        item features: torch.Tensor
        """

    @overload
    def recommend_from_full(self, seqs: torch.Tensor) -> torch.Tensor:
        r"""
        Full ranking recommendation.

        Parameters:
        -----------
        seqs: torch.Tensor, (B, S)
            A batch of sequences.
        
        Returns:
        --------
        scores: torch.Tensor, (B, N)
            `N' denotes the number of items.
        """

    @overload
    def recommend_from_pool(self, seqs: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
        r"""
        Sampling-based recommendation.

        Parameters:
        -----------
        seqs: torch.Tensor, (B, S)
            A batch of sequences.
        pool: torch.Tensor, (B, K)
            A batch of items.

        Returns:
        --------
        scores: torch.Tensor, (B, K)
        """

    def recommend_from_full(self, **kwargs):
        raise NotImplementedError()

    def recommend_from_pool(self, *, pool: torch.Tensor, **kwargs):
        raise NotImplementedError()

    def _kwargs4full(self, kwargs: Dict):
        parameters = inspect.signature(self.recommend_from_full).parameters
        if any(param.kind == param.VAR_KEYWORD for param in parameters.values()):
            return kwargs
        return {key: kwargs[key] for key in parameters}

    def _kwargs4pool(self, kwargs: Dict):
        parameters = inspect.signature(self.recommend_from_pool).parameters
        if any(param.kind == param.VAR_KEYWORD for param in parameters.values()):
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