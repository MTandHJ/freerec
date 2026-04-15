

from typing import Optional, Callable

import torch
import torch.nn as nn

__all__ = ['FeedForwardNetwork']


class FeedForwardNetwork(nn.Module):
    r"""Feed-Forward Network (FFN) with residual connection and layer normalization.

    Parameters
    ----------
    embedding_dim : int
        Size of the input embedding dimension.
    activation : callable, optional
        Activation function class used between linear layers,
        by default :class:`torch.nn.ReLU`.
    hidden_size : int or None, optional
        Hidden dimension of the FFN. If ``None``, it defaults to
        ``4 * embedding_dim``.
    hidden_dropout_rate : float, optional
        Dropout rate applied to the output, by default ``0.0``.
    norm_eps : float, optional
        Epsilon value for :class:`torch.nn.LayerNorm`, by default ``1e-12``.
    bias : bool, optional
        Whether to add bias terms in :class:`torch.nn.Linear` layers,
        by default ``False``.
    """

    def __init__(
        self,
        embedding_dim: int,
        activation: Callable = nn.ReLU,
        hidden_size: Optional[int] = None,
        hidden_dropout_rate: float = 0.,
        norm_eps: float = 1.e-12,
        bias: bool = False,
    ):
        r"""Initialize FeedForwardNetwork."""
        super().__init__()

        self.hidden_size = hidden_size if hidden_size is not None else 4 * embedding_dim
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_size, bias=bias),
            activation(),
            nn.Linear(self.hidden_size, embedding_dim, bias=bias),
            nn.Dropout(p=hidden_dropout_rate)
        )
        self.norm = nn.LayerNorm(embedding_dim, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the feed-forward transformation with residual connection.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Input tensor of shape ``(B, L, D)``, where ``B`` is the batch
            size, ``L`` is the sequence length, and ``D`` is the embedding
            dimension.

        Returns
        -------
        :class:`torch.Tensor`
            Output tensor of shape ``(B, L, D)``.
        """
        return self.norm(self.ffn(x) + x)
