

from typing import Optional, Callable

import torch
import torch.nn as nn

__all__ = ['FeedForwardNetwork']


class FeedForwardNetwork(nn.Module):
    r"""
    Feed Forward Network (FFN).

    Parameters:
    -----------
    embedding_dim : int
        Size of the input tensor (embedding dimension).
    activation : Callable
        Activation function to use between linear layers.
    hidden_size : Optional[int]
        Hidden dimension of the FFN. 
        If `None`, it defaults to 4 * embedding_dim.
    hidden_dropout_rate : float
        Dropout rate applied to the output.
    norm_eps : float
        Epsilon value for LayerNorm.
    bias : bool
        Whether to add a bias term in nn.Linear layers.
    """

    def __init__(
        self,
        embedding_dim: int,
        activation: Callable = nn.GELU,
        hidden_size: Optional[int] = None,
        hidden_dropout_rate: float = 0.,
        norm_eps: float = 1.e-12,
        bias: bool = False,
    ):
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
        r"""
        Forward pass for Feed Forward Network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (B, L, D), 
            where B is batch size, L is sequence length, D is embedding dimension.

        Returns:
        --------
        z : torch.Tensor
            Output tensor of shape (B, L, D).
        """
        return self.norm(self.ffn(x) + x)