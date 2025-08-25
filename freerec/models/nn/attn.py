
from typing import Optional, Callable

import torch, math
import torch.nn as nn
from einops import rearrange

__all__ = ['ScaledDotProductAttention']


class ScaledDotProductAttention(nn.Module):
    r"""
    Scaled Dot Product Attention (SDPA).

    Parameters:
    -----------
    embedding_dim : int
        Size of the input tensor (embedding dimension).
    num_heads : int
        Number of attention heads.
    activation : Callable
        Activation function to use in the output projection.
    hidden_size : Optional[int]
        Hidden size used for attention score computation. 
        If `None`, it is inferred as embedding_dim // num_heads.
    hidden_dropout_rate : float
        Dropout rate applied to the output projection.
    attn_dropout_rate : float
        Dropout rate applied to the attention scores.
    norm_eps : float
        Epsilon value for LayerNorm.
    bias : bool
        Whether to add a bias term in nn.Linear layers.

    Raises:
    -------
    AssertionError
        If `hidden_size` is `None` and `embedding_dim` is not divisible by `num_heads`.
    """

    # value used for masking attention scores
    ATTENTION_MASK_VALUE = -1.e6

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 1,
        activation: Callable = nn.ReLU,
        hidden_size: Optional[int] = None,
        hidden_dropout_rate: float = 0.,
        attn_dropout_rate: float = 0.,
        norm_eps: float = 1.e-12,
        bias: bool = False,
    ):
        super().__init__()

        if hidden_size is None:
            assert embedding_dim % num_heads == 0, \
                f"`embedding_dim` value {embedding_dim} is not divisible by the `num_heads` {num_heads}"
            self.hidden_size =  embedding_dim // num_heads
        else:
            self.hidden_size = hidden_size

        self.scale_factor = math.sqrt(self.hidden_size)

        self.to_q = nn.Linear(embedding_dim, num_heads * self.hidden_size, bias=bias)
        self.to_k = nn.Linear(embedding_dim, num_heads * self.hidden_size, bias=bias)
        self.to_v = nn.Linear(embedding_dim, num_heads * self.hidden_size, bias=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout_rate)

        self.to_out = nn.Sequential(
            nn.Linear(num_heads * self.hidden_size, embedding_dim, bias=bias),
            activation(),
            nn.Dropout(p=hidden_dropout_rate)
        )
        self.norm = nn.LayerNorm(embedding_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.BoolTensor] = None
    ):
        r"""
        Forward pass for Scaled Dot-Product Attention.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (B, M, D), 
            where B is batch size, M is sequence length, D is embedding dimension.
        key : Optional[torch.Tensor]
            Optional key/value tensor of shape (B, N, D). 
            If not provided, uses x as key/value (self-attention).
        attn_mask : Optional[torch.BoolTensor]
            Optional attention mask of shape (M, N). 
            Positions with `True` will be masked by ATTENTION_MASK_VALUE.

        Returns:
        --------
        z : torch.Tensor
            Output tensor of shape (B, M, D).
        """

        key = x if key is None else key
        q = self.to_q(x)
        k, v = self.to_k(key), self.to_v(key)

        q = rearrange(q, "B L (H D) -> B H L D", D=self.hidden_size)
        k = rearrange(k, "B L (H D) -> B H L D", D=self.hidden_size)
        v = rearrange(v, "B L (H D) -> B H L D", D=self.hidden_size)

        scores = torch.einsum("B H M D, B H N D -> B H M N", q, k) / self.scale_factor
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, self.ATTENTION_MASK_VALUE)
        scores = self.attn_dropout(scores.softmax(dim=-1))

        z = torch.einsum("B H M N, B H N D -> B H M D", scores, v)
        z = rearrange(z, "B H L D -> B L (H D)")
        z = self.norm(self.to_out(z) + x)
        return z
