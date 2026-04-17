import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from einops import rearrange

__all__ = ["ScaledDotProductAttention"]


class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention (SDPA).

    Implements multi-head scaled dot-product attention with residual
    connection and layer normalization.

    Parameters
    ----------
    embedding_dim : int
        Size of the input embedding dimension.
    num_heads : int, optional
        Number of attention heads, by default ``1``.
    activation : callable, optional
        Activation function class for the output projection,
        by default :class:`torch.nn.ReLU`.
    hidden_size : int or None, optional
        Hidden size per attention head. If ``None``, it is inferred as
        ``embedding_dim // num_heads``.
    hidden_dropout_rate : float, optional
        Dropout rate applied to the output projection, by default ``0.0``.
    attn_dropout_rate : float, optional
        Dropout rate applied to the attention scores, by default ``0.0``.
    norm_eps : float, optional
        Epsilon value for :class:`torch.nn.LayerNorm`, by default ``1e-12``.
    bias : bool, optional
        Whether to add bias terms in :class:`torch.nn.Linear` layers,
        by default ``False``.

    Raises
    ------
    AssertionError
        If ``hidden_size`` is ``None`` and ``embedding_dim`` is not
        divisible by ``num_heads``.

    Attributes
    ----------
    ATTENTION_MASK_VALUE : float
        Value used for masking attention scores (``-1e6``).
    """

    # value used for masking attention scores
    ATTENTION_MASK_VALUE = -1.0e6

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 1,
        activation: Callable = nn.ReLU,
        hidden_size: Optional[int] = None,
        hidden_dropout_rate: float = 0.0,
        attn_dropout_rate: float = 0.0,
        norm_eps: float = 1.0e-12,
        bias: bool = False,
    ):
        r"""Initialize ScaledDotProductAttention."""
        super().__init__()

        if hidden_size is None:
            assert embedding_dim % num_heads == 0, (
                f"`embedding_dim` value {embedding_dim} is not divisible by the `num_heads` {num_heads}"
            )
            self.hidden_size = embedding_dim // num_heads
        else:
            self.hidden_size = hidden_size

        self.scaling_factor = math.sqrt(self.hidden_size)

        self.to_q = nn.Linear(embedding_dim, num_heads * self.hidden_size, bias=bias)
        self.to_k = nn.Linear(embedding_dim, num_heads * self.hidden_size, bias=bias)
        self.to_v = nn.Linear(embedding_dim, num_heads * self.hidden_size, bias=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout_rate)

        self.to_out = nn.Sequential(
            nn.Linear(num_heads * self.hidden_size, embedding_dim, bias=bias),
            activation(),
            nn.Dropout(p=hidden_dropout_rate),
        )
        self.norm = nn.LayerNorm(embedding_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.BoolTensor] = None,
    ):
        r"""Compute scaled dot-product attention.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Input tensor of shape ``(B, M, D)``, where ``B`` is the batch
            size, ``M`` is the query sequence length, and ``D`` is the
            embedding dimension.
        key : :class:`torch.Tensor` or None, optional
            Key/value tensor of shape ``(B, N, D)``. If ``None``, ``x``
            is used as both key and value (self-attention).
        attn_mask : :class:`torch.BoolTensor` or None, optional
            Attention mask of shape ``(M, N)``. Positions set to ``True``
            are filled with ``ATTENTION_MASK_VALUE``.

        Returns
        -------
        :class:`torch.Tensor`
            Output tensor of shape ``(B, M, D)``.
        """

        key = x if key is None else key
        q = self.to_q(x)
        k, v = self.to_k(key), self.to_v(key)

        q = rearrange(q, "B L (H D) -> B H L D", D=self.hidden_size)
        k = rearrange(k, "B L (H D) -> B H L D", D=self.hidden_size)
        v = rearrange(v, "B L (H D) -> B H L D", D=self.hidden_size)

        scores = torch.einsum("B H M D, B H N D -> B H M N", q, k) / self.scaling_factor
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, self.ATTENTION_MASK_VALUE)
        scores = self.attn_dropout(scores.softmax(dim=-1))

        z = torch.einsum("B H M N, B H N D -> B H M D", scores, v)
        z = rearrange(z, "B H L D -> B L (H D)")
        z = self.norm(self.to_out(z) + x)
        return z
