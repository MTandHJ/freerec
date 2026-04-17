import torch
import torch.nn as nn

__all__ = ["Unsqueeze"]


class Unsqueeze(nn.Module):
    r"""A module that unsqueezes a tensor along the specified dimension.

    Parameters
    ----------
    dim : int
        The dimension along which to unsqueeze.
    """

    def __init__(self, dim: int) -> None:
        r"""Initialize Unsqueeze with the target dimension."""
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        r"""Unsqueeze the input tensor along the configured dimension.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input tensor.

        Returns
        -------
        :class:`torch.Tensor`
            The unsqueezed tensor with an additional dimension.
        """
        return x.unsqueeze(self.dim)
