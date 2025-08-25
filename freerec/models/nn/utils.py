

import torch
import torch.nn as nn


__all__ = ['Unsqueeze']


class Unsqueeze(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.dim)