

from typing import List
import torch
import torch.nn as nn
# from torchrec.modules.deepfm import FactorizationMachine


__all__ = ['FM']



class FM(nn.Module):
    """Tensor: B x [d1 + d2 ...] -> Tensor: B x 1"""

    def forward(
        self,
        inputs,
    ) -> torch.Tensor:
        # flatten each embedding to be [B, N, D] -> [B, N*D], then cat them all on dim=1
        fm_input = inputs
        sum_of_input = torch.sum(fm_input, dim=1, keepdim=True)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        square_of_sum = sum_of_input * sum_of_input
        cross_term = square_of_sum - sum_of_square
        cross_term = torch.sum(cross_term, dim=1, keepdim=True) * 0.5  # [B, 1]
        return cross_term

