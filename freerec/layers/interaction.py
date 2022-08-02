



from typing import List
import torch
import torch.nn as nn
# from torchrec.modules.deepfm import FactorizationMachine


__all__ = ['FM']

def _get_flatten_input(inputs: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [input.flatten(1) for input in inputs],
        dim=1,
    )


class FM(nn.Module):
    """
    1. List[Tensor: B x k x d] -> Tensor: B x [k1 x d1 + k2 x d2 ...] -> Tensor: B x 1
    2. List[Tensor: B x d] -> Tensor: B x [d1 + d2 ...] -> Tensor: B x 1
    """

    def forward(
        self,
        embeddings,
    ) -> torch.Tensor:
        # flatten each embedding to be [B, N, D] -> [B, N*D], then cat them all on dim=1
        fm_input = _get_flatten_input(embeddings)
        sum_of_input = torch.sum(fm_input, dim=1, keepdim=True)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        square_of_sum = sum_of_input * sum_of_input
        cross_term = square_of_sum - sum_of_square
        cross_term = torch.sum(cross_term, dim=1, keepdim=True) * 0.5  # [B, 1]
        return cross_term

