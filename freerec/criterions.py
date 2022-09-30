

from typing import Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import errorLogger


__all__ = ["BaseCriterion", "BCELoss", "MSELoss", "L1Loss"]


class BaseCriterion(nn.Module):
    """Criterion."""

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        assert reduction in ('none', 'sum', 'mean'), f"Invalid reduction of {reduction} got ..."
        self.reduction = reduction

    def regularize(self, params: Union[torch.Tensor, Iterable[torch.Tensor]], rtype: str = 'l2'):
        """Add regularization for given parameters.

        Parameters:
        ---

        params: List of parameters for regularization.
        rtype: Some kind of regularization including 'l1'|'l2'.
        """
        params = [params] if isinstance(params, torch.Tensor) else params
        if self.rtype == 'l1':
            return sum(param.abs().sum() for param in params)
        elif self.rtype == 'l2':
            return sum(param.pow(2).sum() for param in params) / 2
        else:
            errorLogger(f"{rtype} regularization is not supported ...", NotImplementedError)


class BCELoss(BaseCriterion):
    """Binary Cross Entropy with logits !!!"""

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return F.binary_cross_entropy_with_logits(inputs, targets.to(inputs.dtype), reduction=self.reduction)
    

class BPRLoss(BaseCriterion):

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        loss = F.softplus(neg_scores - pos_scores)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class MSELoss(BaseCriterion):
    """Mean Square Loss"""

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return F.mse_loss(inputs, targets, reduction=self.reduction)


class L1Loss(BaseCriterion):

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return F.l1_loss(inputs, targets, reduction=self.reduction)
