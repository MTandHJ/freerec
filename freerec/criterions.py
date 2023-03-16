

from typing import Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["BaseCriterion", "BCELoss", "MSELoss", "L1Loss"]


class BaseCriterion(nn.Module):
    r"""
    Criterion.

    Parameters:
    -----------
    reduction : str, default 'mean'
        Reduction method. Choices are 'none', 'sum', and 'mean'.

    Attributes:
    -----------
    reduction : str
        Reduction method.

    Methods:
    --------
    regularize(params: Union[torch.Tensor, Iterable[torch.Tensor]], rtype: str = 'l2') -> torch.Tensor
        Regularizes the given parameters with the specified regularization method.

    """

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        assert reduction in ('none', 'sum', 'mean'), f"Invalid reduction of {reduction} got ..."
        self.reduction = reduction

    @staticmethod
    def regularize(params: Union[torch.Tensor, Iterable[torch.Tensor]], rtype: str = 'l2'):
        r"""
        Add regularization for given parameters.

        Parameters:
        -----------
        params : Union[torch.Tensor, Iterable[torch.Tensor]]
            List of parameters for regularization.
        rtype : str, default 'l2'
            The type of regularization to use. Options include 'l1' and 'l2'.

        Returns:
        --------
        reg_loss : torch.Tensor
            The regularization loss tensor.
        """
        params = [params] if isinstance(params, torch.Tensor) else params
        if rtype == 'l1':
            return sum(param.abs().sum() for param in params)
        elif rtype == 'l2':
            return sum(param.pow(2).sum() for param in params) / 2
        else:
            raise NotImplementedError(f"{rtype} regularization is not supported ...")


class CrossEntropy4Logits(BaseCriterion):
    """Cross entropy loss with logits."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        return F.cross_entropy(logits, targets)


class BCELoss4Logits(BaseCriterion):
    """Binary Cross Entropy with logits !!!"""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        return F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction=self.reduction)
    

class BPRLoss(BaseCriterion):

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        r"""
        Parameters:
        -----------
        pos_scores: torch.Tensor
            Positive scores
        neg_scores: torch.Tensor
            Negative scores
        
        Returns:
        --------
        loss: torch.Tensor
        """
        loss = F.softplus(neg_scores - pos_scores)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class MSELoss(BaseCriterion):
    """Mean Square Loss."""

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return F.mse_loss(inputs, targets, reduction=self.reduction)


class L1Loss(BaseCriterion):
    """L1 Loss."""
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return F.l1_loss(inputs, targets, reduction=self.reduction)
