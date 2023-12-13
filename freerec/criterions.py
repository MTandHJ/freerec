

from typing import Iterable, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "BaseCriterion", 
    "CrossEntropy4Logits", "KLDivLoss4Logits",
    "BPRLoss", "BCELoss4Logits",
    "MSELoss", "L1Loss",
]


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

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, 
        reduction: Optional[str] = None
    ):
        reduction = reduction if reduction is not None else self.reduction
        return cross_entropy_with_logits(logits, targets, reduction)


class BCELoss4Logits(BaseCriterion):
    """Binary Cross Entropy with logits !!!"""

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
        reduction: Optional[str] = None
    ):
        reduction = reduction if reduction is not None else self.reduction
        return binary_cross_entropy_with_logits(logits, targets, reduction)


class KLDivLoss4Logits(BaseCriterion):
    """KLDivLoss with logits"""

    def __init__(self, reduction: str = 'batchmean') -> None:
        super().__init__(reduction)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
        reduction: Optional[str] = None
    ):
        reduction = reduction if reduction is not None else self.reduction
        return kl_div_loss_with_logits(logits, targets, reduction)
        

class BPRLoss(BaseCriterion):

    def forward(
        self, pos_scores: torch.Tensor, neg_scores: torch.Tensor,
        reduction: Optional[str] = None
    ):
        reduction = reduction if reduction is not None else self.reduction
        return bpr_loss_with_logits(pos_scores, neg_scores, reduction)


class MSELoss(BaseCriterion):
    """Mean Square Loss."""

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor,
        reduction: Optional[str] = None
    ):
        reduction = reduction if reduction is not None else self.reduction
        return F.mse_loss(inputs, targets, reduction=reduction)


class L1Loss(BaseCriterion):
    """L1 Loss."""

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor,
        reduction: Optional[str] = None
    ):
        reduction = reduction if reduction is not None else self.reduction
        return F.l1_loss(inputs, targets, reduction=reduction)





def cross_entropy_with_logits(
    logits: torch.Tensor, targets: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    r"""
    Cross-entropy loss with logits.

    Parameters:
    -----------
    logits: torch.Tensor
    targets: torch.Tensor
    reduction: str, 'none'|'sum'|'mean' (default)

    Shapes:
    -------
    logits:     (C,)    (B, C)  (B, C, ...)
    targets:    ( ,)    (C,  )  (B, ...)
    """
    return F.cross_entropy(logits, targets, reduction=reduction)

def binary_cross_entropy_with_logits(
    logits: torch.Tensor, targets: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    r"""
    Binary cross-entropy loss with logits.

    Parameters:
    -----------
    logits: torch.Tensor
    targets: torch.Tensor, the same shape as `logits`
    reduction: str, 'none'|'sum'|'mean' (default)
    """
    return F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction=reduction)

def kl_div_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor,
    reduction: str = "batchmean"
) -> torch.Tensor:
    r"""
    KL divergence loss with logits.

    Parameters:
    -----------
    logits: torch.Tensor
    targets: torch.Tensor, the same shape as `logits`
    reduction: str, 'none'|'sum'|'mean'|'batchmean' (default)
    """
    assert logits.size() == targets.size()
    inputs = F.log_softmax(logits, dim=-1)
    targets = F.softmax(targets, dim=-1)
    return F.kl_div(inputs, targets, reduction=reduction)

def bpr_loss_with_logits(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    r"""
    BPR loss with logits.

    Parameters:
    -----------
    pos_scores: torch.Tensor
        Positive scores
    neg_scores: torch.Tensor
        Negative scores
    reduction: str, 'none'|'sum'|'mean' (default)
    """
    loss = F.softplus(neg_scores - pos_scores)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise NotImplementedError(f"reduction mode of '{reduction}' is not supported ...")