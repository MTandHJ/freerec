

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import getLogger


__all__ = ["Regularizer", "BaseCriterion", "BCELoss", "MSELoss", "L1Loss"]


class Regularizer(nn.Module):

    def __init__(
        self, parameters: List[nn.parameter.Parameter],
        rtype: str = 'l2', weight: float = 0.
    ) -> None:
        """
        Args:
            parameters: list of parameters for regularization;
        Kwargs: 
            rtype: some kind of regularization including 'l1'|'l2'(default)
            weight: float
        """
        super().__init__()
        self.parameters = parameters
        assert rtype in ('l1', 'l2'), "only 'l1'|'l2' regularization are supported ..."
        self.rtype = rtype
        self.weight = weight
        getLogger().info(f"[Regularizer] >>> Add {self.rtype} regularization with the weight of {self.weight}")

    def forward(self) -> torch.Tensor:
        if self.rtype == 'l1':
            loss = sum(enem.abs().sum() for enem in self.parameters)
        elif self.rtype == 'l2':
            loss = sum(enem.pow(2).sum() for enem in self.parameters)
        return loss * self.weight


class BaseCriterion(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.regularizers: List[Regularizer] = []

        getLogger().info(f"[Criterion] >>> Employ the criterion {self.__class__.__name__}")

    def regulate(self, parameters: List[nn.parameter.Parameter], rtype: str, weight: float = 0.):
        """add regularization for given parameters
        Args:
            parameters: list of parameters for regularization;
        Kwargs: 
            rtype: some kind of regularization including 'l1'|'l2'(default)
            weight: float
        """
        self.regularizers.append(
            Regularizer(
                parameters=parameters,
                rtype=rtype, weight=weight
            )
        )

    def forward(self) -> torch.Tensor:
        return sum(regularizer() for regularizer in self.regularizers)


class BCELoss(BaseCriterion):

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        loss = super().forward()
        loss += F.binary_cross_entropy_with_logits(inputs, targets.to(inputs.dtype), reduction='mean')
        return loss
    

class MSELoss(BaseCriterion):

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        loss = super().forward()
        loss += F.mse_loss(inputs, targets, reduction='mean')
        return loss


class L1Loss(BaseCriterion):

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        loss = super().forward()
        loss += F.l1_loss(inputs, targets, reduction='mean')
        return loss
