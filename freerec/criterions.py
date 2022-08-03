

import torch
import torch.nn as nn



class BCELoss(nn.BCELoss):

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(inputs.dtype)
        return super().forward(inputs, targets)

class MSELoss(nn.MSELoss): ...
class MAELoss(nn.L1Loss): ...