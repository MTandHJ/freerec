


from typing import Dict, List

import torch
import torch.nn as nn

from .base import RecSysArch

from ..data.fields import Tokenizer
from ..layers import FM


__all__ = ['DeepFM']

class DeepFM(RecSysArch):

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__(tokenizer)

        sparse_dim = self.tokenizer.calculate_dimension('sparse')
        dense_dim = self.tokenizer.calculate_dimension('dense')
        self.linear = nn.Linear(dense_dim, 1)
        self.fm = FM()
        self.dnn = nn.Sequential(
            nn.Linear(sparse_dim + dense_dim, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(True),
            # nn.Dropout(0.)
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Dropout(0.)
            nn.Linear(128, 1)
        )
        self.bias = nn.parameter.Parameter(torch.zeros((1,)), requires_grad=True)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        sparse: List[torch.Tensor] = self.tokenizer.look_up(inputs)
        dense: torch.Tensor = self.tokenizer.collect_dense(inputs)

        outs_linear = self.linear(dense)
        outs_fm = self.fm(sparse)
        outs_dnn = self.dnn(
            self.tokenizer.flatten_cat(sparse + [dense])
        )
        outs = outs_linear + outs_fm + outs_dnn + self.bias

        return outs.sigmoid() # B x 1



