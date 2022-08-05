


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
        self.linear = nn.Linear(sparse_dim + dense_dim, 1, bias=False)
        self.fm = FM()
        self.dnn = nn.Sequential(
            nn.Linear(sparse_dim + dense_dim, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(True),
            # nn.Dropout(0.)
            nn.Linear(64, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(True),
            # nn.Dropout(0.)
            nn.Linear(64, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(True),
            # nn.Dropout(0.)
            nn.Linear(64, 1)
        )

        self.initialize()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        sparse: List[torch.Tensor] = self.tokenizer.look_up(inputs, features='sparse')
        dense: List[torch.Tensor] = self.tokenizer.look_up(inputs, features='dense')

        outs_linear = self.linear(self.tokenizer.flatten_cat(sparse + dense))
        outs_fm = self.fm(self.tokenizer.flatten_cat(sparse))
        outs_dnn = self.dnn(self.tokenizer.flatten_cat(sparse + dense))
        outs = outs_linear + outs_fm + outs_dnn

        return outs.sigmoid() # B x 1



