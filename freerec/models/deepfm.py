


from typing import Dict, List

import torch
import torch.nn as nn

from .base import RecSysArch

from ..data.fields import Tokenizer
from ..data.tags import SPARSE, DENSE, FEATURE
from ..layers import FM


__all__ = ['DeepFM']

class DeepFM(RecSysArch):

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        sparse_dim = self.tokenizer.calculate_dimension(FEATURE, SPARSE)
        dense_dim = self.tokenizer.calculate_dimension(FEATURE, DENSE)
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

    def forward(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]) -> torch.Tensor:
        sparse: List[torch.Tensor] = self.tokenizer.look_up(items, (FEATURE, SPARSE))
        dense: List[torch.Tensor] = self.tokenizer.look_up(items, (FEATURE, DENSE))

        outs_linear = self.linear(self.tokenizer.flatten_cat(sparse + dense))
        outs_fm = self.fm(self.tokenizer.flatten_cat(sparse))
        outs_dnn = self.dnn(self.tokenizer.flatten_cat(sparse + dense))
        outs = outs_linear + outs_fm + outs_dnn
        if self.training:
            return outs
        else:
            return outs.sigmoid()



