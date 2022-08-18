


import torch.nn as nn

from ..data.fields import Tokenizer

__all__ = ['RecSysArch']

class RecSysArch(nn.Module):

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)