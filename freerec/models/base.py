


from typing import Optional, Union
import torch
import torch.nn as nn

from ..data.fields import Tokenizer

__all__ = ['RecSysArch']


class RecSysArch(nn.Module):

    def to(
        self, device: Optional[Union[int, torch.device]] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        non_blocking: bool = False
    ):
        if device:
            self.device = device
        return super().to(device, dtype, non_blocking)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)