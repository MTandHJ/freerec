


import torch.nn as nn

from ..data.fields import Tokenizer

__all__ = ['RecSysArch']

class RecSysArch(nn.Module):

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()

        self.tokenizer = tokenizer

    def initialize(self): ... # TODO: