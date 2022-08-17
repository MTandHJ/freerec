





from typing import Dict, List
import torch
import torch.nn as nn

from ..data.fields import Tokenizer
from ..data.tags import USER, ITEM, ID, FEATURE
from .base import RecSysArch



class BehaviorAggregator(nn.Module): ...


class SimpleX(RecSysArch):


    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.aggregator = ...


    def forward(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]) -> torch.Tensor:
        ...


    def user_tower(self, users: Dict[str, torch.Tensor]):
        embs = self.tokenizer.look_up(users, tags=(USER, ID))
        users, neighbors = embs[:, [0]], embs[:, 1:]


