



from typing import Dict
import torch
import torch.nn as nn

from ..data.fields import Tokenizer
from ..data.tags import USER, ITEM, ID, FEATURE
from .base import RecSysArch


__all__ = ['NeuCF']

class GMF(nn.Module):

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        userEmbeds, itemEmbeds = self.tokenizer.look_up(inputs, (FEATURE, ID))
        return userEmbeds * itemEmbeds

class MLP(nn.Module):

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        dimension = self.tokenizer.calculate_dimension(FEATURE, ID)
        self.fc = nn.Sequential(
            nn.Linear(dimension, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        userEmbeds, itemEmbeds = self.tokenizer.look_up(inputs, (FEATURE, ID))
        x = torch.cat([userEmbeds, itemEmbeds], dim=-1)
        x = self.fc(x)
        return x

class NeuCF(RecSysArch):

    def __init__(self, tokenizer_mf: Tokenizer, tokenizer_mlp: Tokenizer) -> None:
        super().__init__()

        self.mf = GMF(tokenizer_mf)
        self.mlp = MLP(tokenizer_mlp)
        self.User = tokenizer_mf.groupby(USER, ID)[0]
        self.Item = tokenizer_mf.groupby(ITEM, ID)[0]
        dimension = (self.User.dimension + self.Item.dimension) // 2 + 8
        self.linear = nn.Linear(dimension, 1)

    def preprocess(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        users = users[self.User.name]
        items = items[self.Item.name]
        users = users.repeat((1, items.size(1)))
        return {self.User.name: users, self.Item.name: items}

    def forward(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.preprocess(users, items)
        feature_mf = self.mf(inputs)
        feature_mlp = self.mlp(inputs)
        features = torch.cat([feature_mf, feature_mlp], dim=-1)
        outs = self.linear(features).squeeze()
        if self.training:
            return outs
        else:
            return outs.sigmoid()
