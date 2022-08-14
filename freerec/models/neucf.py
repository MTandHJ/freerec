



from typing import Dict, List
import torch
import torch.nn as nn

from ..data.fields import Tokenizer
from ..data.tags import USER, ITEM, ID, FEATURE, NEGATIVE
from .base import RecSysArch


__all__ = ['NeuCF']

class GMF(nn.Module):

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        sparse: List[torch.Tensor] = self.tokenizer.look_up(inputs, (FEATURE, ID))
        user_embddings, item_embeddings = sparse
        return (user_embddings * item_embeddings).flatten(1)

class MLP(nn.Module):

    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        dimension = self.tokenizer.calculate_dimension((FEATURE, ID))
        self.fc = nn.Sequential(
            nn.Linear(dimension, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        sparse: List[torch.Tensor] = self.tokenizer.look_up(inputs, (FEATURE, ID))
        user_embddings, item_embeddings = sparse
        x = self.tokenizer.flatten_cat([user_embddings, item_embeddings])
        x = self.fc(x)
        return x

class NeuCF(RecSysArch):

    def __init__(self, tokenizer_mf: Tokenizer, tokenizer_mlp: Tokenizer) -> None:
        super().__init__()

        self.mf = GMF(tokenizer_mf)
        self.mlp = MLP(tokenizer_mlp)
        self._User = tokenizer_mf.groupby((USER, ID))[0].name
        self._Item = tokenizer_mf.groupby((ITEM, ID))[0].name
        self._Negative = NEGATIVE.name
        dimension = (
            tokenizer_mf.calculate_dimension((USER, ID)) + 
            tokenizer_mf.calculate_dimension((ITEM, ID))
        ) // 2 + 8
        self.linear = nn.Linear(dimension, 1)

    def preprocess(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        users = users[self._User]
        items, negItems = items[self._Item], items[self._Negative]
        size = negItems.size(1) + 1
        users = users.repeat((1, size))
        items = torch.cat((items, negItems), dim=-1)
        return {self._User: users.view(-1, 1), self._Item: items.view(-1, 1)}

    def forward(self, users: Dict[str, torch.Tensor], items: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.preprocess(users, items)
        feature_mf = self.mf(inputs)
        feature_mlp = self.mlp(inputs)
        features = torch.cat([feature_mf, feature_mlp], dim=-1)
        outs = self.linear(features)
        return outs.sigmoid()

