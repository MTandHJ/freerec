

from typing import Union, Tuple, Dict

import torch
import torch.nn as nn

from .base import GenRecArch
from ..data.datasets.base import RecDataSet
from ..data.postprocessing import PostProcessor
from ..data.fields import Field


class MF(GenRecArch):

    def __init__(
        self, dataset: RecDataSet,
        embedding_dim: int = 64
    ) -> None:
        super().__init__(dataset)

        self.User.add_module(
            "embeddings", nn.Embedding(
                self.User.count, embedding_dim
            )
        )

        self.Item.add_module(
            "embeddings", nn.Embedding(
                self.Item.count, embedding_dim
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def sure_trainpipe(self, batch_size: int) -> PostProcessor:
        return self.dataset.train().choiced_user_ids_source().sharding_filter(
        ).gen_train_sampling_pos_().gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.User.embeddings.weight, self.Item.embeddings.weight

    def fit(self, data: Dict[Field, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        userEmbds, itemEmbds = self.encode()
        userEmbds = userEmbds[data[self.User]] # (B, 1, D)
        iposEmbds = itemEmbds[data[self.IPos]] # (B, 1, D)
        inegEmbds = itemEmbds[data[self.INeg]] # (B, K, D)
        return torch.einsum("BKD,BKD->BK", userEmbds, iposEmbds), \
            torch.einsum("BKD,BKD->BK", userEmbds, inegEmbds)

    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(self, data: Dict[Field, torch.Tensor]) -> torch.Tensor:
        userEmbds = self.ranking_buffer[self.User][data[self.User]] # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum("BKD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[Field, torch.Tensor]) -> torch.Tensor:
        userEmbds = self.ranking_buffer[self.User][data[self.User]] # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]] # (B, 101, D)
        return torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)