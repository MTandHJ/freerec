

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .base import SeqRecArch
from ..data.fields import Field
from ..data.datasets import RecDataSet
from ..data.postprocessing import PostProcessor
from ..criterions import BCELoss4Logits


class NARM(SeqRecArch):

    def __init__(
        self, dataset: RecDataSet,
        embedding_dim: int = 64, hidden_size: int = 128,
        emb_dropout_rate: float = 0.2, 
        hidden_dropout_rate: float = 0., 
        ct_dropout_rate: float = 0.5,
        num_blocks: int = 1
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks

        self.Item.add_module(
            'embeddings', nn.Embedding(
                num_embeddings=self.Item.count + self.NUM_PADS,
                embedding_dim=embedding_dim,
                padding_idx=self.PADDING_VALUE
            )
        )
        self.emb_dropout = nn.Dropout(emb_dropout_rate)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_blocks,
            bias=False,
            batch_first=True,
            dropout=hidden_dropout_rate
        )
        self.a_1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.a_2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_t = nn.Linear(hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(ct_dropout_rate)
        self.b = nn.Linear(2 * hidden_size, embedding_dim, bias=False)

        self.criterion = BCELoss4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def sure_trainpipe(self, maxlen: int, batch_size: int) -> PostProcessor:
        return self.dataset.train().shuffled_roll_seqs_source(
            minlen=2, maxlen=None
        ).sharding_filter().seq_train_yielding_pos_(
            start_idx_for_target=-1 # only last item as the target
        ).seq_train_sampling_neg_(
            num_negatives=1
        ).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).rpad_( # [i, j, k, ..., 0, ..., 0]
            maxlen, modified_fields=(self.ISeq,)
        ).batch_(batch_size).tensor_()

    def sure_validpipe(self, maxlen: int, ranking: str = 'full', batch_size: int = 256):
        return self.dataset.valid().ordered_user_ids_source().sharding_filter(
        ).valid_sampling_(
            ranking
        ).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).rpad_(
            maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def sure_testpipe(self, maxlen: int, ranking: str = 'full', batch_size: int = 256):
        return self.dataset.test().ordered_user_ids_source().sharding_filter(
        ).test_sampling_(
            ranking
        ).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).rpad_(
            maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def shrink_pads(self, seqs: torch.Tensor):
        mask = seqs.not_equal(self.PADDING_VALUE)
        keeped = mask.any(dim=0, keepdim=False)
        return seqs[:, keeped], mask[:, keeped].unsqueeze(-1) # (B, S), (B, S, 1)
    
    def encode(self, data: Dict[Field, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs, mask = self.shrink_pads(data[self.ISeq])

        seqs = self.Item.embeddings(seqs)
        seqs = self.emb_dropout(seqs)
        gru_out, _ = self.gru(seqs) # (B, S, H)

        c_global = ht = gru_out.gather(
            dim=1,
            index=mask.sum(1, keepdim=True).add(-1).expand((-1, 1, gru_out.size(-1)))
        ) # (B, 1, H)

        q1 = self.a_1(gru_out) # (B, S, H)
        q2 = self.a_2(ht) # (B, 1, H)

        alpha = self.v_t(mask * torch.sigmoid(q1 + q2)) # (B, S, 1)
        c_local = torch.sum(alpha * gru_out, 1) # (B, H)
        c_t = torch.cat([c_local, c_global.squeeze(1)], 1) # (B, 2H)
        c_t = self.ct_dropout(c_t)
        userEmbds = self.b(c_t) # (B, D)
        return userEmbds, self.Item.embeddings.weight[self.NUM_PADS:]

    def fit(self, data: Dict[Field, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        userEmbds, itemEmbds = self.encode(data)
        posEmbds = itemEmbds[data[self.IPos]] # (B, 1, D)
        negEmbds = itemEmbds[data[self.INeg]] # (B, 1, K, D)

        posLogits = torch.einsum("BD,BSD->BS", userEmbds, posEmbds)
        negLogits = torch.einsum("BD,BSKD->BSK", userEmbds, negEmbds)
        posLabels = torch.ones_like(posLogits)
        negLabels = torch.zeros_like(negLogits)

        indices = data[self.ISeq] != 0
        rec_loss = self.criterion(posLogits[indices], posLabels[indices]) + \
            self.criterion(negLogits[indices], negLabels[indices])

        return rec_loss

    def recommend_from_full(self, data: Dict[Field, torch.Tensor]) -> torch.Tensor:
        userEmbds, itemEmbds = self.encode(data)
        return torch.einsum("BD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[Field, torch.Tensor]) -> torch.Tensor:
        userEmbds, itemEmbds = self.encode(data)
        itemEmbds = itemEmbds[data[self.IUnseen]] # (B, K, D)
        return torch.einsum("BD,BKD->BK", userEmbds, itemEmbds)