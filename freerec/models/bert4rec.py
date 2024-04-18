

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .base import SeqRecArch
from ..data.fields import Field
from ..data.datasets import RecDataSet
from ..data.postprocessing import PostProcessor
from ..criterions import CrossEntropy4Logits


class BERT4Rec(SeqRecArch):

    NUM_PADS = 2
    PADDING_VALUE = 0
    MASKING_VALUE = 1

    def __init__(
        self, dataset: RecDataSet,
        maxlen: int = 50, embedding_dim: int = 64,
        mask_ratio: float = 0.3, dropout_rate: float = 0.2, 
        num_blocks: int = 1, num_heads: int = 2,
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = embedding_dim
        self.mask_ratio = mask_ratio
        self.num_blocks = num_blocks

        self.Item.add_module(
            "embeddings", nn.Embedding(
                self.Item.count + self.NUM_PADS, embedding_dim,
                padding_idx=self.PADDING_VALUE
            )
        )

        self.Position = nn.Embedding(maxlen, embedding_dim)
        self.embdDropout = nn.Dropout(p=dropout_rate)
        self.register_buffer(
            "positions",
            torch.tensor(range(0, maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.layernorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_blocks
        )

        self.fc = nn.Linear(embedding_dim, self.Item.count + self.NUM_PADS)

        self.criterion = CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.weight.data.clamp_(-0.02, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
                m.weight.data.clamp_(-0.02, 0.02)

    def sure_trainpipe(self, maxlen: int, batch_size: int) -> PostProcessor:
        return self.dataset.train().shuffled_seqs_source(
            maxlen
        ).sharding_filter().add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,)
        ).batch_(batch_size).tensor_()

    def sure_validpipe(self, maxlen: int, ranking: str = 'full', batch_size: int = 256):
        return self.dataset.valid().ordered_user_ids_source().sharding_filter(
        ).valid_sampling_(
            ranking=ranking
        ).lprune_(
            maxlen - 1, modified_fields=(self.ISeq,)
        ).add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen - 1, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
        ).rpad_(
            maxlen, modified_fields=(self.ISeq,), padding_value=self.MASKING_VALUE
        ).batch_(batch_size).tensor_()

    def sure_testpipe(self, maxlen: int, ranking: str = 'full', batch_size: int = 256):
        return self.dataset.test().ordered_user_ids_source().sharding_filter(
        ).test_sampling_(
            ranking=ranking
        ).lprune_(
            maxlen - 1, modified_fields=(self.ISeq,)
        ).add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen - 1, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
        ).rpad_(
            maxlen, modified_fields=(self.ISeq,), padding_value=self.MASKING_VALUE
        ).batch_(batch_size).tensor_()

    def mark_position(self, seqs: torch.Tensor):
        positions = self.Position(self.positions) # (1, maxlen, D)
        return seqs + positions

    def after_one_block(self, seqs: torch.Tensor, padding_mask: torch.Tensor, l: int):
        # inputs: (B, S, D)
        Q = self.attnLNs[l](seqs)
        seqs = self.attnLayers[l](
            Q, seqs, seqs, 
            attn_mask=self.attnMask,
            need_weights=False
        )[0] + seqs

        seqs = self.fwdLNs[l](seqs)
        seqs = self.fwdLayers[l](seqs)

        return seqs.masked_fill(padding_mask, 0.)
    
    def encode(self, data: Dict[Field, torch.Tensor]) -> torch.Tensor:
        seqs = data[self.ISeq]
        padding_mask = seqs == 0
        seqs = self.mark_position(self.Item.embeddings(seqs)) # (B, S) -> (B, S, D)
        seqs = self.dropout(self.layernorm(seqs))
        seqs = self.encoder(seqs, src_key_padding_mask=padding_mask) # (B, S, D)

        return seqs

    def random_mask(self, seqs: torch.Tensor, p: float):
        padding_mask = seqs == 0
        rnds = torch.rand(seqs.size(), device=seqs.device)
        masked_seqs = torch.where(
            rnds < p, 
            torch.ones_like(seqs).fill_(self.MASKING_VALUE),
            seqs
        )
        masked_seqs.masked_fill_(padding_mask, 0)
        masks = (masked_seqs == self.MASKING_VALUE) # (B, S)
        labels = seqs[masks]
        return masked_seqs, labels, masks

    def fit(self, data: Dict[Field, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        masked_seqs, labels, masks = self.random_mask(
            seqs=data[self.ISeq], p=self.mask_ratio
        )
        data[self.ISeq] = masked_seqs

        userEmbds = self.encode(data) # (B, S, D)
        logits = self.fc(userEmbds)[masks] # (B, S, N)
        rec_loss = self.criterion(logits, labels)

        return rec_loss

    def recommend_from_full(self, data: Dict[Field, torch.Tensor]) -> torch.Tensor:
        userEmbds = self.encode(data)
        userEmbds = userEmbds[:, -1, :] # (B, D)
        return self.fc(userEmbds)[:, self.NUM_PADS:]

    def recommend_from_pool(self, data: Dict[Field, torch.Tensor]) -> torch.Tensor:
        userEmbds = self.encode(data)
        userEmbds = userEmbds[:, -1, :] # (B, D)
        scores = self.fc(userEmbds)[:, self.NUM_PADS:] # (B, N)
        return scores.gather(data[self.IUnseen]) # (B, 101)