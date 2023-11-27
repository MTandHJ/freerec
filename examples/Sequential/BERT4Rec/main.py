

import torch
import torch.nn as nn

import freerec
from freerec.data.fields import FieldModuleList
from freerec.data.tags import USER, SESSION, ITEM, TIMESTAMP, ID, POSITIVE, UNSEEN, SEEN

freerec.declare(version='0.5.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=100)
cfg.add_argument("--num-heads", type=int, default=4)
cfg.add_argument("--num-blocks", type=int, default=2)
cfg.add_argument("--hidden-size", type=int, default=128)
cfg.add_argument("--dropout-rate", type=float, default=0.1)

cfg.add_argument("--mask-prob", type=float, default=0.2, help="the probability of masking")

cfg.add_argument("--decay-step", type=int, default=25)
cfg.add_argument("--decay-factor", type=float, default=1., help="lr *= factor per decay step")

cfg.set_defaults(
    description="BERT4Rec",
    root="../../data",
    dataset='MovieLens1M_550_Chron',
    epochs=100,
    batch_size=256,
    optimizer='adamw',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


NUM_PADS = 2


class BERT4Rec(freerec.models.RecSysArch):

    def __init__(
        self, fields: FieldModuleList,
        maxlen: int = cfg.maxlen,
        hidden_size: int = cfg.hidden_size,
        dropout_rate: float = cfg.dropout_rate,
        num_blocks: int = cfg.num_blocks,
        num_heads: int = cfg.num_heads,
    ) -> None:
        super().__init__()

        self.num_blocks = num_blocks
        self.fields = fields
        self.Item = self.fields[ITEM, ID]

        self.Position = nn.Embedding(maxlen, hidden_size)
        self.embdDropout = nn.Dropout(p=dropout_rate)
        self.register_buffer(
            "positions_ids",
            torch.tensor(range(0, maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_blocks
        )

        self.fc = nn.Linear(hidden_size, self.Item.count + NUM_PADS)

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

    def mark_position(self, seqs: torch.Tensor):
        positions = self.Position(self.positions_ids) # (1, maxlen, D)
        return seqs + positions

    def forward(self, seqs: torch.Tensor):
        padding_mask = seqs == 0
        seqs = self.mark_position(self.Item.look_up(seqs)) # (B, S) -> (B, S, D)
        seqs = self.dropout(self.layernorm(seqs))

        seqs = self.encoder(seqs, src_key_padding_mask=padding_mask)

        logits = self.fc(seqs) # (B, S, N + 2)
        return logits

    def predict(self, seqs: torch.Tensor):
        return self.forward(seqs)

    def recommend_from_pool(self, seqs: torch.Tensor, pool: torch.Tensor):
        logits = self.forward(seqs)
        scores = logits[:, -1, :] # (B, N + 2)
        return scores.gather(1, pool) # (B, 101)

    def recommend_from_full(self, seqs: torch.Tensor):
        logits = self.forward(seqs)
        scores = logits[:, -1, :] # (B, N + 2)
        return scores[:, NUM_PADS:]

class CoachForBERT4Rec(freerec.launcher.SeqCoach):

    def random_mask(self, seqs: torch.Tensor, p: float = cfg.mask_prob):
        padding_mask = seqs == 0
        rnds = torch.rand(seqs.size(), device=seqs.device)
        masked_seqs = torch.where(rnds < p, torch.ones_like(seqs), seqs)
        masked_seqs.masked_fill_(padding_mask, 0)
        masks = (masked_seqs == 1) # (B, S)
        labels = seqs[masks]
        return masked_seqs, labels, masks

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            users, seqs  = [col.to(self.device) for col in data]
            seqs, labels, masks = self.random_mask(seqs)
            logits = self.model(seqs)[masks]
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(loss.item(), n=users.size(0), mode="mean", prefix='train', pool=['LOSS'])


def main():

    dataset = getattr(freerec.data.datasets.sequential, cfg.dataset)(root=cfg.root)
    User, Item = dataset.fields[USER, ID], dataset.fields[ITEM, ID]

    # trainpipe
    trainpipe = freerec.data.postprocessing.source.RandomShuffledSource(
        source=dataset.train().to_seqs(keepid=True)
    ).sharding_filter().lprune_(
        indices=[1], maxlen=cfg.maxlen
    ).add_(
        indices=[1], offset=NUM_PADS
    ).lpad_(
        indices=[1], maxlen=cfg.maxlen, padding_value=0 # 0: padding; 1: mask token
    ).batch(cfg.batch_size).column_().tensor_()

    # validpipe
    if cfg.ranking == 'full':
        validpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().seq_valid_yielding_(
            dataset
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen - 1,
        ).add_(
            indices=[1], offset=NUM_PADS
        ).lpad_(
            indices=[1], maxlen=cfg.maxlen - 1, padding_value=0
        ).rpad_(
            indices=[1], maxlen=cfg.maxlen, padding_value=1 # 1: mask token
        ).batch(128).column_().tensor_().field_(
            User.buffer(), Item.buffer(tags=POSITIVE), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
        )
    elif cfg.ranking == 'pool':
        validpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().seq_valid_sampling_(
            dataset # yielding (user, items, (target + (100) negatives))
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen - 1,
        ).add_(
            indices=[1, 2], offset=NUM_PADS
        ).lpad_(
            indices=[1], maxlen=cfg.maxlen - 1, padding_value=0
        ).rpad_(
            indices=[1], maxlen=cfg.maxlen, padding_value=1 # 1: mask token
        ).batch(128).column_().tensor_()

    # testpipe
    if cfg.ranking == 'full':
        testpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().seq_test_yielding_(
            dataset
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen - 1,
        ).add_(
            indices=[1], offset=NUM_PADS
        ).lpad_(
            indices=[1], maxlen=cfg.maxlen - 1, padding_value=0
        ).rpad_(
            indices=[1], maxlen=cfg.maxlen, padding_value=1 # 1: mask token
        ).batch(100).column_().tensor_().field_(
            User.buffer(), Item.buffer(tags=POSITIVE), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
        )
    elif cfg.ranking == 'pool':
        testpipe = freerec.data.postprocessing.source.OrderedIDs(
            field=User
        ).sharding_filter().seq_test_sampling_(
            dataset # yielding (user, items, (target + (100) negatives))
        ).lprune_(
            indices=[1], maxlen=cfg.maxlen - 1,
        ).add_(
            indices=[1, 2], offset=NUM_PADS
        ).lpad_(
            indices=[1], maxlen=cfg.maxlen - 1, padding_value=0
        ).rpad_(
            indices=[1], maxlen=cfg.maxlen, padding_value=1 # 1: mask token
        ).batch(cfg.batch_size).column_().tensor_()

    Item.embed(
        cfg.hidden_size, 
        num_embeddings=Item.count + NUM_PADS,
        padding_idx=0
    )
    tokenizer = FieldModuleList(dataset.fields)
    model = BERT4Rec(
        tokenizer
    )

    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, 
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay
        )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=cfg.decay_step,
        gamma=cfg.decay_factor
    )
    criterion = freerec.criterions.CrossEntropy4Logits()

    coach = CoachForBERT4Rec(
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        fields=dataset.fields,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=cfg.device
    )
    coach.compile(
        cfg, 
        monitors=[
            'loss', 
            'hitrate@1', 'hitrate@5', 'hitrate@10',
            'ndcg@5', 'ndcg@10'
        ],
        which4best='ndcg@10'
    )
    coach.fit()


if __name__ == "__main__":
    main()