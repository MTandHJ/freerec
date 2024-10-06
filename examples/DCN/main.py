

from typing import Dict, Tuple

import torch
import torch.nn as nn
import freerec
from freerec.data.tags import LABEL

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=4)
cfg.add_argument("--num-layers", type=int, default=3)
cfg.add_argument("--hidden-dims", type=str, default='64,32')
cfg.add_argument("--hidden-dropout-rate", type=float, default=0.)

cfg.set_defaults(
    description="DCN",
    root="../../data",
    dataset='Frappe_x1_BARS',
    epochs=100,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    ranking='pool',
    seed=1
)
cfg.compile()


class CrossInteraction(nn.Module):

    def __init__(self, input_dim: int):
        super(CrossInteraction, self).__init__()

        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X_0: torch.Tensor, X_i: torch.Tensor) -> torch.Tensor:
        interact_out = self.weight(X_i) * X_0 + self.bias
        return interact_out


class MLPBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_rate: float = 0.
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class DCN(freerec.models.PredRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet
    ) -> None:
        super().__init__(dataset)

        self.num_layers = cfg.num_layers

        self.input_fields = self.fields.match_not(LABEL)

        input_dim: int = 0
        for field in self.input_fields:
            field.add_module(
                "embeddings", nn.Embedding(
                    field.count, cfg.embedding_dim
                )
            )

        input_dim += len(self.input_fields) * cfg.embedding_dim

        hidden_dims = [input_dim] + list(map(int, cfg.hidden_dims.split(',')))
        blocks = []
        for input_dim, output_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            blocks.append(
                MLPBlock(
                    input_dim,
                    output_dim,
                    dropout_rate=cfg.hidden_dropout_rate
                )
            )
        self.dnn = nn.Sequential(*blocks) # input_dim -> hidden_dims[-1]

        self.crossnet = nn.ModuleList( # input_dim -> input_dim
            [CrossInteraction(input_dim) for _ in range(3)]
        )

        self.fc = nn.Linear(input_dim + hidden_dims[-1], 1)

        self.criterion = freerec.criterions.BCELoss4Logits(reduction='mean')

        self.reset_parameters()

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().shuffled_inter_source().batch_(
            batch_size
        ).tensor_()

    def encode(self, data: Dict[freerec.data.fields.Field, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat(
            [field.embeddings(data[field]).flatten(1) for field in self.input_fields], 
            dim=-1
        ) # (B, D)
        dnnEmbds = self.dnn(embeddings) # (B, D')
        crossEmbds = embeddings
        for crosslayer in self.crossnet:
            crossEmbds = crosslayer(embeddings, crossEmbds)
        finalEmbds = torch.cat((dnnEmbds, crossEmbds), dim=-1) # (B, D + D')
        return self.fc(finalEmbds) # (B, 1)

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        logits = self.encode(data)
        labels = data[self.Label] # (B, 1)
        rec_loss = self.criterion(logits, labels)
        return rec_loss

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        logits = self.encode(data)
        return logits.sigmoid() # scores


class CoachForDCN(freerec.launcher.Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(
                loss.item(), 
                n=len(data[self.User]), reduction="mean", 
                mode='train', pool=['LOSS']
            )


def main():

    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.PredictionRecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = DCN(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe()
    testpipe = model.sure_testpipe()

    coach = CoachForDCN(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    coach.fit()


if __name__ == "__main__":
    main()