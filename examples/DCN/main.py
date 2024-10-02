

from typing import Dict, Tuple

import torch
import torch.nn as nn
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3)

cfg.set_defaults(
    description="DCN",
    root="../../data",
    dataset='Criteo',
    epochs=1000,
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


class DCN(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        embedding_dim: int = 64, num_layers: int = 3
    ) -> None:
        super().__init__(dataset)

        self.num_layers = num_layers

        for field in self.fields:
            field.add_module(
                "embeddings", nn.Embedding(
                    field.count, embedding_dim
                )
            )

        input_dim = len(self.fields) * embedding_dim

        self.dnn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.crossnet = nn.ModuleList(
            [CrossInteraction(input_dim) for _ in range(3)]
        )

        self.fc = nn.Linear(input_dim + 32, 1)

        self.criterion = freerec.criterions.BCELoss4Logits(reduction='mean')

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

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().choiced_user_ids_source(
        ).gen_train_sampling_pos_().gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def encode(self, data: Dict[freerec.data.fields.Field, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat([field.embeddings(data[field]) for field in self.fields], dim=-1) # (B, D)
        dnnEmbds = self.dnn(embeddings) # (B, D')
        crossEmbds = self.crossnet(embeddings) # (B, D)
        finalEmbds = torch.cat((dnnEmbds, crossEmbds), dim=-1) # (B, D + D')
        return self.fc(finalEmbds) # (B, 1)

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        logits = self.encode(data)
        labels = self.data[self.Label] # (B, 1)
        rec_loss = self.criterion(logits, labels)
        return rec_loss

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        logits = self.encode(data).flatten() # (B,)
        return logits.sigmoid() # scores


class CoachForLightGCN(freerec.launcher.Coach):

    def set_optimizer(self):
        if self.cfg.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.cfg.lr, 
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
                # weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                # weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                # weight_decay=self.cfg.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Unexpected optimizer {self.cfg.optimizer} ..."
            )

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            rec_loss, emb_loss = self.model(data)
            loss = rec_loss + self.cfg.weight_decay * emb_loss

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
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = LightGCN(
        dataset,
        embedding_dim=cfg.embedding_dim, num_layers=cfg.num_layers
    )

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForLightGCN(
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