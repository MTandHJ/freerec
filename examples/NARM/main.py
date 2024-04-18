

import torch
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--hidden-size", type=int, default=128)
cfg.add_argument("--emb-dropout-rate", type=float, default=0.2)
cfg.add_argument("--hidden-dropout-rate", type=float, default=0.)
cfg.add_argument("--ct-dropout-rate", type=float, default=0.5)
cfg.add_argument("--num-blocks", type=int, default=1, help="the number of GRU layers")

cfg.set_defaults(
    description="NARM",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=300,
    batch_size=512,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1.e-8,
    seed=1,
)
cfg.compile()


class CoachForNARM(freerec.launcher.Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.try_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(
                loss.item(), 
                n=len(data[self.model.User]), reduction="mean", 
                mode='train', pool=['LOSS']
            )


def main():

    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = freerec.models.NARM(
        dataset, 
        embedding_dim=cfg.embedding_dim, hidden_size=cfg.hidden_size,
        emb_dropout_rate=cfg.emb_dropout_rate, 
        hidden_dropout_rate=cfg.hidden_dropout_rate, 
        ct_dropout_rate=cfg.ct_dropout_rate,
        num_blocks=cfg.num_blocks
    )

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)


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

    coach = CoachForNARM(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
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