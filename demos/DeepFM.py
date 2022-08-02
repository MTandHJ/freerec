

import torch
import torch.nn as nn

from freerec.parser import Parser
from freerec.launcher import Coach
from freerec.models import DeepFM
from freerec.data.datasets import Criteo
from freerec.data.utils import DataLoader
from freerec.data.fields import Tokenizer


import argparse


METHOD = "DeepFM"
FMT = "{description}={optimizer}-{lr}-{weight_decay}={seed}"

parser = argparse.ArgumentParser()
parser.add_argument("root", type=str)
parser.add_argument("-eb", "--embedding-dim", type=int, default=4)

parser.add_argument("--optimizer", type=str, choices=("sgd", "adam"), default="sgd")
parser.add_argument("-mom", "--momentum", type=float, default=0.9,
                help="the momentum used for SGD")
parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                help="the first beta argument for Adam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                help="the second beta argument for Adam")
parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4,
                help="weight decay")
parser.add_argument("-lr", "--lr", "--LR", "--learning_rate", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("-b", "--batch_size", type=int, default=128)

# eval
parser.add_argument("--eval-train", action="store_true", default=False)
parser.add_argument("--eval-valid", action="store_false", default=True)
parser.add_argument("--eval-freq", type=int, default=5,
                help="for valid dataset only")

parser.add_argument("--nums-workers", type=int, default=0)

parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--benchmark", action="store_false", default=True, 
                help="cudnn.benchmark == True ?")
parser.add_argument("-m", "--description", type=str, default=METHOD)
args = parser.parse_args()
args.description = FMT.format(**args.__dict__)


cfg = Parser()
cfg.compile(args)


datapipe = Criteo(cfg.root, split='train').encoder(batch_size=cfg.batch_size)
trainloader = DataLoader(datapipe, num_workers=cfg.num_workers)
tokenizer = Tokenizer(datapipe.fields)
for feature in tokenizer.sparse:
    feature.embed(cfg.embedding_dim)
model = DeepFM(tokenizer).to(cfg.DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
coach = Coach(
    model=model,
    criterion=nn.BCELoss(),
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    device=cfg.DEVICE
)
coach.compile(cfg, callbacks=['loss'])
coach.fit(trainloader, trainloader, cfg.epochs, 0)






