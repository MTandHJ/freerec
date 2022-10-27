



## Requirements: 

Python >= 3.9 | [PyTorch >= 1.12.1](https://pytorch.org/) | [TorchData](https://github.com/pytorch/data) | [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#)


## Installation

    pip install freerec


## Help


```
optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           data
  --config CONFIG       config.yml
  --optimizer {sgd,adam}
  --nesterov            nesterov for SGD
  -mom MOMENTUM, --momentum MOMENTUM
                        the momentum used for SGD
  -beta1 BETA1, --beta1 BETA1
                        the first beta argument for Adam
  -beta2 BETA2, --beta2 BETA2
                        the second beta argument for Adam
  -wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        weight for 'l1|l2|...' regularzation
  -lr LR, --lr LR, --LR LR, --learning-rate LR
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  --epochs EPOCHS
  --eval-valid          evaluate validset
  --eval-test           evaluate testset
  --eval-freq EVAL_FREQ
                        the evaluation frequency
  --num-workers NUM_WORKERS
  --buffer-size BUFFER_SIZE
                        buffer size for datapipe
  --seed SEED           calling --seed=-1 for a random seed
  --benchmark           cudnn.benchmark == True ?
  --verbose             show the progress bar if true
  --resume              resume the training from the recent checkpoint
  --fmt FMT
  -m DESCRIPTION, --description DESCRIPTION
  -eb EMBEDDING_DIM, --embedding_dim EMBEDDING_DIM
  -neg NUM_NEGS, --num_negs NUM_NEGS
```

## Reference Code

- TorchRec: https://github.com/pytorch/torchrec 
- DeepCTR-Torch: https://github.com/shenweichen/DeepCTR-Torch
- FuxiCTR: https://github.com/xue-pai/FuxiCTR
- BARS: https://github.com/openbenchmark/BARS
