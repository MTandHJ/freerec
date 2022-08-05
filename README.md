



## Requirements: 

Python==3.9 | [PyTorch >= 1.12.0](https://pytorch.org/) | [TorchData](https://github.com/pytorch/data) | [TorchRec](https://github.com/pytorch/torchrec)


## Installation

    pip install freerec


## Help


```
positional arguments:
  root                  data

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       .yml
  --optimizer {sgd,adam}
  --nesterov            nesterov for SGD
  -mom MOMENTUM, --momentum MOMENTUM
                        the momentum used for SGD
  -beta1 BETA1, --beta1 BETA1
                        the first beta argument for Adam
  -beta2 BETA2, --beta2 BETA2
                        the second beta argument for Adam
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        weight decay
  -lr LR, --lr LR, --LR LR, --learning_rate LR
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  --epochs EPOCHS
  --eval-train          evaluate trainloader
  --eval-valid          evaluate validloader
  --eval-freq EVAL_FREQ
                        the evaluation frequency for valid dataset only
  --num-workers NUM_WORKERS
  --buffer-size BUFFER_SIZE
                        buffer size for datapipe
  --seed SEED           calling --seed=-1 for a random seed
  --benchmark           cudnn.benchmark == True ?
  --progress            show the progress if true
  --resume              resume the training from the recent checkpoint
  --fmt FMT
  -m DESCRIPTION, --description DESCRIPTION
```

## Reference Code

- TorchRec: https://github.com/pytorch/torchrec 
- DeepCTR-Torch: https://github.com/shenweichen/DeepCTR-Torch
- FuxiCTR: https://github.com/xue-pai/FuxiCTR
- BARS: https://github.com/openbenchmark/BARS
