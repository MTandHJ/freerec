

## Training and Tuning

### Training

Most of methods implemented based on freerec can be conducted by

```
python main.py --config=XXX.yaml
```
where `XXX.yaml` is the configuration file. For example, it could be

```yaml
root: ../../data
dataset: Amazon2014Beauty_550_LOU

maxlen: 50
num_heads: 1
num_blocks: 2
embedding_dim: 64
dropout_rate: 0.4

epochs: 300
batch_size: 512
optimizer: adam
lr: 5.e-4
weight_decay: 1.e-8

monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
which4best: NDCG@10
```


### Tuning

You can tune some hyper-parameters by using `freerec tune`:

```
freerec tune [Experiment name] config.yaml
```
where `config.yaml` looks like

```yaml
command: python main.py
envs:
  root: ../../data
  dataset: Amazon2014Beauty_550_LOU
  device: '0,1,2,3'
params:
  seed: [0, 1, 2, 3, 4]
defaults:
  config: configs/Amazon2014Beauty_550_LOU.yaml
```

- `command`: the command to execute a single run
- `envs`:
    - `root`
    - `dataset`
    - `device`: depends on your available devices
- `params`: the hyper-parameter to be searched.
- `defaults`: default settings can be placed here.