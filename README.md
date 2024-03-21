

![](docs/src/logo.png)

FreeRec is a repository designed for easy (recommendation) data pre-processing and model training.
I am a beginner in the field of recommender systems, so much of FreeRec's designs may not be as effective. In addition, you are free to specify your own framework based on FreeRec.


## Requirements: 

Python == 3.9 | [PyTorch == 1.12.1](https://pytorch.org/) | [TorchData == 0.4.1](https://github.com/pytorch/data) | [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) (optional)


```
conda create --name=PyT12 python=3.9
conda activate PyT12
```

### GPU

```
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torchdata==0.4.1
```

- Linux

```
pip install torch_geometric==2.1.0.post1
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_sparse-0.6.15%2Bpt112cu116-cp39-cp39-linux_x86_64.whl
```

- Windows

```
pip install torch_geometric==2.1.0.post1
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_scatter-2.0.9-cp39-cp39-win_amd64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_sparse-0.6.15%2Bpt112cu116-cp39-cp39-win_amd64.whl
```

### CPU


```
conda install pytorch==1.12.1 cpuonly -c pytorch
pip install torchdata==0.4.1
```

- Linux

```
pip install torch_geometric==2.1.0.post1
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/torch_sparse-0.6.15%2Bpt112cpu-cp39-cp39-linux_x86_64.whl
```


- Windows

```
pip install torch_geometric==2.1.0.post1
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/torch_scatter-2.0.9-cp39-cp39-win_amd64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcpu/torch_sparse-0.6.15%2Bpt112cpu-cp39-cp39-win_amd64.whl
```



## Installation

    pip install freerec

or (for latest)

    pip install git+https://github.com/MTandHJ/freerec.git



## Data Pipeline

![](docs/src/pipeline.png)

**Note:** To make dataset, please download corresponding Atomic files from [[RecBole](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj)]. 
Then, run `make_dataset.ipynb'.

## Training Flow


![](docs/src/flow.png)


## Reference Code

- TorchRec: https://github.com/pytorch/torchrec 
- DeepCTR-Torch: https://github.com/shenweichen/DeepCTR-Torch
- FuxiCTR: https://github.com/xue-pai/FuxiCTR
- BARS: https://github.com/openbenchmark/BARS
- RecBole: https://github.com/RUCAIBox/RecBole



## Acknowledgements

Thanks to ChatGPT for the annotation of some code. For this reason, some of the comments may be illogical.
