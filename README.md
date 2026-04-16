

![](docs/src/logo.png)

<p align="center">
    <a href="https://pypi.org/project/freerec/"><img src="https://img.shields.io/pypi/v/freerec?color=blue" alt="PyPI"></a>
    <a href="https://pypi.org/project/freerec/"><img src="https://img.shields.io/pypi/pyversions/freerec" alt="Python"></a>
    <a href="https://github.com/MTandHJ/freerec/blob/master/LICENSE"><img src="https://img.shields.io/github/license/MTandHJ/freerec" alt="License"></a>
</p>

<p align="center">
    <a href="https://mtandhj.github.io/freerec/">Documentation</a> &bull;
    <a href="https://github.com/MTandHJ/RecBoard">RecBoard</a>
</p>

---

**FreeRec** is a PyTorch-based library for building, training, and evaluating recommendation systems. It provides modular components — data pipelines, model architectures, training loops, and evaluation metrics — so you can focus on your research instead of boilerplate.

## Key Features

- **Flexible data pipeline** built on `torchdata`, supporting field-level processing, k-core filtering, and multiple splitting strategies
- **Modular model architecture** with base classes for collaborative filtering, sequential recommendation, and CTR prediction
- **Built-in training loop** with distributed training (DDP), TensorBoard logging, and grid-search hyperparameter tuning
- **Standard evaluation metrics** including NDCG, Hit Rate, MRR, Recall, Precision, and AUC

## Installation

**Requirements:** Python >= 3.9 | [PyTorch >= 2.0](https://pytorch.org/)

```bash
# 1. Install PyTorch (choose the CUDA version that matches your environment)
#    See https://pytorch.org/get-started/locally/
pip install torch

# 2. Install FreeRec
pip install freerec

# 3. Install torchdata (handled automatically with --no-deps)
freerec setup
```

> [!NOTE]
> FreeRec relies on torchdata 0.7.0, as later releases no longer support the datapipe functionality.
> `freerec setup` installs it with `--no-deps` to avoid overriding your existing PyTorch installation.

### Optional dependencies

```bash
pip install freerec[graph]    # torch-geometric for graph-based models
pip install freerec[metrics]  # scikit-learn for additional metrics (e.g., ROC-AUC)
pip install freerec[nn]       # einops for attention modules
pip install freerec[all]      # all of the above
```

## Overview

### Data Pipeline

![](docs/src/pipeline.png)

### Training Flow

![](docs/src/flow.png)

For detailed usage, please refer to the [documentation](https://mtandhj.github.io/freerec/).

## References

- [TorchRec](https://github.com/pytorch/torchrec)
- [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch)
- [FuxiCTR](https://github.com/xue-pai/FuxiCTR)
- [BARS](https://github.com/openbenchmark/BARS)
- [RecBole](https://github.com/RUCAIBox/RecBole)

## Acknowledgements

- Code annotation and documentation were written with the assistance of [Claude Code (Opus 4.6)](https://claude.ai/claude-code) by Anthropic.
