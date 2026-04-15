# FreeRec 安装体系重构

## 背景

当前使用 `setup.py` 管理安装，存在以下问题：
- 依赖声明不全：实际使用 14 个第三方包，仅声明 4 个
- torchdata==0.7.0 安装时会强制拉取它要求的 torch 版本，覆盖用户已有环境
- 必须通过 `pip install --no-deps torchdata==0.7.0` 避免覆盖，但 pyproject.toml 依赖声明无法表达 `--no-deps`

## 目标

简化安装流程：

```bash
pip install torch              # 用户自行安装（已有则跳过）
pip install freerec             # 安装 freerec + 所有非 torch 依赖
freerec setup                   # 一键安装 torchdata==0.7.0（内部 --no-deps）
```

freerec 不关心 CUDA 版本，不关心 torch 版本，这些完全由用户环境决定。
`freerec setup` 的唯一职责是安全安装 torchdata 而不破坏已有的 torch。

## 决策记录

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 构建后端 | hatchling | 现代主流，PyPA 推荐 |
| 版本管理 | pyproject.toml 单一来源 | 代码中通过 `importlib.metadata.version()` 读取 |
| 旧 setup.py | 删除 | 完全迁移，不保留 |
| torch | 用户自行安装 | freerec 不管 CUDA/版本，这是用户环境的事 |
| torchdata | `freerec setup` 安装 | 必须 `--no-deps`，pyproject.toml 无法表达 |
| torchdata 版本 | 固定 0.7.0 | 后续版本不可用 |
| numpy 版本 | `>=1.24` 无上限 | torch 安装时已锁定兼容 numpy；如有 2.x 问题在代码层面修复 |
| CLI 框架 | 保持 argparse | 不引入新依赖 |
| 可选依赖 | 按功能分组 extras | `[graph]`, `[metrics]`, `[nn]`, `[all]` |

## 1. pyproject.toml 迁移

### 构建系统

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 项目元数据

```toml
[project]
name = "freerec"
version = "0.9.7"
description = "PyTorch library for recommender systems"
readme = "README.md"
license = "MIT"
requires-python = ">=3.9"
authors = [
    { name = "MTandHJ", email = "congxueric@gmail.com" },
]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

dependencies = [
    "numpy>=1.24",
    "polars>=1.9.0",
    "PyYAML>=6.0",
    "tensorboard>=2.10.0",
    "prettytable>=3.4.1",
    "matplotlib>=3.5.0",
    "requests>=2.28.0",
    "tqdm>=4.64.0",
    "pandas>=1.5.0",
]

[project.optional-dependencies]
graph = ["torch-geometric>=2.4.0"]
metrics = ["scikit-learn>=1.0.0"]
nn = ["einops>=0.6.0"]
all = [
    "freerec[graph]",
    "freerec[metrics]",
    "freerec[nn]",
]

[project.scripts]
freerec = "freerec.__main__:main"

[project.urls]
Homepage = "https://github.com/MTandHJ/freerec"
```

### 包数据

```toml
[tool.hatch.build.targets.wheel]
packages = ["freerec"]

[tool.hatch.build]
include = [
    "freerec/**/*.py",
    "freerec/**/*.pyi",
]
```

### 版本号

- **单一来源**：版本号仅在 `pyproject.toml` 的 `version` 字段定义
- **代码读取**：`__init__.py` 中改为：
  ```python
  from importlib.metadata import version
  __version__ = version("freerec")
  ```
- 删除原 `__version__ = '0.9.7'` 硬编码

## 2. 依赖分类

### 必选依赖（dependencies）

| 包 | 版本约束 | 用途 | 使用文件 |
|---|---------|------|---------|
| numpy | >=1.24 | 数值计算基础 | 8 个文件 |
| polars | >=1.9.0 | DataFrame 操作 | fields, normalizer, datasets |
| PyYAML | >=6.0 | 配置文件解析 | utils, datasets |
| tensorboard | >=2.10.0 | 训练可视化 | launcher, utils |
| prettytable | >=3.4.1 | 表格格式化输出 | datasets, preprocessing |
| matplotlib | >=3.5.0 | 指标曲线绘图 | utils (AverageMeter.plot) |
| requests | >=2.28.0 | 数据集下载 | data/utils |
| tqdm | >=4.64.0 | 下载进度条 | data/utils |
| pandas | >=1.5.0 | 数据预处理 | preprocessing |

### 可选依赖（optional-dependencies）

| 分组 | 包 | 版本约束 | 用途 |
|-----|---|---------|------|
| `[graph]` | torch-geometric | >=2.4.0 | 图神经网络工具 (graph.py) |
| `[metrics]` | scikit-learn | >=1.0.0 | ROC-AUC 等评估指标 (metrics.py) |
| `[nn]` | einops | >=0.6.0 | 注意力机制张量操作 (models/nn/attn.py) |
| `[all]` | 以上全部 | — | 完整安装 |

### 通过 CLI 管理（不放入 pyproject.toml）

| 包 | 版本 | 原因 |
|---|------|------|
| torch | 用户自行安装 | CUDA 变体由用户环境决定，freerec 不干预 |
| torchdata | ==0.7.0 | 必须 `--no-deps` 安装以避免覆盖 torch |

## 3. `freerec setup` CLI 子命令

### 用法

```bash
freerec setup
```

无需任何参数。

### 行为逻辑

1. **检测 torch**：
   - 如果 torch 未安装 → 报错，提示用户先安装 torch
   - 如果已安装 → 打印版本信息，继续
2. **安装 torchdata**：
   - 如果 torchdata 已安装且版本为 0.7.0 → 跳过，打印提示
   - 否则 → 执行 `pip install torchdata==0.7.0 --no-deps`
3. **验证**：
   - 安装完成后尝试 `import torchdata`，确认成功

### 实现位置

在 `freerec/__main__.py` 中新增 `setup` 子命令，通过 `subprocess` 调用 pip。

## 4. 文件变更清单

| 操作 | 文件 |
|------|------|
| 新建 | `pyproject.toml` |
| 删除 | `setup.py` |
| 修改 | `freerec/__init__.py`（版本号改为 importlib.metadata） |
| 修改 | `freerec/__main__.py`（新增 setup 子命令） |
