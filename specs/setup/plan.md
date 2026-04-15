# Plan: FreeRec 安装体系重构

## Context

FreeRec 当前使用 `setup.py` 管理安装，依赖声明不全（14 个第三方包仅声明 4 个），且 torchdata==0.7.0 会覆盖已有 torch。需迁移到 `pyproject.toml` + `freerec setup` CLI，详细 spec 见 `specs/setup/spec.md`。

## 任务清单

### Task 1: 新建 `pyproject.toml`

从 `setup.py` 迁移所有元数据到 `pyproject.toml`：
- 构建后端：hatchling
- 补全 9 个必选依赖 + 3 个可选分组（`[graph]`, `[metrics]`, `[nn]`, `[all]`）
- entry_points → `[project.scripts]`
- package_data → `[tool.hatch.build]` include `.pyi` 文件
- 版本号直接写在 `version` 字段

完整 toml 内容见 spec 第 1 节。

### Task 2: 修改 `freerec/__init__.py`

版本号从硬编码改为动态读取：

```python
# 旧
__version__ = '0.9.7'

# 新
from importlib.metadata import version
__version__ = version("freerec")
```

注意：仅改这一行，其余 import 和 `declare()` 函数不动。

### Task 3: 修改 `freerec/__main__.py` — 新增 `setup` 子命令

在 `main()` 函数中新增 `setup` subparser，实现 `setup()` 函数：

```python
def setup(args):
    """Install torchdata==0.7.0 without dependencies."""
    import subprocess, sys

    # 1. 检测 torch
    try:
        import torch
        print(f"torch {torch.__version__} detected, skipping torch installation.")
    except ImportError:
        print("Error: torch is not installed.")
        print("Please install torch first: https://pytorch.org/get-started/locally/")
        sys.exit(1)

    # 2. 检测 torchdata
    try:
        import torchdata
        if torchdata.__version__ == "0.7.0":
            print("torchdata 0.7.0 already installed, skipping.")
            return
        else:
            print(f"torchdata {torchdata.__version__} found, upgrading to 0.7.0...")
    except ImportError:
        print("Installing torchdata==0.7.0 (--no-deps)...")

    # 3. 安装
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torchdata==0.7.0", "--no-deps"
    ])

    # 4. 验证
    ...
```

subparser 注册（在 `main()` 的 `make_parser` 之后）：

```python
setup_parser = subparsers.add_parser("setup",
    help="Install torchdata==0.7.0 without overriding existing torch")
setup_parser.set_defaults(func=setup)
```

无需额外参数。

### Task 4: 更新 `README.md` 安装说明

当前 README 的 Requirements 和 Installation 部分内容已过时，需同步更新：

- **Requirements 部分**：去掉手动 `pip install --no-deps torchdata==0.7.0` 的说明，改为使用 `freerec setup`
- **Installation 部分**：更新为三步流程：1) 安装 torch → 2) pip install freerec → 3) freerec setup
- **NOTE 块**：保留 torchdata 版本限制说明，但更新安装方式描述
- 其余部分（Data Pipeline、Training Flow 等）不变

### Task 5: 删除 `setup.py`

确认 pyproject.toml、`__init__.py`、`__main__.py`、README 修改完成后，删除 `setup.py`。

### Task 6: 验证

1. `pip install -e .` — 确认 hatchling 构建正常
2. `python -c "import freerec; print(freerec.__version__)"` — 确认版本号正确
3. `freerec setup` — 确认 CLI 子命令可用（检测 torch 和 torchdata）
4. `freerec make --help` / `freerec tune --help` / `freerec skill --help` — 确认原有子命令不受影响

## 文件变更

| 操作 | 文件 |
|------|------|
| 新建 | `pyproject.toml` |
| 修改 | `freerec/__init__.py` |
| 修改 | `freerec/__main__.py` |
| 修改 | `README.md` |
| 删除 | `setup.py` |
