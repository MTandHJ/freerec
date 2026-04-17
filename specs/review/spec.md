# freerec 规范性审查 Spec

## 背景

freerec 是一个基于 PyTorch 的推荐系统研究库。当前项目在构建系统（Hatchling）、文档（Sphinx + 双语）、类型标注等方面已达到主流标准，但在**自动化测试**和**代码检查/格式化**两个维度存在明显差距。

本次改进目标：补齐这两项短板，并通过 CI 集成形成自动化质量门禁。

---

## 一、引入 Ruff（代码检查 + 格式化）

### 1.1 规则配置

在 `pyproject.toml` 中添加 ruff 配置：

- **规则集**：保守方案——`E`（pycodestyle errors）、`F`（pyflakes）、`W`（pycodestyle warnings）、`I`（isort）
- **行宽**：88（与 black 一致）
- **目标版本**：`py39`（与项目最低 Python 版本对齐）
- **引号风格**：双引号（double）
- **命名规则**：**不启用 N 规则集**，不改变现有命名风格（如 `self.User`, `self.Item` 等字段属性保持 PascalCase）
- **拼写错误**：可修复（如 `get_reading_servie()` → `get_reading_service()`）

### 1.2 开发依赖

在 `pyproject.toml` 的 `[project.optional-dependencies]` 中添加 `dev` 依赖组：

```
ruff >= 0.4.0
pre-commit >= 3.0
```

### 1.3 Pre-commit Hook

新增 `.pre-commit-config.yaml`，集成 ruff 的 lint 和 format hook。

### 1.4 首次格式化策略

- 首次运行 `ruff format` + `ruff check --fix` 会产生批量 diff（import 排序、空行、尾部逗号、引号统一等）
- 用**一个独立 commit** 完成首次格式化，与功能改动分离

---

## 二、引入自动化测试（pytest）

### 2.1 测试框架

- **框架**：pytest
- **依赖**：`pytest >= 7.0`，加入 `dev` 可选依赖组

### 2.2 测试范围

按优先级覆盖以下模块：

| 优先级 | 模块 | 测试重点 |
|--------|------|----------|
| P0 | `metrics` | 各评估指标的计算正确性、边界条件、与 sklearn 对齐 |
| P0 | `criterions` | 各损失函数的输出形状、梯度、reduction 模式 |
| P1 | `data` | fields 的创建/转换、preprocessing 策略、normalizer |
| P1 | `parser` | 配置解析、默认值、类型转换 |
| P2 | `launcher` | Coach 的生命周期、回调机制（可能需要 mock） |

### 2.3 测试目录结构

```
tests/
├── conftest.py           # 共享 fixtures
├── test_metrics.py
├── test_criterions.py
├── test_data/
│   ├── test_fields.py
│   ├── test_preprocessing.py
│   └── test_normalizer.py
├── test_parser.py
└── test_launcher.py
```

### 2.4 测试规范

- 使用 `pytest.mark.parametrize` 覆盖多组输入
- 需要 GPU 的测试用 `pytest.mark.skipif` 标记
- 测试文件命名：`test_<模块名>.py`
- 测试函数命名：`test_<功能>_<场景>`

---

## 三、CI 集成（GitHub Actions）

### 3.1 新增 Workflow

在 `.github/workflows/` 下新增 CI workflow（与现有 `docs.yml` 并行），包含以下 jobs：

#### Job 1: Lint

- 运行 `ruff check .`
- 运行 `ruff format --check .`
- 触发条件：push / PR 到 master

#### Job 2: Test

- Python 版本矩阵：3.9, 3.11（最低支持版本 + 最新稳定版）
- 安装项目依赖 + dev 依赖
- 运行 `pytest tests/`
- 触发条件：push / PR 到 master

### 3.2 触发条件

- 当 `freerec/`、`tests/`、`pyproject.toml`、workflow 文件发生变更时触发
- PR 合入 master 前必须通过 lint + test

---

## 四、不在本次范围内

以下事项经评估暂不纳入：

- **类型检查（mypy/pyright）**：必要性中等，后续可独立引入
- **命名规范强制**：现有命名 95% 符合 PEP 8，不单独立项
- **CONTRIBUTING.md**：当前以个人维护为主，优先级低
- **代码覆盖率报告**：随测试成熟度提升后再引入

---

## 五、变更文件清单

| 操作 | 文件 |
|------|------|
| 修改 | `pyproject.toml`（ruff 配置 + dev 依赖 + pytest 依赖） |
| 新增 | `.pre-commit-config.yaml` |
| 新增 | `.github/workflows/ci.yml` |
| 新增 | `tests/` 目录及测试文件 |
| 修改 | 现有代码（ruff 首次格式化 + 拼写修复） |
