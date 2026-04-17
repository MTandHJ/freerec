# 实现方案：freerec 规范性改进

## Context

freerec 在自动化测试和代码检查方面与主流 Python 开源项目存在差距。本次改进引入 ruff（lint + format）和 pytest 测试套件，并通过 CI 集成形成质量门禁。

---

## 阶段一：Ruff 配置与首次格式化

### 步骤 1.1：修改 `pyproject.toml`

添加 ruff 配置和 dev 依赖：

```toml
[tool.ruff]
line-length = 88
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]

[tool.ruff.format]
quote-style = "double"

[project.optional-dependencies]
dev = [
    "ruff>=0.4.0",
    "pytest>=7.0",
    "pre-commit>=3.0",
]
```

### 步骤 1.2：新建 `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.8
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

### 步骤 1.3：修复拼写错误

- `freerec/launcher.py:243`：`get_reading_servie` → `get_reading_service`（及所有引用处 L255, L259, L263）
- `freerec/metrics.py:537`：`_single_adverage_precision` → `_single_average_precision`（及所有引用处）

### 步骤 1.4：首次 ruff 格式化

运行 `ruff check --fix .` 和 `ruff format .`，一个独立 commit 完成。

---

## 阶段二：pytest 测试套件

### 步骤 2.1：创建测试目录结构

```
tests/
├── conftest.py
├── test_metrics.py
├── test_criterions.py
├── test_parser.py
├── test_launcher.py
├── test_data/
│   ├── __init__.py
│   ├── test_fields.py
│   ├── test_normalizer.py
│   └── test_utils.py
├── test_datapipe.py
└── fixtures/
    └── Processed/
        └── Example_XXX_LOU/
            ├── train.txt      # ~50 行交互，10 用户 × 20 物品
            ├── valid.txt      # 10 行（每用户 1 条）
            └── test.txt       # 10 行（每用户 1 条）
```

### 步骤 2.2：`tests/conftest.py` — 共享 fixtures

- 提供常用 torch tensor fixtures（random preds/targets）
- 提供 toy 数据集 fixture：`root="tests/fixtures"`, `filedir="Example_XXX_LOU"`
- 提供临时目录 fixture（`tmp_path`）

### 步骤 2.3：`tests/test_metrics.py`（P0）

覆盖 `freerec/metrics.py` 中 13 个公开函数：

| 函数 | 测试要点 |
|------|----------|
| `mean_abs_error` | 已知输入输出对、reduction 模式（none/mean/sum） |
| `mean_squared_error` | 同上 |
| `root_mse` | 同上 |
| `precision` | k 参数、全 0/全 1 targets |
| `recall` | 同上 |
| `f1_score` | 同上 |
| `hit_rate` | 同上 |
| `normalized_dcg` | 理想排序 vs 随机排序 |
| `mean_reciprocal_rank` | 首个相关项在不同位置 |
| `mean_average_precision` | 同上 |
| `log_loss` | eps 参数、边界值 |
| `auroc` | 完美/随机预测 |
| `group_auroc` | 分组聚合 |

使用 `@pytest.mark.parametrize` 覆盖多组输入。

### 步骤 2.4：`tests/test_criterions.py`（P0）

覆盖 `freerec/criterions.py` 中 6 个损失类 + 4 个辅助函数：

| 类/函数 | 测试要点 |
|---------|----------|
| `BPRLoss` | 正分 > 负分时 loss 应较小、reduction 模式 |
| `BCELoss4Logits` | 与 `F.binary_cross_entropy_with_logits` 对齐 |
| `CrossEntropy4Logits` | 与 `F.cross_entropy` 对齐 |
| `KLDivLoss4Logits` | 与 `F.kl_div` 对齐 |
| `MSELoss` | 与 `F.mse_loss` 对齐 |
| `L1Loss` | 与 `F.l1_loss` 对齐 |
| `BaseCriterion.regularize` | L1/L2 正则化计算 |
| 辅助函数 | 与对应类输出一致 |

### 步骤 2.5：`tests/test_data/test_fields.py`（P1）

| 测试目标 | 要点 |
|----------|------|
| `Field` 创建 | 属性正确性、hash/equality |
| `Field.fork()` | 返回独立副本 |
| `Field.match/match_all/match_any` | tag 匹配逻辑 |
| `FieldTuple` | 索引、过滤、copy |
| `FieldModule` | torch.nn.Module 集成（需要 torch） |
| `FieldModuleList` | 类型检查、增删 |

### 步骤 2.6：`tests/test_data/test_normalizer.py`（P1）

| 测试目标 | 要点 |
|----------|------|
| `Counter` | partial_fit 累计计数、reset |
| `ReIndexer` | 映射连续整数、count 属性 |
| `StandardScaler` | mean/std 计算正确性 |
| `MinMaxScaler` | 缩放到 [0,1] |
| `Normalizer`（基类） | identity 变换 |

依赖 `polars.Series`，直接构造测试数据。

### 步骤 2.7：`tests/test_data/test_utils.py`（P1）

| 测试目标 | 要点 |
|----------|------|
| `safe_cast` | 正常转换、失败回退 |
| `check_sha1` | 字符串和字节输入 |
| `is_empty_dir` | 空/非空目录 |
| `negsamp_vectorized_bsearch` | 采样结果不含正例、size 正确 |

`download_from_url` 和 `extract_archive` 涉及网络/文件 I/O，本轮暂不覆盖。

### 步骤 2.8：`tests/test_parser.py`（P1）

| 测试目标 | 要点 |
|----------|------|
| `Parser` 默认值 | 默认 CONFIG 正确加载 |
| `add_argument` | 新增参数可用 |
| `set_defaults` | 覆盖默认值 |
| `CoreParser` | CORE_CONFIG 加载、check() 验证 |

Parser 依赖 argparse，测试时通过 `parse_args([])` 模拟。

### 步骤 2.9：`tests/test_launcher.py`（P2）

| 测试目标 | 要点 |
|----------|------|
| `AverageMeter` | update/step/reset/history、reduction 模式 |
| `Monitor` | state_dict/load_state_dict |
| `set_seed` | 返回值正确 |
| `export/import_pickle` | 序列化往返 |
| `export/import_yaml` | 序列化往返 |

注：`AverageMeter` 和 `Monitor` 定义在 `utils.py`，但逻辑上与 launcher 训练循环紧密相关。Coach 类的完整训练循环因依赖复杂暂不覆盖。

### 步骤 2.10：`tests/fixtures/` — Toy 数据集

构造 `Example_XXX_LOU` 数据集（10 用户、20 物品），TSV 格式，列为 `USER ITEM RATING TIMESTAMP`。

- `train.txt`：约 50 行交互，每用户 3-8 条历史
- `valid.txt`：10 行，每用户 1 条
- `test.txt`：10 行，每用户 1 条
- 路径：`tests/fixtures/Processed/Example_XXX_LOU/`
- 加载方式：`NextItemRecDataSet(root="tests/fixtures", filedir="Example_XXX_LOU")`

`schema.pkl` 和 `chunks/` 由首次加载自动生成，加入 `.gitignore`。

### 步骤 2.11：`tests/test_datapipe.py`（P1）

基于 toy 数据集，测试 examples 中常用的完整 datapipe 链路：

| 链路类型 | 覆盖的 datapipe | 对应 examples |
|----------|----------------|---------------|
| **Gen 训练链路** | `choiced_user_ids_source → gen_train_sampling_pos_ → gen_train_sampling_neg_ → batch_ → tensor_` | MF-BPR, LightGCN |
| **Seq 训练链路** | `shuffled_seqs_source → seq_train_yielding_pos_ → seq_train_sampling_neg_ → add_ → lpad_ → batch_ → tensor_` | SASRec, GRU4Rec |
| **评估链路** | `ordered_user_ids_source → valid_sampling_ → lprune_ → add_ → rpad_ → batch_ → tensor_` | 通用 |

每条链路验证：
- 迭代不报错
- 输出为 `Dict[Field, Tensor]`
- batch 维度正确
- 正负采样结果不含交叉

---

## 阶段三：CI 集成

### 步骤 3.1：新建 `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [master]
    paths: ['freerec/**', 'tests/**', 'pyproject.toml', '.github/workflows/ci.yml']
  pull_request:
    branches: [master]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ruff
      - run: ruff check .
      - run: ruff format --check .

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.11']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -e ".[dev,metrics]"
      - run: pytest tests/ -v
```

---

## 验证方式

1. `ruff check .` 和 `ruff format --check .` 通过（零违规）
2. `pytest tests/ -v` 全部通过
3. GitHub Actions CI workflow 在 push 后自动运行并通过
4. `pre-commit run --all-files` 通过

---

## 变更文件汇总

| 操作 | 文件 |
|------|------|
| 修改 | `pyproject.toml` |
| 修改 | `freerec/launcher.py`（拼写修复） |
| 修改 | `freerec/metrics.py`（拼写修复） |
| 修改 | `freerec/**`（ruff 首次格式化） |
| 新增 | `.pre-commit-config.yaml` |
| 新增 | `.github/workflows/ci.yml` |
| 新增 | `tests/conftest.py` |
| 新增 | `tests/test_metrics.py` |
| 新增 | `tests/test_criterions.py` |
| 新增 | `tests/test_parser.py` |
| 新增 | `tests/test_launcher.py` |
| 新增 | `tests/test_data/__init__.py` |
| 新增 | `tests/test_data/test_fields.py` |
| 新增 | `tests/test_data/test_normalizer.py` |
| 新增 | `tests/test_data/test_utils.py` |
| 新增 | `tests/test_datapipe.py` |
| 新增 | `tests/fixtures/Processed/Example_XXX_LOU/train.txt` |
| 新增 | `tests/fixtures/Processed/Example_XXX_LOU/valid.txt` |
| 新增 | `tests/fixtures/Processed/Example_XXX_LOU/test.txt` |
