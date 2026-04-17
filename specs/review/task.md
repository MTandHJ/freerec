# Task List：freerec 规范性改进

## 阶段一：Ruff 配置与首次格式化

- [x] 1.1 修改 `pyproject.toml`：添加 ruff 配置 + dev 依赖
- [x] 1.2 新建 `.pre-commit-config.yaml`
- [x] 1.3 修复拼写错误（launcher.py `servie`→`service`，metrics.py `adverage`→`average`）
- [x] 1.4 首次 ruff 格式化（`ruff check --fix .` + `ruff format .`）

## 阶段二：pytest 测试套件

- [x] 2.1 创建 toy 数据集 `tests/fixtures/Processed/Example_XXX_LOU/`
- [x] 2.2 创建 `tests/conftest.py`
- [x] 2.3 创建 `tests/test_metrics.py`（P0）
- [x] 2.4 创建 `tests/test_criterions.py`（P0）
- [x] 2.5 创建 `tests/test_data/test_fields.py`（P1）
- [x] 2.6 创建 `tests/test_data/test_normalizer.py`（P1）
- [x] 2.7 创建 `tests/test_data/test_utils.py`（P1）
- [x] 2.8 创建 `tests/test_parser.py`（P1）
- [x] 2.9 创建 `tests/test_launcher.py`（P2）
- [x] 2.10 创建 `tests/test_datapipe.py`（P1）

## 阶段三：CI 集成

- [x] 3.1 新建 `.github/workflows/ci.yml`

## 验证

- [x] 4.1 `ruff check .` + `ruff format --check .` 通过
- [x] 4.2 `pytest tests/ -v` 全部通过（140 tests passed）
