
# Log 输出格式 — 实现方案

## 概述

在 `freerec tune` 完成后自动输出 `results.json` 到 `logs/{description}/core/`，包含所有子实验的参数和指标，同步更新相关文档和注释。

## 变更清单

### 1. 核心实现：`freerec/launcher.py`

| 变更 | 位置 | 说明 |
|------|------|------|
| 新增 `import json` | 文件顶部 | 标准库导入 |
| 修改 `write()` | ~line 1071 | TensorBoard 写入后直接追加写入 `results.json` |
| 更新 docstrings | `Adapter` 类、`write()` | 补充 JSON 输出说明 |

#### `write()` 中追加的 JSON 写入逻辑

```python
# 在 write() 的 try 块末尾，TensorBoard 写入之后追加：
path = os.path.join(self.cfg.CORE_LOG_PATH, self.cfg.RESULTS_FILENAME)
if os.path.exists(path):
    with open(path, "r") as f:
        results = json.load(f)
else:
    results = {
        "description": self.cfg.ENVS.get("description", ""),
        "dataset": self.cfg.ENVS.get("dataset", ""),
        "runs": [],
    }
results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
results["runs"].append({"id": id_, "params": params, "metrics": data})
with open(path, "w") as f:
    json.dump(results, f, indent=2)
```

每个子实验完成时立即写入，无需额外的累积逻辑。

### 2. 常量定义：`freerec/parser.py`

| 变更 | 位置 | 说明 |
|------|------|------|
| 新增 `RESULTS_FILENAME` | CONFIG 定义区（~line 92） | `"results.json"` |
| 更新模块级 docstring | ~line 35 | `CORE_LOG_PATH` 描述补充 `results.json` |
| 更新 CONFIG docstring | ~line 54 | 新增 `RESULTS_FILENAME` 说明 |

### 3. 内嵌教程：`freerec/skills.py`

| 变更 | 位置 | 说明 |
|------|------|------|
| 更新目录树 | `tune` skill（~line 221-240） | `core/` 下补充 `results.json` |
| 新增 JSON 格式说明 | `tune` skill（~line 244-275） | 在 `best.pkl` 说明后补充 `results.json` 的结构 |
| 更新目录树 | `log` skill（~line 330-346） | 补充 `core/results.json` |
| 新增 JSON 格式说明 | `log` skill（~line 383-450） | 文件格式部分补充 `results.json` |
| 更新输出说明 | `workflow` skill（~line 730-745） | LOGGING LAYER 补充 `results.json` |

### 4. 用户文档：`docs/tutorials/output.rst`

| 变更 | 位置 | 说明 |
|------|------|------|
| 更新目录树 | ~line 1-24 | 补充 `core/results.json` |
| 新增章节 | ~line 138 之后 | `results.json` 的字段说明和示例 |

### 5. 用户文档：`docs/tutorials/training_and_tuning.rst`

| 变更 | 位置 | 说明 |
|------|------|------|
| 补充 tune 输出 | ~line 80-102 | 说明 tune 完成后生成 `results.json` |

## 验证方式（由用户执行）

1. 选一个模型（如 SASRec），用小配置跑 `freerec tune`（2 个种子）
2. 确认 `logs/{desc}/core/results.json` 生成，格式符合 spec
3. 再次运行 tune，确认追加模式正常
4. 确认 TensorBoard 输出不受影响
