
# Log 输出格式 Spec

## 背景

目前 freerec 主要依赖以下目录结构保存实验结果：

```
./logs/{description}/{dataset}/{id}/
├── README.md                   # 配置快照
├── log.txt                     # 完整训练日志
├── model.pt                    # 最终模型权重
├── best.pt                     # 最佳模型权重
├── data/
│   ├── monitors.pkl            # 所有指标的历史记录
│   └── best.pkl                # 最佳模型的测试结果
└── summary/
    ├── SUMMARY.md              # 指标汇总表
    ├── validNDCG@10.png        # 验证集 NDCG@10 曲线
    ├── validHitRate@10.png     # 验证集 HitRate@10 曲线
    ├── trainLOSS.png           # 训练损失曲线
    └── (TensorBoard 事件文件)

./infos/{description}/{dataset}/{device}/
└── checkpoint.tar              # 训练断点（仅用于 --resume 恢复）
```

`freerec tune` 会通过网格搜索跑多组实验（调参或多种子），结果通过 TensorBoard `add_hparams` 写入 `logs/{description}/core/`。

### 痛点

1. `best.pkl` 只存单次实验结果，没有跨种子/跨配置的结构化聚合
2. 结果分散在各 `logs/{desc}/{dataset}/{id}/` 下，缺少统一索引
3. 缺少机器可读的结构化格式（pkl 不便于 web 前端消费）
4. 用户目前需手动从 TensorBoard 取多种子结果、求平均、填入 Notion leaderboard

## 目标

为后续 leaderboard 实现提供结构化的输出格式，使得：
- 每次 `freerec tune` 完成后自动产出 JSON 格式的聚合结果
- 包含所有子实验的完整配置和指标，便于下游过滤和聚合
- 支持追加模式，多次 tune 的结果累积到同一文件

## 设计决策

| 问题 | 决策 | 原因 |
|------|------|------|
| 数据来源 | 从 `best.pkl` 提取 | 已有完整指标，无需重复计算 |
| 输出位置 | `logs/{description}/core/results.json` | 复用现有 core/ 目录 |
| 输出格式 | JSON | 结构化、web 友好、易解析 |
| 聚合粒度 | 保留所有子实验的完整指标 | 便于下游按需聚合（mean±std 等） |
| 区分调参/跑种子 | 不区分，统一记录 | 由下游 leaderboard 负责过滤 |
| 多次 tune 冲突 | 追加模式 | 不同调试用不同 description，理论不冲突 |
| 配置标识 | 记录 description + dataset + 变化参数 | 便于按模型×数据集聚合 |

## JSON Schema

输出路径：`logs/{description}/core/results.json`

```json
{
  "description": "SASRec",
  "dataset": "Amazon2014Beauty_550_LOU",
  "timestamp": "2026-04-17T10:30:00",
  "runs": [
    {
      "id": "0417103000",
      "params": {"seed": 0, "lr": 5e-4, "embedding_dim": 64},
      "metrics": {
        "train": {"LOSS": 0.123},
        "valid": {"NDCG@10": 0.456, "HitRate@10": 0.789},
        "test":  {"NDCG@10": 0.432, "HitRate@10": 0.765},
        "best":  {"NDCG@10": 0.445, "HitRate@10": 0.778}
      }
    },
    {
      "id": "0417103500",
      "params": {"seed": 1, "lr": 5e-4, "embedding_dim": 64},
      "metrics": {
        "train": {"LOSS": 0.119},
        "valid": {"NDCG@10": 0.461, "HitRate@10": 0.793},
        "test":  {"NDCG@10": 0.438, "HitRate@10": 0.770},
        "best":  {"NDCG@10": 0.450, "HitRate@10": 0.782}
      }
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `description` | string | 实验描述，对应 `{description}` 路径变量（如 "SASRec"） |
| `dataset` | string | 数据集名称，对应 `{dataset}` 路径变量 |
| `timestamp` | string | tune session 的启动时间（ISO 8601 格式） |
| `runs` | array | 本次 tune 的所有子实验 |
| `runs[].id` | string | 子实验 ID（时间戳格式 MMDDHHMMSS） |
| `runs[].params` | object | 本次实验的参数（包含 seed 和所有网格搜索变量） |
| `runs[].metrics` | object | 各模式下的最佳指标 |
| `runs[].metrics.train` | object | 训练集指标 |
| `runs[].metrics.valid` | object | 验证集指标 |
| `runs[].metrics.test` | object | 测试集指标 |
| `runs[].metrics.best` | object | 最佳验证 epoch 对应的测试集指标 |

### 追加行为

- 文件不存在时：创建新文件，写入第一个 tune session 的结果
- 文件已存在时：读取现有内容，将新 runs 追加到 `runs` 数组，更新 `timestamp`
- `description` 和 `dataset` 保持不变（同一 core/ 下应一致）

## 连带变更：文档与注释更新

新增 `results.json` 输出后，以下文档和注释需同步更新：

### 目录树更新

`logs/{description}/core/` 下新增 `results.json`，完整结构变为：

```
./logs/{description}/{dataset}/{id}/
├── README.md                   # 配置快照
├── log.txt                     # 完整训练日志
├── model.pt                    # 最终模型权重
├── best.pt                     # 最佳模型权重
├── data/
│   ├── monitors.pkl            # 所有指标的历史记录
│   └── best.pkl                # 最佳模型的测试结果
└── summary/
    ├── SUMMARY.md              # 指标汇总表
    ├── *.png                   # 指标曲线图
    └── (TensorBoard 事件文件)

./logs/{description}/core/
├── results.json                # tune 聚合结果（新增）
└── {id}/                       # TensorBoard hparams 事件文件（已有）

./infos/{description}/{dataset}/{device}/
└── checkpoint.tar              # 训练断点（仅用于 --resume 恢复）
```

### 代码注释更新清单

| 文件 | 位置 | 更新内容 |
|------|------|----------|
| `freerec/parser.py` | 模块级 docstring（line ~35） | `CORE_LOG_PATH` 描述补充 `results.json` |
| `freerec/parser.py` | CONFIG 定义区（line ~92） | 新增 `RESULTS_FILENAME = "results.json"` 常量及注释 |
| `freerec/launcher.py` | `Adapter` 类 docstring（line ~963） | 补充 JSON 输出的说明 |
| `freerec/launcher.py` | `Adapter.write()` docstring（line ~1071） | 补充 JSON 收集行为 |
| `freerec/launcher.py` | `Adapter.fit()` docstring（line ~1190） | 补充 `results.json` 写入时机 |

### Skills 内嵌教程更新清单

| 文件 | 位置 | 更新内容 |
|------|------|----------|
| `freerec/skills.py` | `tune` skill（line ~221-240） | LOG STRUCTURE 目录树补充 `results.json` |
| `freerec/skills.py` | `tune` skill（line ~244-275） | BEST RESULTS FORMAT 部分补充 JSON 聚合输出说明 |
| `freerec/skills.py` | `log` skill（line ~330-346） | LOG DIRECTORY STRUCTURE 目录树补充 `core/results.json` |
| `freerec/skills.py` | `log` skill（line ~383-450） | pickle 文件结构部分补充 JSON 格式说明 |
| `freerec/skills.py` | `workflow` skill（line ~730-745） | LOGGING LAYER 补充 `results.json` 输出 |

### 用户文档更新清单

| 文件 | 位置 | 更新内容 |
|------|------|----------|
| `docs/tutorials/output.rst` | 目录结构树（line ~1-24） | 补充 `core/results.json` |
| `docs/tutorials/output.rst` | 文件格式说明（line ~62-138） | 新增 `results.json` 的结构说明 |
| `docs/tutorials/training_and_tuning.rst` | tune 输出部分（line ~80-102） | 补充 tune 完成后生成 `results.json` 的说明 |

## 后续规划

本 spec 仅涉及输出格式设计。后续 leaderboard 功能将基于此格式实现：
- 扫描 `logs/**/core/results.json` 聚合跨模型结果
- 按 dataset 分组生成排行榜
- 支持按 params 过滤（区分调参 vs 跑种子）
- 计算 mean ± std 等聚合统计
- 最终展示为 web 页面（在 RecBoard 中实现）
