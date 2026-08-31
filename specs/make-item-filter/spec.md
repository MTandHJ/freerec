# Spec

## Goal

为 `freerec make` 增加一个可选的数据清洗能力: 当原始输入目录中存在 `[DATASET].item` 文件时, 可以先删除 `[DATASET].inter` 中不在 item feature 表里的 item 交互, 再继续后续预处理流程。

该能力用于处理 item feature 表不完整或需要以 item feature 表作为候选 item universe 的数据集, 避免最终交互数据中出现缺失 item metadata 的 item。

## Scope

- `freerec make` 新增一个显式选项来开启该过滤行为。
- 默认行为保持不变: 不开启选项时, `.inter` 中出现但 `.item` 中不存在的 item 仍按现有逻辑保留。
- 开启选项且存在 `[DATASET].item` 时, 删除 `.inter` 中 `ITEM` 不存在于 `.item` 的交互。
- 过滤发生在加载原始文件之后, rating filter 和 k-core filter 之前。
- 开启选项但没有 `[DATASET].item` 时, 跳过该过滤并打印日志。
- 日志应说明是否执行过滤, 并报告过滤掉的 interaction 数量和涉及的缺失 item 数量。
- 输出目录命名沿用现有规则, 不因为该选项变化而增加新的命名片段。

## Non-goals

- 不改变默认 `freerec make` 的数据处理结果。
- 不改变 `[DATASET].user` 的处理规则。
- 不改变 rating filter, k-core filter, tokenization, split, save 的既有语义。
- 不要求 processed dataset loader 读取 `item.txt` 来约束 item universe。
- 不为同一参数组合生成新的输出目录命名规则。

## User Scenarios

- 用户拥有 `.inter` 和 `.item` 文件, 其中 `.inter` 包含部分没有 item feature 的 item。用户开启新选项后, 这些交互会在 k-core 前被删除。
- 用户拥有 `.inter` 文件但没有 `.item` 文件。用户误开启新选项时, `freerec make` 不失败, 只打印跳过日志并继续现有流程。
- 用户不关心 item feature 完整性。用户不传新选项时, `freerec make` 的行为与当前版本一致。

## Inputs / Outputs

输入:

- 原始 `[DATASET].inter`, 必需。
- 原始 `[DATASET].item`, 可选。
- `freerec make` 的新增显式选项, 用于开启基于 `.item` 的 interaction 过滤。

输出:

- `train.txt`, `valid.txt`, `test.txt` 保持现有格式。
- 如果源数据存在 `.item`, 继续输出 `item.txt`。
- 输出目录路径保持现有格式: `Processed/[DATASET]_[code]`。
- 控制台日志包含该过滤步骤的执行或跳过信息。

## Constraints

- 新功能必须是 opt-in, 不能改变未传选项时的兼容行为。
- 过滤依据使用重命名后的标准 `ITEM` 字段, 并尊重 `--itemColname` 对原始列名的映射。
- 过滤必须发生在 k-core 之前, 使 k-core 统计基于已清理的 interactions。
- 缺少 `.item` 文件时不能报错退出。
- 输出目录命名不携带该选项状态, 用户需要自行避免覆盖同名 processed 输出。

## Acceptance Criteria

- 当开启该选项且 `.item` 存在时, 所有输出 split 中的 item 都来自原始 `.item` 文件。
- 当开启该选项且 `.inter` 中存在 `.item` 缺失的 item 时, 这些 item 对应的 interactions 在 rating filter 和 k-core 前被删除。
- 当开启该选项但 `.item` 不存在时, 命令继续运行, 并打印清晰的跳过日志。
- 当不开启该选项时, 当前 `freerec make` 行为保持不变。
- 输出目录命名与现有规则一致, 不新增过滤选项相关后缀。

## Assumptions

- 这里的 `item.txt` 指的是 `freerec make` 输入阶段的 `[DATASET].item` 原始文件。
- `[DATASET].item` 的 item ID 列经过现有列名映射后为标准 `ITEM` 字段。
- 用户接受开启和不开启该选项时可能写入同一个 processed 输出目录。

## Open Questions

- 新 CLI 选项的具体名称待后续实现细化阶段确定。
