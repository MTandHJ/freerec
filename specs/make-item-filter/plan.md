# Plan

## Overview

为 `freerec make` 增加 opt-in 过滤选项 `-mif, --match-item-file`。

开启后, `AtomicConverter` 在加载 `.inter/.user/.item` 后, 使用原始 `.item` 文件中的 `ITEM` 列作为允许集合, 删除 `.inter` 中 `ITEM` 不在该集合内的 interactions。该过滤发生在 rating filter 和 k-core filter 之前, 使后续统计和 tokenization 都基于清理后的 interactions。

默认不启用该过滤, 以保持现有行为兼容。

## Implementation

修改 `freerec/__main__.py`:

- 在 `make_parser` 中新增布尔参数:
  - `-mif`
  - `--match-item-file`
  - `action="store_true"`
  - `dest="match_item_file"`
- help 文案使用直白描述: `drop interactions whose ITEM does not appear in the .item file`。
- 在 `make(args)` 中将 `args.match_item_file` 传给 `converter.make_dataset(...)`。

修改 `freerec/data/preprocessing/base.py`:

- 在 `AtomicConverter` 中新增方法 `filter_inter_by_item_file()`。
- 方法逻辑:
  - 如果 `self.itemFeats is None`, 打印 skip 日志并返回。
  - 从 `self.itemFeats[ITEM.name]` 取唯一 item 集合作为 allowed items。
  - 统计 `self.interactions[ITEM.name]` 中不在 allowed items 的唯一 item 数量。
  - 删除 `self.interactions` 中不在 allowed items 的行。
  - 打印删除的 interaction 数量和缺失 item 数量。
- 在 `make_dataset()` 签名中新增参数 `match_item_file: bool = False`。
- 在 `self.load()` 之后, `filter_by_rating()` 之前, 当 `match_item_file` 为 `True` 时调用 `filter_inter_by_item_file()`。
- 不改变 `filter_by_rating`, `filter_by_core`, `user2token`, `item2token`, splitting, save 的语义。

更新文档:

- 更新 `docs/tutorials/dataset_processing.rst`:
  - 在命令行参数表中加入 `-mif, --match-item-file`。
  - 增加一段说明: 开启后会删除 `.inter` 中不在 `.item` 文件里的 item interactions, 该过滤在 k-core 前执行。
  - 增加一个简短示例命令。
- 更新 `freerec/skills.py` 中 `make` skill:
  - Key Options 加入 `--match-item-file`。
  - Pipeline 文案体现可选 item-file match 过滤步骤。

## Behavior

未传 `-mif/--match-item-file`:

- 行为与当前版本一致。
- `.inter` 中存在但 `.item` 中不存在的 item 仍会保留。
- 输出目录命名不变。

传入 `-mif/--match-item-file` 且 `.item` 存在:

- 只保留 `interactions[ITEM]` 出现在 `.item` 文件 `ITEM` 列中的 rows。
- 删除发生在 rating filter 和 k-core filter 前。
- 输出 split 中的 item 都来自原始 `.item` 文件。
- 日志报告删除了多少 interactions, 以及涉及多少个不在 `.item` 中的 item。

传入 `-mif/--match-item-file` 但 `.item` 不存在:

- 不报错。
- 打印清晰的 skip 日志。
- 后续流程按当前逻辑继续执行。

## Compatibility

- 新选项是 opt-in, 不影响已有命令和脚本。
- 使用 kebab-case 长选项 `--match-item-file`; `argparse` 内部变量使用 `match_item_file`。
- 短选项 `-mif` 采用当前项目已有的多字符单横线风格, 与 `-ku`, `-ki`, `-uc`, `-ic` 一致。
- 输出目录仍使用现有规则, 不包含该过滤选项状态。用户需要自行避免开启/关闭该选项时写入同一 processed 目录造成混淆。
- `.item` 文件存在但缺少可映射到标准 `ITEM` 的列时, 不新增特殊兼容逻辑, 继续遵循现有列名映射和后续报错行为。

## Notes

- 这里的 `.item` 指 `freerec make` 输入阶段的 `[DATASET].item` 原始文件, 不是 processed 输出目录中的 `item.txt`。
- 该功能只检查 item 是否出现在 `.item` 文件中, 不要求 `.item` 中存在额外 feature 列。
- 文档以 `--match-item-file` 为主展示, 同时保留 `-mif` 作为简洁用法。
