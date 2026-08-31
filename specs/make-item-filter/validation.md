# Validation

## Scope

验证 `freerec make` 的 `-mif, --match-item-file` 选项是否正确实现以下行为:

- 默认不改变现有数据处理结果。
- 开启后, 使用输入阶段 `[DATASET].item` 文件中的 raw item IDs 过滤 `[DATASET].inter`。
- 过滤发生在 rating filter 和 k-core filter 之前。
- 缺少 `.item` 文件时跳过并继续运行。
- 输出目录命名保持现有规则。
- 文档和 `freerec skill --make` 同步说明新选项。

## Acceptance Checks

- 对应 spec: “开启该选项且 `.item` 存在时, 所有输出 split 中的 item 都来自原始 `.item` 文件。”
  - 检查 raw item IDs 在 tokenization 前已经被 `.item` 限定。
  - 检查 tokenization 后输出 split 中不再包含由 `.item` 缺失 item 产生的 rows。

- 对应 spec: “`.inter` 中存在 `.item` 缺失的 item 时, 这些 item 对应的 interactions 在 rating filter 和 k-core 前被删除。”
  - 使用一个会因为提前删除 missing item 而改变 k-core 结果的小数据集验证执行顺序。

- 对应 spec: “开启该选项但 `.item` 不存在时, 命令继续运行, 并打印清晰的跳过日志。”
  - 构造只有 `.inter` 的数据集, 开启 `match_item_file`, 验证流程完成并捕获 skip 日志。

- 对应 spec: “不开启该选项时, 当前 `freerec make` 行为保持不变。”
  - 使用包含 missing item 的相同数据集, 不启用 `match_item_file`, 验证 missing item 对应 interactions 没有被该步骤删除。

- 对应 spec: “输出目录命名与现有规则一致, 不新增过滤选项相关后缀。”
  - 验证输出目录仍为 `Processed/[DATASET]_[code]`, 不包含 `match-item-file` 或类似片段。

## Test Cases

- `AtomicConverter.filter_inter_by_item_file()` unit test:
  - 用 `tmp_path` 构造 `.inter` 和 `.item`。
  - `.inter` 包含 `i1`, `i2`, `i_missing`。
  - `.item` 只包含 `i1`, `i2`。
  - 调用 `load()` 后再调用 `filter_inter_by_item_file()`。
  - 断言 `i_missing` 对应 rows 被删除, `i1/i2` rows 保留。

- `make_dataset(match_item_file=True)` integration test:
  - 构造 RecBole-style atomic 文件。
  - 使用 `kcore4user=0`, `kcore4item=0`, `star4pos=0`, 降低无关过滤影响。
  - 运行完整 `make_dataset()`。
  - 检查 `train.txt`, `valid.txt`, `test.txt` 的总 rows 数符合过滤后的预期。
  - 检查输出 split 中没有缺失 item 产生的 tokenized rows。

- Default regression test:
  - 使用与 integration test 相同的输入。
  - 运行 `make_dataset(match_item_file=False)`。
  - 断言 missing item 对应 interactions 不会被 item-file match 步骤删除。

- Missing `.item` test:
  - 构造只有 `.inter` 的输入目录。
  - 运行 `make_dataset(match_item_file=True)`。
  - 断言命令完成, 输出 split 文件生成。
  - 捕获日志, 断言包含 skip `.item` filtering 的信息。

- K-core ordering test:
  - 构造一个数据集, 其中 missing item 的提前删除会让某个 user 或 item 低于 k-core 阈值。
  - 运行 `make_dataset(match_item_file=True, kcore4user=..., kcore4item=...)`。
  - 断言结果符合 “先按 `.item` 删除, 再做 k-core” 的预期。

- CLI parser test:
  - 验证 `freerec make --help` 包含 `-mif` 和 `--match-item-file`。
  - 验证传入 `--match-item-file` 后 argparse namespace 中 `match_item_file` 为 `True`。

- Custom item column test:
  - 原始 `.inter` 和 `.item` 使用非默认 item 列名。
  - 通过 converter 参数或 CLI `--itemColname` 指定映射。
  - 断言过滤仍基于映射后的标准 `ITEM` 字段执行。

## Edge Cases

- `.item` 中存在重复 item IDs:
  - allowed items 应去重后使用。
  - 重复 rows 不应改变过滤结果。

- `.item` 文件存在但没有额外 feature 列:
  - 只要存在 item ID 列, 过滤应正常工作。
  - 该功能不要求 `.item` 有 feature columns。

- `.inter` 中所有 items 都存在于 `.item`:
  - 开启选项后不删除任何 interactions。
  - 日志中的 removed interactions 应为 0。

- `.inter` 中部分 missing item 在 rating filter 后本来也会被删除:
  - 验证日志和结果体现 item-file match 先执行, 不依赖 rating filter 间接删除。

## Failure Cases

- `.item` 文件不存在:
  - 不报错, 打印 skip 日志, 后续流程继续。

- `.item` 文件存在但没有可映射到标准 `ITEM` 的列:
  - 不新增特殊兼容逻辑。
  - 继续通过现有列名映射和 pandas/key error 路径暴露问题。

- 开启过滤后 interactions 变空:
  - 不新增特殊容错。
  - 由后续现有流程失败或报错。
  - 该行为不应被测试包装成静默成功。

## Manual Checks

- 运行 `freerec make --help`, 确认新选项展示为 `-mif, --match-item-file`, help 文案说明 “drop interactions whose ITEM does not appear in the .item file”。
- 运行 `freerec skill --make`, 确认 Key Options 和 Pipeline 说明包含新选项和可选 item-file match 过滤步骤。
- 阅读 `docs/tutorials/dataset_processing.rst`, 确认:
  - 参数表包含 `-mif, --match-item-file`。
  - 文档说明过滤发生在 k-core 前。
  - 示例命令展示新选项。
- 对一个小型临时数据集手动运行:
  - 不带 `--match-item-file`。
  - 带 `--match-item-file`。
  - 比较输出 rows 数和日志, 确认行为差异符合预期。

## Open Questions

None.
