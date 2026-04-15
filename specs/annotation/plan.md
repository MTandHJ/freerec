# Plan: FreeRec Docstring Annotation Overhaul

## Context

FreeRec 项目需要对所有 Python 代码进行全面的 docstring 审查与重写，统一为 NumPy-style 格式，加入 rST 交叉引用和 math 语法，以支持后续构建 PyTorch 风格的 Sphinx 文档网站。详细 spec 见 `specs/annotation/spec.md`。

全量约 48 个文件、450+ 个需注释对象，当前覆盖率约 45%。

## 执行策略

按模块分批处理，每批完成后标记任务完成。每个文件内：
1. 审查所有类/函数/方法的 docstring
2. 按 spec 重写或补全（复杂对象完整 section，简单对象一行摘要）
3. 类型引用加 rST 语法（`:class:`、`:func:` 等），内置类型除外
4. 数学公式用 `:math:` / `.. math::`
5. Parameters 放类级别，`__init__` 仅摘要

## 任务分批

### Batch 1: 核心基础模块（高优先级，覆盖率低）
1. **`freerec/parser.py`** — 8% 覆盖率，Parser/CoreParser 类及方法
2. **`freerec/launcher.py`** — 29% 覆盖率，ChiefCoach/Coach/Adapter 及 30+ 方法
3. **`freerec/utils.py`** — 33% 覆盖率，AverageMeter/Monitor 及工具函数
4. **`freerec/ddp.py`** — 45% 覆盖率，分布式训练工具函数

### Batch 2: 数据管道模块
5. **`freerec/data/fields.py`** — 53% 覆盖率，Field/FieldTuple/FieldModule
6. **`freerec/data/normalizer.py`** — 30% 覆盖率，Normalizer 及子类
7. **`freerec/data/tags.py`** — 0% 覆盖率，FieldTags/TaskTags 枚举
8. **`freerec/data/utils.py`** — 67% 覆盖率

### Batch 3: 数据后处理模块
9. **`freerec/data/postprocessing/base.py`** — 53% 覆盖率
10. **`freerec/data/postprocessing/sampler.py`** — 40% 覆盖率
11. **`freerec/data/postprocessing/row.py`** — 47% 覆盖率
12. **`freerec/data/postprocessing/source.py`** — 63% 覆盖率
13. **`freerec/data/postprocessing/column.py`**
14. **`freerec/data/postprocessing/other.py`** — 53% 覆盖率
15. **`freerec/data/postprocessing/base.pyi`** — 同步更新，torchdata 原生方法保持 Google-style

### Batch 4: 数据预处理与数据集
16. **`freerec/data/preprocessing/base.py`** — 39% 覆盖率
17. **`freerec/data/preprocessing/amazon2023.py`** — 50% 覆盖率
18. **`freerec/data/datasets/base.py`** — 67% 覆盖率，核心基类

### Batch 5: 模型模块
19. **`freerec/models/base.py`** — 48% 覆盖率，RecSysArch 及子类
20. **`freerec/models/nn/attn.py`** — 100% 但需格式统一和 rST 引用
21. **`freerec/models/nn/ffn.py`** — 100% 但需格式统一和 rST 引用
22. **`freerec/models/nn/utils.py`** — 0% 覆盖率

### Batch 6: 指标与损失函数
23. **`freerec/metrics.py`** — 59% 覆盖率，需补全 + 加数学公式
24. **`freerec/criterions.py`** — 92% 覆盖率，微调格式和 rST 引用

### Batch 7: 辅助模块
25. **`freerec/dict2obj.py`** — 70% 覆盖率
26. **`freerec/graph.py`** — 67% 覆盖率
27. **`freerec/__init__.py`** — declare 函数
28. **`freerec/__main__.py`** — 0% 覆盖率
29. **`freerec/skills.py`** — 0% 覆盖率

### Batch 8: 数据集定义文件（已 100% 但需格式审查）
30. **`freerec/data/datasets/`** 下所有数据集文件 — 已有 docstring，审查格式一致性和 rST 引用

## 验证

- 每批完成后用 `python -c "import freerec"` 验证无语法错误
- 抽查 docstring 格式是否符合 spec（rST 引用、section 格式、数学公式）
- 最终全量 `python -m pytest`（如有测试）确认无回归
