# freerec Documentation Spec

## 目标

为 freerec 搭建一个类似 PyTorch 的文档站点，方便用户查询 API、学习使用方法。

## 技术选型

| 项目 | 方案 |
|------|------|
| 文档生成工具 | Sphinx |
| 主题 | PyData Sphinx Theme |
| Docstring 解析 | `sphinx.ext.napoleon`（NumPy-style） |
| 自动 API 生成 | `sphinx.ext.autodoc` + `sphinx.ext.autosummary` |
| 部署 | GitHub Pages |
| CI/CD | GitHub Actions（push 到主分支时自动构建部署） |

## 文档语言

- **教程/指南部分**：中文
- **API Reference**：英文（直接从 docstring 生成）

## 文档结构

```
docs/
├── conf.py                  # Sphinx 配置
├── index.rst                # 首页
├── Makefile                 # 构建脚本
├── installation.rst         # 安装指南
├── quickstart.rst           # 快速上手
├── tutorials/               # 教程指南
│   ├── index.rst
│   ├── dataset_processing.rst    # 数据集处理（迁移自现有 MD）
│   └── training_and_tuning.rst   # 训练与调参（迁移自现有 MD）
├── examples/                # 模型示例
│   ├── index.rst
│   └── ...                  # 从 examples/ 目录整合
└── api/                     # API Reference
    ├── index.rst
    ├── freerec.data.rst
    ├── freerec.models.rst
    ├── freerec.launcher.rst
    ├── freerec.metrics.rst
    ├── freerec.criterions.rst
    └── freerec.utils.rst
```

## API Reference 组织

按模块分页，每个模块一个独立页面，包含：
- 模块概述
- 类/函数列表（autosummary 表格）
- 各类/函数的详细文档（autodoc 自动提取）

主要模块划分：

| 页面 | 覆盖内容 |
|------|----------|
| `freerec.data` | fields, tags, datasets, preprocessing, postprocessing, normalizer |
| `freerec.models` | RecSysArch 基类, nn 子模块（attn, ffn, utils） |
| `freerec.launcher` | Coach, ChiefCoach |
| `freerec.metrics` | 全部评估指标 |
| `freerec.criterions` | 全部损失函数 |
| `freerec.utils` | AverageMeter, Monitor, ddp, parser 等工具模块 |

## 首页内容

- 项目简介
- 核心特性
- 安装方式（简版，链接到完整安装页）
- 快速上手代码片段
- 导航到各部分的链接

## CI/CD 配置

GitHub Actions workflow：
- 触发条件：push 到 `master` 分支且 `docs/` 或 `freerec/` 目录有变更
- 构建步骤：安装依赖 → `sphinx-build` → 部署到 `gh-pages` 分支
- 使用 `peaceiris/actions-gh-pages` 或类似 action 部署

## Sphinx 扩展

- `sphinx.ext.autodoc` — 自动从源码提取文档
- `sphinx.ext.autosummary` — 生成模块摘要表格
- `sphinx.ext.napoleon` — 解析 NumPy-style docstring
- `sphinx.ext.viewcode` — 添加 [source] 链接查看源码
- `sphinx.ext.intersphinx` — 链接到 PyTorch/Python 外部文档
- `sphinx.ext.mathjax` — 数学公式渲染

## 视觉风格

- **基础主题**：PyData Sphinx Theme
- **主色调**：像素橙 `#e07020`
- **暗色模式**：支持 light/dark 切换（PyData 主题原生支持）
- **像素风装饰元素**：
  - 像素风 Logo / favicon
  - 像素风分隔线、装饰边框等 CSS 点缀
  - 可选：像素风图标用于导航或章节标题装饰
- **整体布局**：保持 PyData 主题的专业布局不变，像素元素仅作点缀

## 内容编写要求

- **安装指南、快速上手、教程指南**：根据现有内容（README、`dataset processing.md`、`training and tuning.md`）重写，使语言简洁易懂
- **迁移后清理**：文档站点建成后，删除根目录下的旧文档文件（`dataset processing.md`、`training and tuning.md`）
- **API Reference**：从 docstring 自动生成，不需手动编写

## 非目标（当前不做）

- 多版本文档切换
- 搜索功能定制（使用 Sphinx 内置搜索即可）
- 国际化 i18n 框架（直接手写中英内容，不用 gettext）
