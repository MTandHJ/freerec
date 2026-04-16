# FreeRec Documentation Site - Implementation Plan

## Context

freerec 目前没有文档系统，需要搭建一个 Sphinx 文档站点（像素橙主题 + 像素装饰元素），覆盖安装指南、快速上手、教程、示例和 API Reference，部署到 GitHub Pages。

## Phase 1: 项目脚手架

### 1.1 添加文档依赖到 `pyproject.toml`

在 `[project.optional-dependencies]` 中新增：

```toml
docs = [
    "sphinx>=7.2",
    "pydata-sphinx-theme>=0.15",
    "sphinx-copybutton>=0.5",
]
```

### 1.2 创建 docs/ 目录结构

```
docs/
  conf.py
  index.rst
  Makefile
  make.bat
  requirements.txt
  installation.rst
  quickstart.rst
  tutorials/
    index.rst
    dataset_processing.rst
    training_and_tuning.rst
  examples/
    index.rst
    sasrec.rst / lightgcn.rst / bert4rec.rst / dcn.rst / gru4rec.rst / mf_bpr.rst
  api/
    index.rst
    freerec.data.rst
    freerec.models.rst
    freerec.launcher.rst
    freerec.metrics.rst
    freerec.criterions.rst
    freerec.utils.rst
  _static/
    css/custom.css
    img/  (logo.png, favicon.ico, pipeline.png, flow.png, splitting.png)
  _templates/
    autosummary/module.rst
    autosummary/class.rst
```

## Phase 2: Sphinx 配置 (`docs/conf.py`)

关键配置：
- **Extensions**: autodoc, autosummary, napoleon, viewcode, intersphinx, mathjax, sphinx_copybutton
- **Napoleon**: NumPy-style only, include `__init__` docs
- **Autodoc**: bysource 排序, typehints in description, mock imports (torch, torchdata, torch_geometric, tensorboard, sklearn, einops, polars, pandas, matplotlib, tqdm, prettytable, yaml)
- **Intersphinx**: Python, PyTorch, NumPy
- **Theme**: pydata_sphinx_theme, logo, favicon, light/dark toggle
- **Exclude**: `['_build', 'src']`（保留旧 docs/src/ 源文件但不参与构建）

## Phase 3: 像素橙自定义 CSS (`docs/_static/css/custom.css`)

- 覆盖 PyData 主题 CSS 变量：`--pst-color-primary: #e07020`
- 暗色模式适配：稍亮的 `#f08030`
- 像素装饰：admonition 像素边框、logo `image-rendering: pixelated`
- 像素风分隔线、按钮圆角去除等点缀

## Phase 4: Autosummary 模板

- `module.rst`: 自动列出 Classes 和 Functions 并生成子页面
- `class.rst`: 展示类成员、继承关系、方法摘要

## Phase 5: API Reference 页面（6 页）

| 页面 | 内容 |
|------|------|
| `freerec.data.rst` | autosummary recursive: fields, tags, datasets, preprocessing, postprocessing, normalizer, utils |
| `freerec.models.rst` | autosummary: models.base, models.nn |
| `freerec.launcher.rst` | automodule: Coach, ChiefCoach |
| `freerec.metrics.rst` | automodule: 全部指标函数 |
| `freerec.criterions.rst` | automodule: 全部损失类 |
| `freerec.utils.rst` | automodule: utils + parser + ddp（graph 可选模块也在此简述） |

## Phase 6: 内容页面（中文，重写）

- **index.rst**: 项目简介、核心特性、快速安装片段、导航链接
- **installation.rst**: 从 README 迁移重写，覆盖 Python/PyTorch 前置要求、pip install、可选依赖
- **quickstart.rst**: 端到端流程：安装 → 准备数据集 → 运行模型 → 查看结果
- **tutorials/dataset_processing.rst**: 从 `dataset processing.md` 重写，含数据格式、CLI 参考、拆分方法表、示意图
- **tutorials/training_and_tuning.rst**: 从 `training and tuning.md` 重写，含 YAML 配置、训练命令、分布式训练、超参调优
- **examples/**: 每个模型一页，含论文引用、模型简介、用法命令、配置示例

## Phase 7: 构建脚本

- `docs/Makefile`: 标准 Sphinx Makefile
- `docs/make.bat`: Windows 版本
- `docs/requirements.txt`: CI 用的独立依赖文件

## Phase 8: GitHub Actions (`/.github/workflows/docs.yml`)

- 触发：push to master, paths: docs/**, freerec/**, workflow file
- 构建：Python 3.11, 安装 docs deps + freerec, `sphinx-build -W --keep-going`
- 部署：actions/upload-pages-artifact + actions/deploy-pages@v4

## Phase 9: 迁移清理

- 删除根目录 `dataset processing.md`、`training and tuning.md`
- 更新 README.md 链接指向文档站点
- 迁移 `docs/src/` 图片到 `docs/_static/img/`

## 实施顺序

| 步骤 | 任务 |
|------|------|
| 1 | pyproject.toml 加 docs 依赖 |
| 2 | 创建 conf.py + Makefile + make.bat + requirements.txt |
| 3 | 创建 _static/css/custom.css + 迁移图片资源 |
| 4 | 创建 autosummary 模板 |
| 5 | 创建 6 个 API rst 页面 + api/index.rst |
| 6 | 本地构建验证 API 部分 |
| 7 | 编写 index.rst + installation.rst + quickstart.rst |
| 8 | 编写 tutorials 两页 |
| 9 | 编写 examples 页面 |
| 10 | 完整本地构建 + 视觉检查 |
| 11 | 创建 GitHub Actions workflow |
| 12 | 删除旧文档 + 更新 README |

## 验证方式

1. 本地运行 `sphinx-build docs docs/_build/html -b html -W` 确认零 warning
2. 浏览器打开 `docs/_build/html/index.html` 检查：主题配色、像素装饰、暗色模式、导航、API 页面
3. Push 后检查 GitHub Actions 构建成功 + Pages 部署正常

## 注意事项

- `autodoc_mock_imports` 必须覆盖所有 torch 相关依赖，否则 CI 构建失败
- `freerec.graph` 模块（依赖 torch_geometric）放在 utils 页面简述
- RST 中混排中文注意指令前后空行
