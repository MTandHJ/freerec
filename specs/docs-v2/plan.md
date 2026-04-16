# 文档风格变更 - 实施方案

## 变更概述

将文档视觉风格从像素橙切换为博客同款大地色系，添加中英双语支持。

## 步骤

### 1. 重写 `docs/_static/css/custom.css`

删除所有像素风 CSS，替换为博客配色：

- CSS 变量覆盖：`--pst-color-primary: #4b3425`
- 背景色：亮色 `#f0eeec` / 暗色 `#292524`
- 链接 hover：`#c0c0c0`
- 代码高亮：亮色暗色都用 Monokai
- 引入 Google Fonts Rubik，覆盖全局字体
- 去掉所有侧边栏宽度覆盖，恢复默认布局

### 2. 更新 `docs/conf.py`

- 删除 `html_logo`（不用图片 logo）
- `logo.text` 设为 `"FreeRec"`
- `pygments_light_style` 改为 `"monokai"`
- 删除 mermaid 图表中的像素橙配色引用

### 3. 清理像素资源

- 删除 `docs/_static/img/logo_pixel.png`
- 更新 `docs/index.rst` 中 logo 图片引用（改为无图片）

### 4. 更新 Mermaid 图表配色

- index.rst 和 quickstart.rst 中的 mermaid 图表颜色从 `#e07020` 改为 `#4b3425`

### 5. 中英双语支持

使用 Sphinx 多语言构建（推荐方案）：

**目录结构：**
```
docs/
├── conf.py           # 共享配置
├── locales/          # 翻译文件
│   └── en/
│       └── LC_MESSAGES/
│           └── *.po  # 英文翻译
├── index.rst         # 中文（默认语言）
├── installation.rst
├── ...
```

**conf.py 变更：**
```python
language = 'zh_CN'
locale_dirs = ['locales/']
gettext_compact = False
```

**构建流程：**
```bash
# 提取翻译模板
sphinx-build -b gettext docs docs/_build/gettext
sphinx-intl update -p docs/_build/gettext -l en

# 构建中文版
sphinx-build docs docs/_build/html/zh

# 构建英文版
sphinx-build -D language=en docs docs/_build/html/en
```

**导航栏语言切换：**
PyData 主题原生支持通过 `html_theme_options` 中的 `switcher` 或自定义模板添加中/EN 链接。

### 6. 更新 GitHub Actions

修改 `.github/workflows/docs.yml`：
- 安装 `sphinx-intl`
- 分别构建中文和英文版本
- 合并输出到同一 `_build/html/` 下

## 实施顺序

| 步骤 | 任务 |
|------|------|
| 1 | 重写 custom.css（配色 + 字体 + 去像素风） |
| 2 | 更新 conf.py（logo + 高亮 + 语言设置） |
| 3 | 清理像素资源 + 更新 rst 引用 |
| 4 | 更新 mermaid 图表配色 |
| 5 | 本地构建验证（中文版） |
| 6 | 设置 i18n，生成 .po 翻译模板 |
| 7 | 翻译 .po 文件为英文 |
| 8 | 双语构建验证 |
| 9 | 更新 GitHub Actions |

## 验证

1. 本地构建中文版，检查配色、字体、布局与博客一致
2. 本地构建英文版，检查翻译正确性
3. 确认语言切换链接可用
