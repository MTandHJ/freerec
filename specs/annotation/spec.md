

## 需求

对 freerec 项目的所有 Python 代码进行全面的 docstring 审查与重写，统一为 NumPy-style 格式，确保质量和风格一致。

## 范围

- **全面重写**：对所有代码的 docstring 进行审查和重写，不仅补全缺失的，还统一已有但格式不一致的注释。
- **覆盖模块**：`freerec/` 下所有 Python 文件，包括 `data/`、`models/`、`launcher.py`、`criterions.py`、`metrics.py`、`utils.py`、`parser.py`、`graph.py`、`dict2obj.py`、`ddp.py`、`skills.py` 等。
- **包含 .pyi 文件**：`data/postprocessing/base.pyi` 也在范围内（见下方 .pyi 专项规范）。

## 注释粒度

采用**严格模式**：所有类、函数、方法都必须有 docstring。

### 完整 section（复杂对象）

适用于：多参数的公开函数/方法、核心类、有异常抛出的方法、复杂逻辑。

必须包含的 section（视情况）：
- **Summary**：一行功能描述
- **Parameters**：所有参数说明（含类型、默认值）
- **Returns**：返回值说明
- **Raises**：异常说明（如有）
- **Examples**：使用示例（复杂接口建议提供）
- **Notes**：补充说明（如有数学公式、算法细节等）

### 仅摘要（简单对象）

适用于：trivial getter/setter、单行逻辑方法、`__repr__`/`__str__` 等 dunder 方法、简单内部辅助函数。

只需一行摘要即可。

## Parameters 位置

- **类的 Parameters 放在类级别 docstring 中**（非 `__init__`），以便 IDE hover 和 `help()` 直接展示。
- `__init__` 方法写一行摘要或省略 docstring。

## 格式规范

- **语言**：英文
- **风格**：NumPy-style
- **字符串前缀**：使用 `r"""..."""`（raw docstring）
- **section 分隔**：section 名后接换行和等长的短横线 `----------`

### rST 交叉引用

为支持后续构建 PyTorch 风格的 Sphinx 文档网站（类型可点击跳转），docstring 中的类型引用需使用 rST 语法：

- **类引用**：`:class:\`torch.Tensor\``、`:class:\`~RecSysArch\``（`~` 前缀只显示短名称）
- **函数引用**：`:func:\`to_adjacency\``
- **方法引用**：`:meth:\`forward\``
- **模块引用**：`:mod:\`freerec.data\``
- **参数引用**：`:attr:\`device\``

**适用范围**：所有 docstring 中出现的类型标注和交叉引用。

**Python 内置类型例外**：`str`、`int`、`float`、`bool`、`None`、`dict`、`list`、`tuple` 等内置类型**不需要**加 rST 引用，直接写纯文本即可。

### 数学公式

使用 rST math 语法，Sphinx 可直接渲染为 LaTeX：

- **行内公式**：`` :math:`\alpha + \beta` ``
- **块级公式**：
  ```rst
  .. math::

      \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
  ```

## .pyi Stub 文件规范

`data/postprocessing/base.pyi` 为动态注册到 `BaseProcessor` 上的 datapipe 方法提供类型提示和文档。

### 同步策略

- **.py 和 .pyi 都写完整 docstring**，手动保持同步。
- 修改 `.py` 中实现类的 docstring 时，必须同步更新 `.pyi` 中对应的 functional method docstring，反之亦然。

### torchdata 原生方法

`.pyi` 中从 torchdata 上游复制的方法（`batch`、`collate`、`map`、`unbatch`、`sharding_filter`、`map_batches`）**保持原有 Google-style (Args:) 不改写**，以便后续与上游同步更新。

### 仅 postprocessing 需要 .pyi

当前只有 `data/postprocessing/` 存在动态注册问题，其他模块不需要创建 .pyi 文件。

### 模板

```python
class MyClass:
    r"""One-line summary of the class.

    Extended description if needed.

    Parameters
    ----------
    param1 : :class:`torch.Tensor`
        Description of param1.
    param2 : :class:`~RecSysArch`, optional
        Description of param2, by default None.

    Attributes
    ----------
    attr1 : :class:`torch.device`
        Description of attr1.

    Raises
    ------
    ValueError
        When something is wrong.

    Examples
    --------
    >>> obj = MyClass(param1, param2)
    """

    def __init__(self, param1, param2):
        r"""Initialize MyClass."""
        ...

    def complex_method(self, x):
        r"""One-line summary.

        Extended description if needed.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            Description of x.

        Returns
        -------
        :class:`torch.Tensor`
            Description of result.
        """
        ...

    def compute_score(self, y_true, y_pred):
        r"""Compute the evaluation score.

        The score is defined as:

        .. math::

            \text{score} = \frac{1}{N} \sum_{i=1}^{N} f(y_i, \hat{y}_i)

        Parameters
        ----------
        y_true : :class:`torch.Tensor`
            Ground truth values.
        y_pred : :class:`torch.Tensor`
            Predicted values.

        Returns
        -------
        float
            The computed score.
        """
        ...

    @property
    def name(self):
        r"""The name of this object."""
        ...

    def __repr__(self):
        r"""Return a string representation."""
        ...
```
