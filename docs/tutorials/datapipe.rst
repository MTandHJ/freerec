数据管道 (DataPipe)
===================

FreeRec 基于 torchdata 的 DataPipe 机制构建了链式数据处理管道，
通过方法链将数据从原始格式转换为模型可用的张量批次。

核心理念
--------

数据管道由一系列 **Processor** 组成，每个 Processor 消费上游数据并产出处理后的数据，
形成一条从数据源到最终张量的流水线：

.. code-block:: text

   Source → Sampler → Row Ops → Batch → Tensor
   ↑                                      ↓
   (原始数据)                          (模型输入)

所有 Processor 通过 **方法链** (method chaining) 串联：

.. code-block:: python

   trainpipe = (
       dataset.train()
       .shuffled_seqs_source(maxlen=50)        # 1. 数据源
       .seq_train_yielding_pos_()               # 2. 正样本生成
       .seq_train_sampling_neg_(num_negatives=1)# 3. 负采样
       .add_(offset=1, modified_fields=(ISeq,)) # 4. 行操作
       .lpad_(50, modified_fields=(ISeq,))      # 5. 行操作
       .batch_(256)                              # 6. 批处理
       .tensor_()                                # 7. 转为张量
   )

类层次结构
----------

.. code-block:: text

   torchdata.datapipes.iter.IterDataPipe
   └── BaseProcessor          # 持有 dataset 和 fields 引用
       ├── Source              # 数据源，管道的起点
       ├── PostProcessor       # 处理节点，自动回溯找到 dataset
       └── SampleMultiplexer   # 多数据源加权采样

- **Source** ：管道的起点，从 RecDataSet 中读取数据并封装为可迭代对象
- **PostProcessor** ：所有中间处理节点的基类，包装上游管道并逐条处理
- **SampleMultiplexer** ：按权重从多个数据源交替采样

内置 Processor 一览
-------------------

**数据源 (Source)**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 方法
     - 说明
   * - ``.ordered_source_()``
     - 按原始顺序输出（用于验证/测试）
   * - ``.shuffled_source_()``
     - 每个 epoch 随机打乱
   * - ``.choiced_source_()``
     - 有放回随机采样

**采样器 (Sampler)**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 方法
     - 说明
   * - ``.gen_train_sampling_pos_()``
     - 为每个用户采样一个正样本（协同过滤）
   * - ``.gen_train_sampling_neg_(num_negatives)``
     - 采样负样本（协同过滤）
   * - ``.seq_train_yielding_pos_()``
     - 从序列中生成正样本目标（序列推荐）
   * - ``.seq_train_sampling_neg_(num_negatives)``
     - 为序列采样负样本
   * - ``.valid_sampling_(ranking)``
     - 验证集采样（full 或 pool 模式）
   * - ``.test_sampling_(ranking)``
     - 测试集采样

**行操作 (Row Ops)**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 方法
     - 说明
   * - ``.lpad_(maxlen, ...)``
     - 左填充到指定长度
   * - ``.rpad_(maxlen, ...)``
     - 右填充到指定长度
   * - ``.lprune_(maxlen, ...)``
     - 从左截断到指定长度
   * - ``.rprune_(maxlen, ...)``
     - 从右截断到指定长度
   * - ``.add_(offset, ...)``
     - 对指定字段加偏移量

**批处理与张量化**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 方法
     - 说明
   * - ``.batch_(batch_size)``
     - 将数据分批
   * - ``.tensor_()``
     - 将批次转换为 PyTorch 张量

典型管道示例
------------

**序列推荐（SASRec 风格）：**

.. code-block:: python

   trainpipe = (
       dataset.train()
       .shuffled_seqs_source(maxlen=50)         # 打乱的用户序列
       .seq_train_yielding_pos_(                 # 序列右移生成目标
           start_idx_for_target=1,
           end_idx_for_input=-1
       )
       .seq_train_sampling_neg_(num_negatives=1) # 每个位置采一个负样本
       .add_(offset=NUM_PADS, modified_fields=(ISeq,))  # 为 padding 留出空间
       .lpad_(maxlen, modified_fields=(ISeq, IPos, INeg),
              padding_value=PADDING_VALUE)        # 左填充到固定长度
       .batch_(batch_size)
       .tensor_()
   )

**协同过滤（MF-BPR 风格）：**

.. code-block:: python

   trainpipe = (
       dataset.train()
       .choiced_user_ids_source()        # 有放回采样用户 ID
       .gen_train_sampling_pos_()         # 为每个用户采正样本
       .gen_train_sampling_neg_(          # 为每个用户采负样本
           num_negatives=1
       )
       .batch_(batch_size)
       .tensor_()
   )

自定义 DataPipe
---------------

通过 ``@dp.functional_datapipe`` 装饰器注册自定义管道，
继承 ``PostProcessor`` 并实现 ``__iter__`` 方法：

.. code-block:: python

   import torchdata.datapipes as dp
   from freerec.data.postprocessing.base import PostProcessor

   @dp.functional_datapipe("my_transform_")
   class MyTransform(PostProcessor):
       """对指定字段应用自定义变换。"""

       def __init__(self, source, scale: float = 1.0):
           super().__init__(source)
           self.scale = scale

       def __iter__(self):
           for row in self.source:
               # row 是一个 {Field: value} 字典
               row[self.ISeq] = tuple(x * self.scale for x in row[self.ISeq])
               yield row

   # 注册后即可在管道中使用
   trainpipe = (
       dataset.train()
       .shuffled_seqs_source(maxlen=50)
       .my_transform_(scale=2.0)  # 使用自定义管道
       .batch_(256)
       .tensor_()
   )

**关键点：**

- 装饰器名称（如 ``"my_transform_"``）就是链式调用的方法名
- ``__init__`` 第一个参数必须是 ``source``（上游管道）
- ``__iter__`` 逐条处理并 ``yield`` 修改后的行
- ``row`` 是 ``Dict[Field, Any]``，通过 Field 对象访问和修改数据
