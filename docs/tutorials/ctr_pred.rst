点击率预测
==========

点击率预测 (CTR Prediction) 模型根据用户和物品的多维特征预测点击概率。
FreeRec 中的 CTR 模型继承自 :class:`~freerec.models.base.PredRecArch`。

DCN
---

**Deep & Cross Network** — 通过交叉网络 (Cross Network) 显式建模特征间的高阶交互，
结合深度网络 (DNN) 捕捉隐式交互。

.. code-block:: bash

   python main.py --config=configs/xxx.yaml

.. literalinclude:: ../../examples/DCN/main.py
   :language: python
   :caption: examples/DCN/main.py
   :linenos:

**代码要点：**

- **多特征嵌入** ：对离散特征使用 ``nn.Embedding``，对连续特征使用 ``nn.Linear`` + ``Unsqueeze``，拼接为统一输入
- **交叉网络** ：``CrossInteraction`` 层实现 :math:`x_{i+1} = x_0 \cdot w^T x_i + b + x_i`，逐层叠加特征交叉
- **并行结构** ：Cross Network 和 DNN 并行处理同一输入，最终拼接后通过全连接层输出
- **差异化正则** ：``marked_params`` 对嵌入层和其他参数使用不同的 weight_decay
- **学习率调度** ：使用 ``ReduceLROnPlateau``，根据验证集最佳指标自动降低学习率
- **训练管道** ：``shuffled_inter_source`` 直接打乱交互记录（非序列），每条记录包含所有特征字段
