模型示例
========

以下示例展示了如何使用 FreeRec 实现常见的推荐模型。
每个模型均提供完整的训练代码和配置文件，可直接运行。

**序列推荐：**

.. toctree::
   :maxdepth: 1

   sasrec
   bert4rec
   gru4rec

**协同过滤：**

.. toctree::
   :maxdepth: 1

   mf_bpr
   lightgcn

**点击率预测：**

.. toctree::
   :maxdepth: 1

   dcn

通用用法
--------

大多数示例的运行方式相同：

.. code-block:: bash

   # 全量排序评估
   python main.py --config=configs/xxx.yaml --ranking=full

   # 采样排序评估（更快）
   python main.py --config=configs/xxx.yaml --ranking=pool
