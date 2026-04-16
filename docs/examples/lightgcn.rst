LightGCN
========

**LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**

轻量级图卷积推荐模型，通过简化 GCN 去除特征变换和非线性激活，仅保留邻域聚合。

.. note::

   需要安装 ``torch-geometric``：``pip install freerec[graph]``

用法
----

.. code-block:: bash

   # 全量排序
   python main.py --config=configs/xxx.yaml --ranking=full

   # 采样排序
   python main.py --config=configs/xxx.yaml --ranking=pool
