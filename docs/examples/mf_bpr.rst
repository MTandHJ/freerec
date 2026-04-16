MF-BPR
======

**Matrix Factorization with Bayesian Personalized Ranking**

经典的矩阵分解协同过滤模型，使用 BPR 损失函数进行隐式反馈推荐。

用法
----

.. code-block:: bash

   # 全量排序
   python main.py --config=configs/xxx.yaml --ranking=full

   # 采样排序
   python main.py --config=configs/xxx.yaml --ranking=pool
