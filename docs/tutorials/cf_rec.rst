协同过滤
========

协同过滤模型基于用户-物品交互矩阵进行推荐，不依赖序列信息。
FreeRec 中的协同过滤模型继承自 :class:`~freerec.models.base.GenRecArch`。

MF-BPR
------

**Matrix Factorization with Bayesian Personalized Ranking** — 经典的矩阵分解模型，
使用 BPR 损失函数优化隐式反馈推荐的 pairwise 排序。

.. code-block:: bash

   python main.py --config=configs/xxx.yaml --ranking=full

.. literalinclude:: ../../examples/MF-BPR/main.py
   :language: python
   :caption: examples/MF-BPR/main.py
   :linenos:

**代码要点：**

- **模型结构** ：用户和物品各一个 Embedding 层，直接内积计算相似度
- **训练管道** ：``choiced_user_ids_source`` 随机采样用户 → 正样本采样 → 负样本采样
- **排序缓存** ：``reset_ranking_buffers`` 在评估前预计算所有嵌入，避免重复计算
- **损失函数** ：BPR Loss，优化正样本得分高于负样本

LightGCN
--------

**Simplifying Graph Convolution Network for Recommendation** — 轻量级图卷积推荐模型，
去除特征变换和非线性激活，仅保留邻域聚合操作。

.. note::

   需要安装 ``torch-geometric``：``pip install freerec[graph]``

.. code-block:: bash

   python main.py --config=configs/xxx.yaml --ranking=full

.. literalinclude:: ../../examples/LightGCN/main.py
   :language: python
   :caption: examples/LightGCN/main.py
   :linenos:

**代码要点：**

- **图构建** ：``to_normalized_adj`` 从训练集构建对称归一化邻接矩阵，注册为 buffer
- **图卷积** ：``encode`` 方法执行多层邻域聚合，最终嵌入为各层的均值
- **双损失** ：``fit`` 返回 ``(rec_loss, emb_loss)``，Coach 中以 ``weight_decay`` 加权合并
- **自定义 Coach** ：``CoachForLightGCN`` 重写了 ``set_optimizer`` 以禁用 PyTorch 内置 weight_decay（改由 emb_loss 手动控制正则化）
