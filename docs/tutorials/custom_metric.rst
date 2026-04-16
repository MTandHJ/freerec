自定义指标
==========

FreeRec 内置了 16 种评估指标，同时支持通过 ``register_metric`` 注册自定义指标。

内置指标
--------

**排序指标** （支持 ``@K`` 截断）：

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - 名称
     - 配置写法
     - 说明
   * - Precision
     - ``Precision@10``
     - Top-K 精确率
   * - Recall
     - ``Recall@20``
     - Top-K 召回率
   * - F1
     - ``F1@10``
     - Precision 和 Recall 的调和平均
   * - HitRate
     - ``HitRate@10``
     - Top-K 命中率（至少命中一个为 1）
   * - NDCG
     - ``NDCG@10``
     - 归一化折损累计增益
   * - MRR
     - ``MRR@10``
     - 平均倒数排名
   * - MAP
     - ``MAP@10``
     - 平均精度均值

**回归指标：**

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - 名称
     - 配置写法
     - 说明
   * - MSE
     - ``MSE``
     - 均方误差
   * - MAE
     - ``MAE``
     - 平均绝对误差
   * - RMSE
     - ``RMSE``
     - 均方根误差

**分类指标：**

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - 名称
     - 配置写法
     - 说明
   * - AUC
     - ``AUC``
     - ROC 曲线下面积
   * - GAUC
     - ``GAUC``
     - 分组 AUC（按用户加权）
   * - LogLoss
     - ``LOGLOSS``
     - 二元对数损失

**特殊指标：**

- ``LOSS`` — 训练损失，自动跟踪，无需手动注册

配置方式
--------

在 YAML 中指定要监控的指标和用于早停的指标：

.. code-block:: yaml

   monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
   which4best: NDCG@10

- ``monitors`` — 训练过程中跟踪的指标列表
- ``which4best`` — 用于选择最佳模型和早停判断的指标

注册自定义指标
--------------

使用 ``coach.register_metric()`` 在训练前注册：

.. code-block:: python

   import torch

   def my_topk_accuracy(
       preds: torch.Tensor,
       targets: torch.Tensor,
       *, k: int = 10
   ) -> torch.Tensor:
       """Top-K 准确率：预测的 Top-K 中包含正样本的比例。

       Parameters
       ----------
       preds : torch.Tensor
           预测分数，shape (B, N)
       targets : torch.Tensor
           真实标签（稀疏矩阵或 0/1 向量）
       k : int
           截断位置

       Returns
       -------
       torch.Tensor
           每个样本的准确率，shape (B,)
       """
       _, topk_indices = preds.topk(k, dim=-1)
       hits = targets.gather(1, topk_indices).sum(dim=-1)
       return hits / k

   # 在创建 Coach 后、调用 fit() 前注册
   coach = CoachForMyModel(dataset=dataset, ..., cfg=cfg)

   coach.register_metric(
       name='TopKAcc@10',          # 指标名称（@K 会自动提取为 k 参数）
       func=my_topk_accuracy,      # 指标函数
       fmt='.4f',                  # 显示格式
       best_caster=max,            # max 表示越大越好，min 表示越小越好
   )

   coach.fit()

**register_metric 参数说明：**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 参数
     - 说明
   * - ``name``
     - 指标全名，如 ``'TopKAcc@10'``。含 ``@`` 时自动提取 K 值传给函数
   * - ``func``
     - 指标函数，签名为 ``func(preds, targets, *, k=...) -> Tensor``
   * - ``fmt``
     - 数值显示格式，默认 ``'.4f'``
   * - ``best_caster``
     - ``max`` 或 ``min``，表示指标是越大越好还是越小越好

指标函数规范
------------

自定义指标函数需满足：

1. 接受 ``preds`` 和 ``targets`` 两个位置参数
2. 如果支持 ``@K`` 截断，需接受 ``k`` 关键字参数
3. 返回一个标量或一维张量（会被自动 reduce）

可以使用 ``freerec.metrics._reduce`` 装饰器自动添加 reduction 支持：

.. code-block:: python

   from freerec.metrics import _reduce

   @_reduce('mean')
   def my_metric(preds, targets, *, k=10):
       # 返回每个样本的指标值，形状 (B,)
       _, topk = preds.topk(k, dim=-1)
       hits = targets.gather(1, topk).sum(-1)
       return hits.float() / k

   # 调用时可覆盖 reduction
   result = my_metric(preds, targets, k=20, reduction='sum')
