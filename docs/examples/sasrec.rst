SASRec
======

**Self-Attentive Sequential Recommendation**

基于自注意力机制的序列推荐模型，通过 Transformer 编码器捕捉用户行为序列中的长短期依赖关系。

用法
----

.. code-block:: bash

   # 全量排序
   python main.py --config=configs/xxx.yaml --ranking=full

   # 采样排序
   python main.py --config=configs/xxx.yaml --ranking=pool

配置示例
--------

.. code-block:: yaml

   root: ./data
   dataset: Amazon2014Beauty_550_LOU

   maxlen: 50
   num_heads: 1
   num_blocks: 2
   embedding_dim: 64
   dropout_rate: 0.4

   epochs: 300
   batch_size: 512
   optimizer: adam
   lr: 5.e-4
   weight_decay: 1.e-8

   monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
   which4best: NDCG@10

模型继承自 :class:`~freerec.models.base.SeqRecArch`，使用
:class:`~freerec.launcher.Coach` 管理训练流程。
