BERT4Rec
========

**BERT-based Sequential Recommendation**

采用双向 Transformer 编码器和 Masked Item Prediction 任务进行序列推荐。

用法
----

.. code-block:: bash

   # 全量排序
   python main.py --config=configs/xxx.yaml --ranking=full

   # 采样排序
   python main.py --config=configs/xxx.yaml --ranking=pool
