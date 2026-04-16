GRU4Rec
=======

**Session-based Recommendations with Recurrent Neural Networks**

基于 GRU 的会话推荐模型，通过循环神经网络建模用户行为序列。

用法
----

.. code-block:: bash

   # 全量排序
   python main.py --config=configs/xxx.yaml --ranking=full

   # 采样排序
   python main.py --config=configs/xxx.yaml --ranking=pool
