训练与调参
==========

本节介绍如何使用 FreeRec 进行模型训练和超参数搜索。

配置文件
--------

FreeRec 使用 YAML 文件管理训练配置。一个典型的配置文件包含：

.. code-block:: yaml

   # 数据设置
   root: ./data
   dataset: Amazon2014Beauty_550_LOU

   # 模型超参数
   maxlen: 50
   num_heads: 1
   num_blocks: 2
   embedding_dim: 64
   dropout_rate: 0.4

   # 训练参数
   epochs: 300
   batch_size: 512
   optimizer: adam
   lr: 5.e-4
   weight_decay: 1.e-8

   # 评估指标
   monitors: [LOSS, HitRate@1, HitRate@5, HitRate@10, NDCG@5, NDCG@10]
   which4best: NDCG@10

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 字段
     - 说明
   * - ``root`` / ``dataset``
     - 数据根目录和处理后的数据集名称
   * - ``epochs``
     - 训练轮数
   * - ``batch_size``
     - 批大小
   * - ``optimizer``
     - 优化器（``adam``、``sgd`` 等）
   * - ``lr`` / ``weight_decay``
     - 学习率和权重衰减
   * - ``monitors``
     - 训练过程中监控的指标列表
   * - ``which4best``
     - 用于选择最佳模型的指标

单机训练
--------

.. code-block:: bash

   python main.py --config=config.yaml

命令行参数会覆盖配置文件中的同名设置，方便临时调整：

.. code-block:: bash

   python main.py --config=config.yaml --lr=1e-3 --epochs=500

分布式训练
----------

使用 PyTorch 的 ``torchrun`` 进行多 GPU 训练：

.. code-block:: bash

   torchrun --nproc_per_node=4 main.py --config=config.yaml

``--nproc_per_node`` 指定使用的 GPU 数量。

超参数搜索
----------

使用 ``freerec tune`` 进行网格搜索：

.. code-block:: bash

   freerec tune <experiment_name> tune_config.yaml

搜索配置文件格式：

.. code-block:: yaml

   command: python main.py
   envs:
     root: ./data
     dataset: Amazon2014Beauty_550_LOU
     device: '0,1,2,3'
   params:
     seed: [0, 1, 2, 3, 4]
   defaults:
     config: configs/Amazon2014Beauty_550_LOU.yaml

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 字段
     - 说明
   * - ``command``
     - 单次运行的命令
   * - ``envs``
     - 环境变量（数据路径、设备等）
   * - ``params``
     - 待搜索的超参数及其候选值
   * - ``defaults``
     - 所有运行共享的默认参数
