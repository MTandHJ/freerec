快速上手
========

本节演示一个完整的推荐模型训练流程：准备数据、运行模型、查看结果。

第一步：准备数据集
------------------

使用 ``freerec make`` 对原始数据进行拆分与过滤。以 MovieLens1M 为例，采用 Leave-one-out (LOU) 拆分：

.. code-block:: bash

   freerec make MovieLens1M --root ./data --star4pos 0 --kcore4user 5 --kcore4item 5 --splitting LOU

处理完成后，数据集保存在 ``data/Processed/MovieLens1M_550_LOU/``。

第二步：运行模型
----------------

以 SASRec 为例，创建配置文件 ``config.yaml``：

.. code-block:: yaml

   root: ./data
   dataset: MovieLens1M_550_LOU

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

然后运行训练：

.. code-block:: bash

   python main.py --config=config.yaml

第三步：分布式训练（可选）
--------------------------

使用 ``torchrun`` 进行多卡训练：

.. code-block:: bash

   torchrun --nproc_per_node=4 main.py --config=config.yaml

第四步：超参搜索（可选）
------------------------

使用 ``freerec tune`` 自动搜索超参数：

.. code-block:: bash

   freerec tune MyExperiment tune_config.yaml

其中 ``tune_config.yaml`` 定义搜索空间：

.. code-block:: yaml

   command: python main.py
   envs:
     root: ./data
     dataset: MovieLens1M_550_LOU
     device: '0,1,2,3'
   params:
     seed: [0, 1, 2, 3, 4]
   defaults:
     config: config.yaml

训练流程总览
------------

.. mermaid::

   graph LR
       subgraph setup ["1 · 初始化"]
           direction TB
           dataset["RecDataSet(root, dataset)"]
           model["Model(dataset, ...)"]
           pipe["model.sure_trainpipe()\nmodel.sure_validpipe()\nmodel.sure_testpipe()"]
           dataset --> model --> pipe
       end

       subgraph coach_init ["2 · 组装"]
           direction TB
           coach["Coach(\n  dataset, trainpipe,\n  validpipe, testpipe,\n  model, cfg\n)"]
       end

       subgraph fit ["3 · 训练  coach.fit()"]
           direction TB
           train["train_per_epoch\ndata → model → loss\n→ backward → step"]
           eval["evaluate\nmodel.recommend()\n→ metrics"]
           ckpt["check_best\nsave if improved"]
           train --> eval --> ckpt
           ckpt -.->|"next epoch"| train
       end

       subgraph out ["4 · 输出"]
           direction TB
           summary["SUMMARY.md"]
           tb["TensorBoard logs"]
           plots["[metric].png"]
       end

       setup --> coach_init --> fit --> out

   style setup fill:#fff8f0,stroke:#e07020,stroke-width:2px
   style coach_init fill:#fff8f0,stroke:#e07020,stroke-width:2px
   style fit fill:#fff8f0,stroke:#e07020,stroke-width:2px,stroke-dasharray: 6 3
   style out fill:#fff8f0,stroke:#e07020,stroke-width:2px
   style coach fill:#e07020,stroke:#b85a10,color:#fff
