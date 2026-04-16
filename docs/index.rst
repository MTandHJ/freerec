FreeRec
=======

.. image:: _static/img/logo_pixel.png
   :align: center
   :alt: FreeRec Logo

|

FreeRec 是一个基于 PyTorch 的推荐系统库，提供从数据预处理到模型训练的全流程支持。

.. code-block:: bash

   pip install freerec

核心特性
--------

- **数据处理流水线** — 内置 30+ 公开数据集，一键完成拆分与过滤
- **模型架构基类** — 通用推荐、序列推荐、评分预测等基类，快速搭建模型
- **丰富的评估指标** — Precision、Recall、NDCG、AUC 等 20+ 指标
- **灵活的训练框架** — YAML 配置驱动，支持分布式训练与超参搜索
- **可选图神经网络** — 集成 PyG，支持图推荐模型（如 LightGCN）

数据流水线
----------

.. mermaid::

   graph LR
       raw["📄 Raw Files\n.inter / .user / .item"]
       make["⚙️ freerec make\nfilter & split"]
       chunks["📦 Chunks\ntrain / valid / test .pkl"]

       raw --> make --> chunks

       chunks --> dataset["RecDataSet"]

       dataset -->|"fields\n(USER·ID, ITEM·ID, ...)"| source

       subgraph datapipe ["Postprocessor Chain"]
           direction LR
           source["Source\nordered / shuffled\n/ choiced"]
           sampler["Sampler\ntrain_pos · train_neg\nvalid · test"]
           rowop["Row Ops\nlpad · rpad\nlprune · add"]
           batch["batch_"]
           tensor["tensor_"]

           source --> sampler --> rowop --> batch --> tensor
       end

       tensor --> trainpipe["trainpipe"]
       tensor --> validpipe["validpipe"]
       tensor --> testpipe["testpipe"]

   style raw fill:#fef3e2,stroke:#e07020,color:#333
   style make fill:#fef3e2,stroke:#e07020,color:#333
   style chunks fill:#fef3e2,stroke:#e07020,color:#333
   style dataset fill:#e07020,stroke:#b85a10,color:#fff
   style datapipe fill:#fff8f0,stroke:#e07020,stroke-width:2px
   style trainpipe fill:#e07020,stroke:#b85a10,color:#fff
   style validpipe fill:#e07020,stroke:#b85a10,color:#fff
   style testpipe fill:#e07020,stroke:#b85a10,color:#fff

训练流程
--------

.. mermaid::

   graph TB
       subgraph inputs ["输入组件"]
           direction TB
           dataset["RecDataSet"]
           pipes["trainpipe / validpipe / testpipe"]
           model["Model\nGenRecArch / SeqRecArch / PredRecArch"]
           optim["Optimizer + LR Scheduler"]
       end

       cfg["📝 config.yaml"] -->|compile| coach["Coach"]
       inputs --> coach

       coach -->|fit| loop

       subgraph loop ["Epoch Loop  ×N"]
           direction TB
           train["🔄 train_per_epoch\nforward → loss → backward → step"]
           eval["📊 evaluate\nrecommend → metrics"]
           best["🏆 check_best\nsave if improved"]

           train --> eval --> best
       end

       loop --> output

       subgraph output ["输出"]
           direction LR
           summary["SUMMARY.md"]
           tb["TensorBoard"]
           plots["[metric].png"]
       end

   style cfg fill:#fef3e2,stroke:#e07020,color:#333
   style inputs fill:#fff8f0,stroke:#e07020,stroke-width:2px
   style coach fill:#e07020,stroke:#b85a10,color:#fff
   style loop fill:#fff8f0,stroke:#e07020,stroke-width:2px,stroke-dasharray: 6 3
   style output fill:#fff8f0,stroke:#e07020,stroke-width:2px
   style train fill:#fef3e2,stroke:#e07020,color:#333
   style eval fill:#fef3e2,stroke:#e07020,color:#333
   style best fill:#fef3e2,stroke:#e07020,color:#333

.. toctree::
   :maxdepth: 2
   :caption: 入门

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: 教程

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: 示例

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API 参考

   api/index
