配置系统 (Parser)
==================

FreeRec 使用 :class:`~freerec.parser.Parser` 统一管理命令行参数和 YAML 配置文件。

基本用法
--------

.. code-block:: python

   import freerec

   cfg = freerec.parser.Parser()

   # 添加自定义参数
   cfg.add_argument("--embedding-dim", type=int, default=64)
   cfg.add_argument("--dropout-rate", type=float, default=0.2)

   # 设置默认值
   cfg.set_defaults(
       description="MyModel",
       root="./data",
       dataset="Amazon2014Beauty_550_LOU",
       epochs=200,
       batch_size=256,
       optimizer='adam',
       lr=1e-3,
   )

   # 编译配置（解析参数、初始化设备、创建日志目录等）
   cfg.compile()

YAML 配置文件
--------------

通过 ``--config`` 指定 YAML 文件，文件中的值会作为默认值加载：

.. code-block:: yaml

   root: ./data
   dataset: Amazon2014Beauty_550_LOU
   embedding_dim: 64
   dropout_rate: 0.2
   epochs: 200
   batch_size: 256
   optimizer: adam
   lr: 1.e-3
   weight_decay: 1.e-4
   monitors: [LOSS, HitRate@10, NDCG@10]
   which4best: NDCG@10

.. code-block:: bash

   python main.py --config=config.yaml

**优先级规则：** 命令行参数 > YAML 文件 > ``set_defaults``

.. code-block:: bash

   # YAML 中 lr=1e-3，命令行覆盖为 1e-2
   python main.py --config=config.yaml --lr=1e-2

内置参数一览
------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - 参数
     - 默认值
     - 说明
   * - ``--root``
     - ``"."``
     - 数据根目录
   * - ``--dataset``
     - ``"RecDataSet"``
     - 数据集名称
   * - ``--config``
     - ``None``
     - YAML 配置文件路径
   * - ``--tasktag``
     - ``None``
     - 任务类型：``MATCHING`` / ``NEXTITEM`` / ``PREDICTION``
   * - ``--ranking``
     - ``"full"``
     - 评估方式：``full``（全量排序）/ ``pool``（候选池排序）
   * - ``--epochs``
     - ``None``
     - 训练轮数
   * - ``--batch-size`` (``-b``)
     - ``None``
     - 批大小
   * - ``--optimizer``
     - ``"adam"``
     - 优化器：``adam`` / ``sgd`` / ``adamw``
   * - ``--lr`` (``-lr``)
     - ``None``
     - 学习率
   * - ``--weight-decay`` (``-wd``)
     - ``None``
     - 权重衰减
   * - ``--momentum`` (``-mom``)
     - ``0.9``
     - SGD 动量
   * - ``--beta1``
     - ``0.9``
     - Adam beta1
   * - ``--beta2``
     - ``0.999``
     - Adam beta2
   * - ``--device``
     - 自动检测
     - 计算设备
   * - ``--num-workers``
     - ``4``
     - DataLoader 工作进程数
   * - ``--seed``
     - ``1``
     - 随机种子
   * - ``--eval-freq``
     - ``5``
     - 每隔多少 epoch 评估一次
   * - ``--eval-valid``
     - ``True``
     - 是否评估验证集
   * - ``--eval-test``
     - ``False``
     - 是否评估测试集（默认仅在最后评估）
   * - ``--early-stop-patience`` (``-esp``)
     - 很大
     - 早停耐心值
   * - ``--resume``
     - ``False``
     - 是否从 checkpoint 恢复
   * - ``--description`` (``-m``)
     - ``"RecSys"``
     - 实验描述（用于日志路径）
   * - ``--benchmark``
     - ``False``
     - 启用 cuDNN benchmark 模式

compile() 做了什么
-------------------

调用 ``cfg.compile()`` 会依次执行：

1. 加载 YAML 配置文件（如指定了 ``--config``）
2. 生成日志和 checkpoint 路径
3. 初始化分布式训练（如有多卡）
4. 配置设备、日志、随机种子
5. 在日志目录写入 ``README.md``（保存当前配置快照）

生成的目录结构：

.. code-block:: text

   ./logs/{description}/{dataset}/{id}/
   ├── README.md         # 配置快照
   ├── log.txt           # 训练日志
   ├── data/
   │   ├── monitors.pkl  # 指标历史记录
   │   └── best.pkl      # 最佳测试结果
   ├── summary/
   │   ├── SUMMARY.md    # 指标汇总表
   │   └── *.png         # 指标曲线图
   ├── model.pt          # 最终模型权重
   └── best.pt           # 最佳模型权重

   ./infos/{description}/{dataset}/{device}/
   └── checkpoint.tar    # 训练断点（仅用于 --resume 恢复）
