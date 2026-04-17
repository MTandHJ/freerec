训练输出
========

训练完成后，FreeRec 会在 ``logs/`` 和 ``infos/`` 目录下生成完整的实验记录。

目录结构
--------

.. code-block:: text

   ./logs/{description}/{dataset}/{id}/
   ├── README.md                   # 配置快照
   ├── log.txt                     # 完整训练日志
   ├── model.pt                    # 最终模型权重
   ├── best.pt                     # 最佳模型权重
   ├── data/
   │   ├── monitors.pkl            # 所有指标的历史记录
   │   └── best.pkl                # 最佳模型的测试结果
   └── summary/
       ├── SUMMARY.md              # 指标汇总表
       ├── validNDCG@10.png        # 验证集 NDCG@10 曲线
       ├── validHitRate@10.png     # 验证集 HitRate@10 曲线
       ├── trainLOSS.png           # 训练损失曲线
       └── (TensorBoard 事件文件)

   ./logs/{description}/core/          # freerec tune 专用
   ├── results.json                    # 所有子实验的聚合结果
   ├── README.md                       # 配置快照
   └── log.txt                         # 调参协调日志

   ./infos/{description}/{dataset}/{device}/
   └── checkpoint.tar              # 训练断点（仅用于 --resume 恢复）

各文件说明
----------

README.md
^^^^^^^^^

配置快照，记录了本次实验的所有参数设置，便于复现。

log.txt
^^^^^^^

完整的训练日志，包括每个 epoch 的损失值、评估指标、最佳指标更新记录等。

SUMMARY.md
^^^^^^^^^^

训练结束后自动生成的 Markdown 汇总表，包含各指标的最佳值及其对应的 epoch：

.. code-block:: text

   | Mode  | Metric     | Best   | @Step |
   |-------|------------|--------|-------|
   | valid | NDCG@10    | 0.0512 | 150   |
   | valid | HitRate@10 | 0.0893 | 150   |
   | test  | NDCG@10    | 0.0498 | 150   |

指标曲线图 (\*.png)
^^^^^^^^^^^^^^^^^^^^

每个被监控的指标生成一张 PNG 曲线图（横轴为 epoch，纵轴为指标值），
文件名格式为 ``{mode}{METRIC}.png``，例如 ``validNDCG@10.png``。

monitors.pkl
^^^^^^^^^^^^

所有指标的完整历史记录（pickle 格式），三层嵌套结构：

**第一层** — 模式：``train`` / ``valid`` / ``test``

**第二层** — 指标族：``LOSS`` / ``NDCG`` / ``HITRATE`` 等（不含 @K）

**第三层** — 具体指标：``NDCG@5`` / ``NDCG@10`` / ``HITRATE@1`` 等（含 @K），值为历史列表。对于无 K 的指标如 ``LOSS``，键重复为 ``LOSS -> LOSS``。

.. code-block:: python

   {
       'train': {
           'LOSS': {
               'LOSS': [0.5, 0.4, 0.35, ...],
           },
           'HITRATE': {
               'HITRATE@1': [],    # train 模式下未评估则为空
               'HITRATE@5': [],
               'HITRATE@10': [],
           },
           'NDCG': {
               'NDCG@5': [],
               'NDCG@10': [],
           },
       },
       'valid': {
           'LOSS': {'LOSS': []},
           'HITRATE': {
               'HITRATE@1': [0.0099, 0.0123, ...],
               'HITRATE@5': [0.0407, 0.0452, ...],
               'HITRATE@10': [0.0654, 0.0712, ...],
           },
           'NDCG': {
               'NDCG@5': [0.0253, 0.0289, ...],
               'NDCG@10': [0.0332, 0.0371, ...],
           },
       },
       'test': { ... },   # 结构同 valid
   }

可用于自定义可视化：

.. code-block:: python

   import pickle
   with open("logs/.../data/monitors.pkl", "rb") as f:
       monitors = pickle.load(f)
   ndcg_history = monitors['valid']['NDCG']['NDCG@10']

best.pkl
^^^^^^^^

最佳 checkpoint 对应的测试集评估结果，
用于超参搜索场景下的 TensorBoard 对比：

.. code-block:: python

   import pickle
   with open("logs/.../data/best.pkl", "rb") as f:
       best = pickle.load(f)
   # best['best']['NDCG@10'] -> 0.0498

model.pt / best.pt
^^^^^^^^^^^^^^^^^^^

均保存在 ``logs/`` 目录下（不在 ``infos/``）：

- ``model.pt`` — 训练结束时的最终模型权重
- ``best.pt`` — 验证集指标最优时保存的模型权重

.. code-block:: python

   import torch
   state_dict = torch.load("logs/.../best.pt")
   model.load_state_dict(state_dict)

checkpoint.tar
^^^^^^^^^^^^^^

保存在 ``infos/`` 目录下，仅用于 ``--resume`` 恢复训练。包含：

.. code-block:: python

   {
       'epoch': 150,
       'model': model.state_dict(),
       'optimizer': optimizer.state_dict(),
       'lr_scheduler': lr_scheduler.state_dict(),
       'monitors': monitors.state_dict()
   }

.. code-block:: bash

   python main.py --config=config.yaml --resume

TensorBoard
^^^^^^^^^^^^

指标历史会同时写入 TensorBoard 格式，可用标准工具查看：

.. code-block:: bash

   tensorboard --logdir ./logs/{description}

results.json（freerec tune）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``freerec tune`` 完成每个子实验后，会将参数和指标追加写入
``logs/{description}/core/results.json``，用于下游 leaderboard 聚合。

.. code-block:: json

   {
       "description": "SASRec",
       "dataset": "Amazon2014Beauty_550_LOU",
       "timestamp": "2026-04-17T10:30:00",
       "runs": [
           {
               "id": "0417103000",
               "params": {"seed": 0, "lr": 5e-4, "embedding_dim": 64},
               "metrics": {
                   "train": {"LOSS": 0.123},
                   "valid": {"NDCG@10": 0.456, "HitRate@10": 0.789},
                   "test":  {"NDCG@10": 0.432, "HitRate@10": 0.765},
                   "best":  {"NDCG@10": 0.445, "HitRate@10": 0.778}
               }
           }
       ]
   }

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 字段
     - 说明
   * - ``description``
     - 实验描述
   * - ``dataset``
     - 数据集名称
   * - ``timestamp``
     - 最近一次写入时间（ISO 8601）
   * - ``runs[].id``
     - 子实验 ID（时间戳格式）
   * - ``runs[].params``
     - 网格搜索变量（含 seed）
   * - ``runs[].metrics``
     - 各模式下的最佳指标（train/valid/test/best）

多次 ``freerec tune`` 的结果会追加到同一文件。
