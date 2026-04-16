序列推荐
========

序列推荐模型基于用户的历史交互序列预测下一个可能感兴趣的物品。
FreeRec 中的序列推荐模型继承自 :class:`~freerec.models.base.SeqRecArch`。

SASRec
------

**Self-Attentive Sequential Recommendation** — 基于自注意力机制，
通过 Transformer 编码器捕捉用户行为序列中的长短期依赖关系。

.. code-block:: bash

   # 全量排序
   python main.py --config=configs/xxx.yaml --ranking=full
   # 采样排序
   python main.py --config=configs/xxx.yaml --ranking=pool

.. literalinclude:: ../../examples/SASRec/main.py
   :language: python
   :caption: examples/SASRec/main.py
   :linenos:

**代码要点：**

- **配置解析** (L11-30)：通过 ``freerec.parser.Parser`` 定义超参数，``cfg.compile()`` 完成解析
- **模型定义** (L54-231)：继承 ``SeqRecArch``，核心是多层自注意力 + Position Embedding
- **数据管道** (L142-154)：``sure_trainpipe`` 定义训练数据流水线——打乱序列 → 正样本生成 → 负采样 → 偏移填充 → 批处理 → 张量化
- **前向传播** (L191-216)：``fit()`` 方法计算损失，支持 BCE / BPR / CE 三种损失函数
- **推荐** (L218-231)：``recommend_from_full`` 对全量物品打分，``recommend_from_pool`` 对候选池打分
- **训练循环** (L234-249)：``CoachForSASRec`` 继承 ``Coach``，实现 ``train_per_epoch``
- **主函数** (L252-283)：加载数据集 → 创建模型 → 构建数据管道 → 组装 Coach → 训练

BERT4Rec
--------

**BERT-based Sequential Recommendation** — 采用双向 Transformer 编码器
和 Masked Item Prediction 任务进行序列推荐。与 SASRec 的单向注意力不同，
BERT4Rec 使用随机遮蔽策略实现双向建模。

.. code-block:: bash

   python main.py --config=configs/xxx.yaml --ranking=full

.. literalinclude:: ../../examples/BERT4Rec/main.py
   :language: python
   :caption: examples/BERT4Rec/main.py
   :linenos:

**代码要点：**

- **遮蔽策略** ：``NUM_PADS = 2`` (0=padding, 1=mask)，``random_mask`` 方法以概率 ``mask_ratio`` 随机遮蔽序列中的物品
- **训练管道** ：与 SASRec 不同，BERT4Rec 不需要正/负样本拆分——直接对整个序列做遮蔽预测
- **验证/测试管道** ：在序列末尾添加 MASK token，预测最后一个位置
- **损失函数** ：固定使用 CrossEntropy，在遮蔽位置上计算分类损失

GRU4Rec
-------

**Session-based Recommendations with RNNs** — 基于 GRU 的会话推荐模型，
通过循环神经网络建模用户行为序列。结构简单但效果稳定。

.. code-block:: bash

   python main.py --config=configs/xxx.yaml --ranking=full

.. literalinclude:: ../../examples/GRU4Rec/main.py
   :language: python
   :caption: examples/GRU4Rec/main.py
   :linenos:

**代码要点：**

- **模型结构** ：Embedding → Dropout → GRU → Linear，使用最后一个有效隐状态作为用户表示
- **训练管道** ：使用 ``shuffled_roll_seqs_source`` 生成滚动子序列，只以最后一个物品为目标
- **序列压缩** ：``shrink_pads`` 去除全 padding 的列，提高 GRU 效率
- **隐状态提取** ：通过 ``gather`` 根据实际序列长度取最后一个非 padding 位置的输出
