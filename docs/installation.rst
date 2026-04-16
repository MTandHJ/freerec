安装指南
========

环境要求
--------

- Python >= 3.9
- `PyTorch >= 2.0 <https://pytorch.org/get-started/locally/>`_

安装步骤
--------

.. code-block:: bash

   # 1. 安装 PyTorch（根据你的 CUDA 版本选择）
   #    参考 https://pytorch.org/get-started/locally/
   pip install torch

   # 2. 安装 FreeRec
   pip install freerec

   # 3. 安装 torchdata（FreeRec 依赖的数据管道库）
   freerec setup

.. note::

   FreeRec 依赖 torchdata 0.7.0，因为更新版本不再支持 datapipe 功能。
   ``freerec setup`` 会以 ``--no-deps`` 方式安装，避免覆盖你已有的 PyTorch。

可选依赖
--------

根据需要安装额外功能：

.. code-block:: bash

   pip install freerec[graph]    # torch-geometric，用于图推荐模型
   pip install freerec[metrics]  # scikit-learn，用于 ROC-AUC 等指标
   pip install freerec[nn]       # einops，用于注意力模块
   pip install freerec[all]      # 以上全部
