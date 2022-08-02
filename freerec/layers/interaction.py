



import torch
import torch.nn as nn
from torchrec.modules.deepfm import FactorizationMachine


__all__ = ['FM']

class FM(FactorizationMachine):
    """
    1. List[Tensor: B x k x d] -> Tensor: B x [k1 x d1 + k2 x d2 ...] -> Tensor: B x 1
    2. List[Tensor: B x d] -> Tensor: B x [d1 + d2 ...] -> Tensor: B x 1
    """
