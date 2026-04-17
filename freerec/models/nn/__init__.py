import torch.nn as nn

from freerec.models.nn.attn import ScaledDotProductAttention
from freerec.models.nn.ffn import FeedForwardNetwork
from freerec.models.nn.utils import Unsqueeze
