

from typing import Optional, Union
import torch
import torch.nn as nn

from ..data.fields import Tokenizer

__all__ = ['RecSysArch']


class RecSysArch(nn.Module):

    def to(
        self, device: Optional[Union[int, torch.device]] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        non_blocking: bool = False
    ):
        if device:
            self.device = device
        return super().to(device, dtype, non_blocking)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1e-4)

    @staticmethod
    def broadcast(*tensors: torch.Tensor):
        """Broadcasts the given tensors according to Broadcasting semantics.
        See [here](https://pytorch.org/docs/stable/generated/torch.broadcast_tensors.html#torch.broadcast_tensors) for details.

        Examples:
        ---

        >>> users = torch.rand(4, 1, 4)
        >>> items = torch.rand(4, 2, 1)
        >>> users, items = RecSysArch.broadcast(users, items)
        >>> users.size()
        torch.Size(4, 2, 4)
        >>> items.size()
        torch.Size(4, 2, 4)
        """
        return torch.broadcast_tensors(*tensors)