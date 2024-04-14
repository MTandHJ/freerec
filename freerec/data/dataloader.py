

import torch
# import torchdata.datapipes as dp
# from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService

__all__ = ['DataLoader']


def _collate_fn(batch):
    return batch[0]

class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, datapipe, num_workers: int = 0, **kwargs) -> None:
        super().__init__(
            dataset=datapipe, num_workers=num_workers, 
            batch_size=1, collate_fn=_collate_fn,
            **kwargs
        )