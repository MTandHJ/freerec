

import torch
# import torchdata.datapipes as dp
# from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService


def _collate_fn(batch):
    return batch[0]

class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, datapipe, num_workers: int = 0, **kwargs) -> None:
        super().__init__(
            dataset=datapipe, num_workers=num_workers, 
            batch_size=1, collate_fn=_collate_fn,
            **kwargs
        )


# TODO: torchdata==0.4.1 has the following issue:
# Multiprocessing and batching / collation:
# https://github.com/pytorch/data/issues/530
# class DataLoader(DataLoader2):

#     def __init__(
#         self, datapipe: dp.iter.IterDataPipe, 
#         num_workers: int = 0, **kwargs
#     ) -> None:
#         super().__init__(
#             datapipe=datapipe,
#             reading_service=MultiProcessingReadingService(
#                 num_workers=num_workers, **kwargs
#             )
#         )