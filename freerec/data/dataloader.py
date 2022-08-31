

import torch


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, datapipe, num_workers: int = 0, **kwargs) -> None:
        def collate_fn(batch): return batch[0]
        super().__init__(
            dataset=datapipe, num_workers=num_workers, 
            batch_size=1, collate_fn=collate_fn, **kwargs
        )
