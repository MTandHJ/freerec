



from typing import TypeVar, Callable

import torch
import tqdm


T = TypeVar('T')

def safe_cast(val: T, dest_type: Callable[[T], T], default: T) -> T:
    try:
        if val:
            return dest_type(val)
        else: # fill_na
            return default
    except ValueError:
        raise ValueError(f"Using '{dest_type.__name__}' to convert '{val}' where the default value is {default}")


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, datapipe, num_workers: int = 0, **kwargs) -> None:
        def collate_fn(batch): return batch[0]
        super().__init__(
            dataset=datapipe, num_workers=num_workers, 
            batch_size=1, collate_fn=collate_fn, **kwargs
        )


class TQDMDataLoader(DataLoader):
    def __iter__(self):
        return iter(
            tqdm(
                super(TQDMDataLoader, self).__iter__(), 
                leave=False, desc="վ'ᴗ' ի-"
            )
        )


