


import os
import torch.distributed as dist


def is_primary_process() -> bool:
    try:
        if os.environ['WORLD_SIZE'] >= 1 and dist.get_rank() != 0:
            return False
    except KeyError:
        return True
    return True

def primary_process_only(func):
    def wrapper(self, *args, **kwargs):
        if self.cfg.IS_PRIMARY_PROCESS:
            return func(self, *args, **kwargs)
        else:
            return 0
    wrapper.__name__ = func.__name__
    wrapper.__func__ = func.__doc__
    return wrapper