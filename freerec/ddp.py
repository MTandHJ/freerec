

import os
import torch.distributed as dist


def is_distributed() -> bool:
    world_size = os.environ.get('WORLD_SIZE', None)
    # world_size > 1 will be better?
    return world_size is not None

def is_primary_process() -> bool:
    """Checks if the current process is the primary process"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True

def primary_process_only(func):
    def wrapper(*args, **kwargs):
        if is_primary_process():
            return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__func__ = func.__doc__
    return wrapper