# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

from typing import List, TypeVar, Dict

import torch
import numpy as np
import os, pickle, functools, warnings
import torch.distributed as dist


_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""

T = TypeVar("T")


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if dist.is_available() and dist.is_initialized():
        assert _LOCAL_PROCESS_GROUP is not None
        return dist.get_rank(group=_LOCAL_PROCESS_GROUP)
    return 0

def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)
    return 1

def is_distributed() -> bool:
    world_size = os.environ.get('WORLD_SIZE', None)
    # world_size > 1 will be better?
    return world_size is not None

def is_main_process() -> bool:
    """Checks if the current process is the main process"""
    return get_rank() == 0

def main_process_only(func):
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__func__ = func.__doc__
    return wrapper

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if get_world_size() == 1:
        return
    dist.barrier()

@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD

def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        warnings.warn(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor

def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor(
        [tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor

def all_gather(data: T, group=None) -> List[T]:
    r"""
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    
    Parameters:
    -----------
    data: any picklable object
    group: a torch process group. By default, will use a group which
        contains all ranks on gloo backend.

    Returns:
    --------
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]

    device = None
    if isinstance(data, torch.Tensor):
        device = data.device
        data = data.cpu()
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    if device:
        data_list = [data.to(device) for data in data_list]
    return data_list

def gather(data: T, dst: int = 0, group=None) -> List[T]:
    r"""
    Run gather on arbitrary picklable data (not necessarily tensors).

    Parameters:
    -----------
    data: any picklable object
    dst (int): destination rank
    group: a torch process group. By default, will use a group which
        contains all ranks on gloo backend.

    Returns:
    --------
    list[data]: on dst, a list of data gathered from each rank. Otherwise,
        an empty list.
    """
    if get_world_size() == 1:
        return [data]

    device = None
    if isinstance(data, torch.Tensor):
        device = data.device
        data = data.cpu()
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    device = tensor.device
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=device)
            for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        if device:
            data_list = [data.to(device) for data in data_list]
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []

def shared_random_seed() -> int:
    r"""
    Returns:
    --------
    int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    Notes:
    ------
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]

def reduce_dict(input_dict: Dict, average: bool = True) -> Dict:
    r"""
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Parameters:
    -----------
    input_dict: dict
        inputs to be reduced. (values not necessarily tensors).
    average: bool, default to `True`
        whether to do average or sum

    Returns:
    --------
        a dict with the same keys as input_dict, after reduction.
    """

    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():

        # Convert to CUDA Tensor for dist.reduce()
        input_dict_cuda_vals = {}
        for k, v in input_dict.items():
            if type(v) == torch.Tensor:
                input_dict_cuda_vals[k] = v.to('cuda')
            else:
                input_dict_cuda_vals[k] = torch.tensor(v, device='cuda')

        names = []
        values = []
        for k, v in sorted(input_dict_cuda_vals.items()):
            names.append(k)
            values.append(v)
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0) # reduce to gpu 0

        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict