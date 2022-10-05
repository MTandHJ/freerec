

from typing import Union
import numpy as np
import dgl
import torch

from .fields import SparseField, SparseToken
from .datasets import RecDataSet
from ..utils import timemeter


@timemeter("dict2graph")
def dict2graph(
    datapipe: RecDataSet, 
    src: Union[SparseField, SparseToken], 
    dst: Union[SparseField, SparseToken]
):
    """Convert dict flows to a homogeneous graph.

    Parameters:
    ---

    datapipe: RecDataSet
        RecDataSet yielding dict flows.
    src: SparseField or SparseToken
        Srouce Node.
    dst: SparseField or SparseToken
        Destination Node.

    Notes:
    ---

    1. The indices of source nodes are from 0 to src.count - 1.
    2. The indices of destination nodes are from src.count to src.count + dst.count - 1

    Examples:
    ---

    >>> from freerec.data import MovieLens1M
    >>> from freerec.data.tags import USER, ITEM, ID
    >>> basepipe = MovieLens1M("../../data/MovieLens1M")
    >>> User = basepipe.fields.whichis(USER, ID)
    >>> Item = basepipe.fields.whichis(Item, ID)
    >>> g = dict2graph(basepipe, USER, ITEM)
    """
    data = {src.name: np.empty((0, 1), dtype=np.int32), dst.name: np.empty((0, 1), dtype=np.int32)}
    for chunk in datapipe:
        for key, vals in data.items():
            data[key] = np.concatenate((vals, chunk[key]), axis=0)
    u, v = torch.from_numpy(data[src.name]), torch.from_numpy(data[dst.name])
    v = v + src.count
    g = dgl.graph(
        (u.flatten(), v.flatten()), num_nodes=src.count + dst.count, row_sorted=True
    )
    return dgl.to_bidirected(g).int() # int32

 
