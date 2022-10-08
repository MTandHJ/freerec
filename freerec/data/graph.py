

from typing import Union
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T

from .fields import SparseField, SparseToken
from .datasets import RecDataSet
from ..utils import timemeter


@timemeter("dict2graph")
def dict2graph(
    datapipe: RecDataSet, 
    src: Union[SparseField, SparseToken], 
    dst: Union[SparseField, SparseToken],
    mode: str = 'train'
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
    mode: 'train' (default) |'valid'|'test'
        Graph from trainset or validset or testset

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
    getattr(datapipe, mode)()
    data = {src.name: np.empty((0, 1), dtype=np.int32), dst.name: np.empty((0, 1), dtype=np.int32)}
    for chunk in datapipe:
        for key, vals in data.items():
            data[key] = np.concatenate((vals, chunk[key]), axis=0)
    u, v = torch.from_numpy(data[src.name]).flatten(), torch.from_numpy(data[dst.name]).flatten()
    graph = HeteroData()
    graph[src.name] = torch.empty((src.count, 0), dtype=torch.long)
    graph[dst.name] = torch.empty((dst.count, 0), dtype=torch.long)
    graph[src.name, 'towards', dst.name].edge_index = torch.stack((u, v), dim=0).contiguous()
    graph = graph.to_homogeneous()
    graph.edge_index = to_undirected(graph.edge_index)
    return graph.to_homogeneous()

 
