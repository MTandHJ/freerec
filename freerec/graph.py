

from typing import Optional, Tuple

import torch
from torch_geometric.utils import coalesce, \
                scatter, spmm, \
                to_undirected, to_edge_index, \
                add_remaining_self_loops, remove_self_loops, \
                k_hop_subgraph, \
                dropout_node, dropout_edge, dropout_path
from torch_geometric.utils.num_nodes import maybe_num_nodes


__all__ = [
    'coalesce', 
    'scatter', 'spmm',
    'to_undirected', 'to_edge_index',
    'add_self_loops', 'remove_self_loops',
    'k_hop_subgraph',
    'dropout_node', 'dropout_edge', 'dropout_path',
    'to_adjacency', 'to_normalized',
    'get_knn_graph'
]


add_self_loops = add_remaining_self_loops

def to_adjacency(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
):
    r"""
    Get adjacency matrix.

    Parameters:
    -----------
    edge_index: torch.Tensor, (2, N)
    edge_weight: torch.Tensor, optional
        `None`: edge_weight will be set to 1
    num_nodes: int, optional
        The number of nodes.
    
    Returns:
    --------
    Adjacency matrix: CSR sparse tensor
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if edge_weight is None:
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
    
    return torch.sparse_coo_tensor(
        edge_index.clone(),
        edge_weight.clone(),
        size=(num_nodes, num_nodes)
    ).to_sparse_csr()

def to_normalized(
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    normalization: str = 'sym',
) -> Tuple[torch.Tensor]:
    r"""
    Symmetric sqrt | Left-side | Right-side normalization.

    Parameters:
    -----------
    edge_index: torch.Tensor, (2, N)
    edge_weight: torch.Tensor, optional
        `None`: edge_weight will be set to 1
    normalization: str, optional
        `sym`: Symmetric sqrt normalization
            :math: `\mathbf{\tilde{A}} = \mathbf{D}_l^{-1/2} \mathbf{A} \mathbf{D}_r^{-1/2}`
        'left': Left-side normalization
            :math: `\mathbf{\tilde{A}} = \mathbf{D}_l^{-1} \mathbf{A}`
        'right': Right-side normalization
            :math: `\mathbf{\tilde{A}} = \mathbf{A} \mathbf{D}_r^{-1}`
    
    Returns:
    --------
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    """
    row, col = edge_index[0], edge_index[1]
    if edge_weight is None:
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)

    adj = torch.sparse_coo_tensor(
        edge_index, values=edge_weight
    )

    row_sum = torch.sparse.sum(adj, dim=1).to_dense()
    col_sum = torch.sparse.sum(adj, dim=0).to_dense()

    if normalization == 'sym':
        row_inv_sqrt = row_sum.pow(-0.5)
        col_inv_sqrt = col_sum.pow(-0.5)
        row_inv_sqrt.masked_fill_(torch.isinf(row_inv_sqrt), 0.)
        col_inv_sqrt.masked_fill_(torch.isinf(col_inv_sqrt), 0.)
        edge_weight = row_inv_sqrt[row] * edge_weight * col_inv_sqrt[col]
    elif normalization == 'left':
        row_inv = row_sum.pow(-1.)
        row_inv.masked_fill_(torch.isinf(row_inv), 0.)
        edge_weight = row_inv[row] * edge_weight
    elif normalization == 'right':
        col_inv = col_sum.pow(-1.)
        col_inv.masked_fill_(torch.isinf(col_inv), 0.)
        edge_weight = edge_weight * col_inv[col]
    else:
        raise NotImplementedError(
            f"Normalization should be in 'sym', 'left' or 'right' ..."
        )

    return edge_index, edge_weight

def get_knn_graph(
    sim_mat: torch.Tensor, k: int,
    symmetric: bool = True,
    reduce: str = 'max'
):
    r"""
    Get kNN graph.

    Parameters:
    -----------
    sim_mat: torch.Tensor
    k: int,
        top-K for each row
    symmetric: bool, default to `True`
        `True`: Return undirected edge weight
    reduce: str 
        The reduce operation to use for merging edge

    Returns:
    --------
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    """
    M, N = sim_mat.shape
    device = sim_mat.device
    if sim_mat.is_sparse:
        sim_mat = sim_mat.to_dense()
    vals, cols = torch.topk(sim_mat, k, dim=1, largest=True)
    del sim_mat

    rows = torch.arange(0, M, device=device).unsqueeze(-1).repeat(1, k)
    rows, cols = rows.flatten(), cols.flatten()
    edge_index = torch.stack(
        (rows, cols), dim=0
    )
    edge_weight = vals.flatten()

    if symmetric:
        assert M == N, "`symmetric == True` but `sim_mat` is not a square matrix ..."
        edge_index, edge_weight = to_undirected(edge_index, edge_attr=edge_weight, reduce=reduce)
    
    return edge_index, edge_weight