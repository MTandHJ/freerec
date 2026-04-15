

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
    r"""Build a CSR sparse adjacency matrix from an edge index.

    Parameters
    ----------
    edge_index : :class:`torch.Tensor`
        Edge index of shape ``(2, E)`` where ``E`` is the number of edges.
    edge_weight : :class:`torch.Tensor`, optional
        Edge weights of shape ``(E,)``.  When ``None``, all weights are
        set to 1.
    num_nodes : int, optional
        Number of nodes.  Inferred from *edge_index* when ``None``.

    Returns
    -------
    :class:`torch.Tensor`
        Sparse CSR adjacency matrix of shape ``(num_nodes, num_nodes)``.
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
    r"""Normalize edge weights using degree-based normalization.

    Supports symmetric square-root, left-side, and right-side normalization:

    .. math::

        \text{sym:}   \quad \tilde{\mathbf{A}} = \mathbf{D}_l^{-1/2}\,\mathbf{A}\,\mathbf{D}_r^{-1/2}

        \text{left:}  \quad \tilde{\mathbf{A}} = \mathbf{D}_l^{-1}\,\mathbf{A}

        \text{right:} \quad \tilde{\mathbf{A}} = \mathbf{A}\,\mathbf{D}_r^{-1}

    Parameters
    ----------
    edge_index : :class:`torch.Tensor`
        Edge index of shape ``(2, E)``.
    edge_weight : :class:`torch.Tensor`, optional
        Edge weights of shape ``(E,)``.  When ``None``, all weights are
        set to 1.
    normalization : str, optional
        Normalization type.  One of ``'sym'`` (default), ``'left'``, or
        ``'right'``.

    Returns
    -------
    edge_index : :class:`torch.Tensor`
        The (unchanged) edge index.
    edge_weight : :class:`torch.Tensor`
        Normalized edge weights.

    Raises
    ------
    NotImplementedError
        If *normalization* is not one of the supported values.
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
    r"""Construct a k-nearest-neighbor graph from a similarity matrix.

    For each row of *sim_mat* the top-*k* entries are selected.  Optionally
    the resulting directed graph is symmetrized into an undirected one.

    Parameters
    ----------
    sim_mat : :class:`torch.Tensor`
        Similarity matrix of shape ``(M, N)``.  May be dense or sparse
        COO (converted to dense internally).
    k : int
        Number of nearest neighbors per node.
    symmetric : bool, optional
        If ``True`` (default), convert to an undirected graph using
        :func:`torch_geometric.utils.to_undirected`.  Requires ``M == N``.
    reduce : str, optional
        Reduce operation passed to :func:`torch_geometric.utils.to_undirected`
        for merging duplicate edges.  Default is ``'max'``.

    Returns
    -------
    edge_index : :class:`torch.Tensor`
        Edge index of shape ``(2, E)``.
    edge_weight : :class:`torch.Tensor`
        Corresponding edge weights of shape ``(E,)``.
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
