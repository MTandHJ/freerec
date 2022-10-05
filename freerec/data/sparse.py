

from typing import Optional, Tuple

import torch
import torchdata.datapipes as dp
import scipy.sparse as sp
import pandas as pd


def get_csr_matrix(
    datapipe: dp.iter.IterDataPipe, 
    rowName: str, colName: str, valName: str,
    shape: Optional[Tuple[int, int]] = None
) -> sp.csr_array:
    """Compressed Sparse Row matrix
    Advantages of the CSR format
      - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
      - efficient row slicing
      - fast matrix vector products

    Disadvantages of the CSR format
      - slow column slicing operations (consider CSC)
      - changes to the sparsity structure are expensive (consider LIL or DOK)
    """
    df = None
    for chunk in datapipe:
        df = pd.concat((df, chunk))
    df = df[[rowName, colName, valName]]
    df = df[df[valName] > 0]
    df = df.sort_values(by=[rowName, colName])
    rows, cols, vals = df[rowName].values, df[colName].values, df[valName].values
    return sp.csr_array((vals, (rows, cols)), shape=shape)

def get_csc_matrix(
    datapipe: dp.iter.IterDataPipe, 
    rowName: str, colName: str, valName: str,
    shape: Optional[Tuple[int, int]] = None
) -> sp.csc_array:
    """Compressed Sparse Column matrix
    Advantages of the CSC format
        - efficient arithmetic operations CSC + CSC, CSC * CSC, etc.
        - efficient column slicing
        - fast matrix vector products (CSR, BSR may be faster)

    Disadvantages of the CSC format
      - slow row slicing operations (consider CSR)
      - changes to the sparsity structure are expensive (consider LIL or DOK)
    """
    matrix = get_csr_matrix(datapipe, rowName, colName, valName, shape)
    return matrix.tocsc()

def get_coo_matrix(
    datapipe: dp.iter.IterDataPipe, 
    rowName: str, colName: str, valName: str,
    shape: Optional[Tuple[int, int]] = None
) -> sp.coo_array:
    """A sparse matrix in COOrdinate format.
    Also known as the 'ijv' or 'triplet' format.
    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing
    """
    matrix = get_csr_matrix(datapipe, rowName, colName, valName, shape)
    return matrix.tocoo()

def get_lil_matrix(
    datapipe: dp.iter.IterDataPipe, 
    rowName: str, colName: str, valName: str,
    shape: Optional[Tuple[int, int]] = None
) -> sp.lil_array:
    """Row-based list of lists sparse matrix
    Advantages of the LIL format
        - supports flexible slicing
        - changes to the matrix sparsity structure are efficient

    Disadvantages of the LIL format
        - arithmetic operations LIL + LIL are slow (consider CSR or CSC)
        - slow column slicing (consider CSC)
        - slow matrix vector products (consider CSR or CSC)
    """
    matrix = get_csr_matrix(datapipe, rowName, colName, valName, shape)
    return matrix.tolil()

def get_dok_matrix(
    datapipe: dp.iter.IterDataPipe, 
    rowName: str, colName: str, valName: str,
    shape: Optional[Tuple[int, int]] = None
) -> sp.dok_array:
    """Dictionary Of Keys based sparse matrix.
    This is an efficient structure for constructing sparse
    matrices incrementally.
    """
    matrix = get_csr_matrix(datapipe, rowName, colName, valName, shape)
    return matrix.todok()

def sparse_matrix_to_coo_tensor(sparse_matrix):
    sparse_matrix = sparse_matrix.tocoo()
    rows = torch.from_numpy(sparse_matrix.row).long()
    cols = torch.from_numpy(sparse_matrix.col).long()
    vals = torch.from_numpy(sparse_matrix.data)
    indices = torch.stack((rows, cols), axis=0)
    return torch.sparse_coo_tensor(indices, vals, torch.Size(sparse_matrix.shape))

def sparse_matrix_to_csr_tensor(sparse_matrix):
    return sparse_matrix_to_coo_tensor(sparse_matrix).to_sparse_csr()