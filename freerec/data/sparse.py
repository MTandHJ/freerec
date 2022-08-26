

from typing import Optional, Tuple

import torch
import torchdata.datapipes as dp
import numpy as np
import scipy.sparse as sp
import pandas as pd



def get_csr_matrix(
    datapipe: dp.iter.IterDataPipe, 
    rowName: str, colName: str, valName: str,
    shape: Optional[Tuple[int, int]] = None
) -> sp.csr_array:
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
    matrix = get_csr_matrix(datapipe, rowName, colName, valName, shape)
    return matrix.tocsc()

def get_coo_matrix(
    datapipe: dp.iter.IterDataPipe, 
    rowName: str, colName: str, valName: str,
    shape: Optional[Tuple[int, int]] = None
) -> sp.coo_array:
    matrix = get_csr_matrix(datapipe, rowName, colName, valName, shape)
    return matrix.tocoo()

def get_lil_matrix(
    datapipe: dp.iter.IterDataPipe, 
    rowName: str, colName: str, valName: str,
    shape: Optional[Tuple[int, int]] = None
) -> sp.lil_array:
    matrix = get_csr_matrix(datapipe, rowName, colName, valName, shape)
    return matrix.tolil()

def get_dok_matrix(
    datapipe: dp.iter.IterDataPipe, 
    rowName: str, colName: str, valName: str,
    shape: Optional[Tuple[int, int]] = None
) -> sp.dok_array:
    matrix = get_csr_matrix(datapipe, rowName, colName, valName, shape)
    return matrix.todok()

def sparse_matrix_to_tensor(sparse_matrix):
    sparse_matrix = sparse_matrix.tocoo()
    rows = torch.from_numpy(sparse_matrix.row).long()
    cols = torch.from_numpy(sparse_matrix.col).long()
    vals = torch.from_numpy(sparse_matrix.data)
    indices = torch.stack((rows, cols), axis=0)
    return torch.sparse_coo_tensor(indices, vals, torch.Size(sparse_matrix.shape))