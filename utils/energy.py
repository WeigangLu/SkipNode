import numpy as np
import scipy
import torch
from scipy.sparse import coo_matrix
from torch_geometric.utils import get_laplacian


def tonp(tsr):
    if isinstance(tsr, np.ndarray):
        return tsr
    elif isinstance(tsr, np.matrix):
        return np.array(tsr)
    elif isinstance(tsr, scipy.sparse.csc.csc_matrix):
        return np.array(tsr.todense())

    assert isinstance(tsr, torch.Tensor)
    tsr = tsr.cpu()
    assert isinstance(tsr, torch.Tensor)

    try:
        arr = tsr.numpy()
    except TypeError:
        arr = tsr.detach().to_dense().numpy()
    except:
        arr = tsr.detach().numpy()

    assert isinstance(arr, np.ndarray)
    return arr


def get_laplacian_mat(edge_index, edge_weight, num_node, normalization='sym'):  # todo: change back
    """ return a laplacian (torch.sparse.tensor)"""
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                            normalization=normalization)  # see https://bit.ly/3c70FJK for format
    return torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([num_node, num_node]))


def cal_energy(v1, L1):
    """ compute the mapgap
        v1: n * d
        L1 : n * n
        return tr(v.T * L * v)
    """

    L1 = tonp(L1)
    assert v1.shape[0] == L1.shape[0] == L1.shape[1]
    v1 = v1.detach().cpu().numpy()
    E = np.dot(np.dot(v1.T, L1), v1)
    E = np.diag(E)
    return E