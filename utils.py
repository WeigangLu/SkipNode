import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
import json
from networkx.readwrite import json_graph
import os
from sklearn.metrics import pairwise_distances

sys.setrecursionlimit(99999)


def mad_value(in_arr, mask_arr, distance_metric='cosine', digt_num=4, target_idx=None):
    in_arr = in_arr + 1e-6
    dist_arr = pairwise_distances(in_arr, in_arr, metric=distance_metric)

    mask_dist = np.multiply(dist_arr, mask_arr)

    divide_arr = (mask_dist != 0).sum(1) + 1e-8

    node_dist = mask_dist.sum(1) / divide_arr

    if target_idx is None:
        mad = np.mean(node_dist)
    else:
        node_dist = np.multiply(node_dist, target_idx)
        mad = node_dist.sum() / ((node_dist != 0).sum() + 1e-8)

    mad = round(mad, digt_num)

    return mad


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum == 0) * 1 + rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# adapted from tkipf/gcn
def load_citation(dataset_str="cora"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    pre_processed_adj = adj.copy()
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    features = normalize(features)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    # adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    p_mat = sys_normalized_adjacency(adj)
    p_mat = sparse_mx_to_torch_sparse_tensor(p_mat)

    degree = torch.Tensor(adj.sum(1))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return p_mat, features, labels, idx_train, idx_val, idx_test, adj, degree, pre_processed_adj


def load_het_data(dataset_name):
    graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                            'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
        graph_node_features_and_labels_file.readline()
        for line in graph_node_features_and_labels_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 3)
            assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
            graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
            graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    pre_processed_adj = adj.copy()
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = normalize(features)

    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    degree = torch.Tensor(adj.sum(1))
    p_mat = sys_normalized_adjacency(adj)
    p_mat = sparse_mx_to_torch_sparse_tensor(p_mat)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return p_mat, features, labels, None, None, None, adj, degree, pre_processed_adj


def split_data(splits_file_path):
    with np.load(splits_file_path) as splits_file:
        idx_train = splits_file['train_mask']
        idx_val = splits_file['val_mask']
        idx_test = splits_file['test_mask']

    return idx_train, idx_val, idx_test


def load_data(datastr):
    datastr = datastr.lower()
    if datastr in ['cora', 'citeseer', 'pubmed']:
        return load_citation(datastr)
    elif datastr in ['wisconsin', 'chameleon', 'cornell', 'texas']:
        return load_het_data(datastr)
    else:
        raise ValueError(f"Not supported data {datastr}")


# adapt from DropEdge https://github.com/DropEdge/DropEdge
def sample_edge(sampling_rate, adj):
    nnz = adj.nnz
    perm = np.random.permutation(nnz)
    preserve_nnz = int(nnz * (1 - sampling_rate))
    perm = perm[:preserve_nnz]
    adj = sp.coo_matrix(adj)
    sampled_adj = sp.coo_matrix((adj.data[perm], (adj.row[perm], adj.col[perm])), shape=adj.shape)

    p_mat = sys_normalized_adjacency(sampled_adj)
    p_mat = sparse_mx_to_torch_sparse_tensor(p_mat)

    # degree = torch.Tensor(adj.sum(1))
    sampled_adj = sparse_mx_to_torch_sparse_tensor(sampled_adj)

    return sampled_adj, p_mat
