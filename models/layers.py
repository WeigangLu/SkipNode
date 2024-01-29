import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GCN2Conv, APPNP, SGConv, Sequential
from torch.nn import ReLU, Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import torch
import torch_geometric
import torch.nn.functional as F
from torch_sparse import SparseTensor, set_diag
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch.nn import Parameter
import numpy as np


class Layer(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, layer_type: str, layer_index, strategy=None, bias=True):
        super(Layer, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.layer_type = layer_type
        self.layer_index = layer_index
        self.bias = bias

        self.conv = None
        self.cache = None
        self.strategy = strategy

    def get_hid(self):
        return self.cache

    def forward(self, x, edge_index, x_0=None):

        # for DropEdge and DropNode
        if self.strategy.name == "DropEdge" and self.training:
            _, _, edge_index = self.strategy(None, None, edge_index)
        elif self.strategy.name == "DropNode":
            x, _, _ = self.strategy(x, None, edge_index)

        if self.conv is None:
            raise NotImplementedError

        if x_0 is not None:
            # for GCNII
            out = self.conv(x, x_0, edge_index)
        else:
            out = self.conv(x, edge_index)

        # for SkipNode, PairNorm, and SkipConnection
        if self.strategy.name in ("SkipNode", "PairNorm", "SkipConnection"):
            _, x_out, edge_index = self.strategy(x, out, edge_index,
                                                 self.layer_type)

        # if self.training:
        self.cache = out
        # else:
        #     self.cache = None

        return out


class GCNLayer(Layer):
    def __init__(self, in_channels, hid_channels, out_channels, layer_type, layer_index, strategy, bias=True):
        super(GCNLayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index, strategy, bias)
        strategy_name = strategy.name
        cached = True if strategy.name != "DropEdge" else False
        if self.layer_type == "in":
            if strategy_name == "DropMessage":
                self.conv = BbGCN(self.in_channels, self.hid_channels, strategy.drop_message_ratio)
            else:
                self.conv = GCNConv(self.in_channels, self.hid_channels, bias=self.bias, cached=cached)
        elif self.layer_type == "hid":
            if strategy_name == "DropMessage":
                self.conv = BbGCN(self.hid_channels, self.hid_channels, strategy.drop_message_ratio)
            else:
                self.conv = GCNConv(self.hid_channels, self.hid_channels, bias=self.bias, cached=cached)
        elif self.layer_type == "out":
            if strategy_name == "DropMessage":
                self.conv = BbGCN(self.hid_channels, self.out_channels, strategy.drop_message_ratio)
            else:
                self.conv = GCNConv(self.hid_channels, self.out_channels, bias=self.bias, cached=cached)


class GATLayer(Layer):
    def __init__(self, in_channels, hid_channels, out_channels, layer_type, layer_index, strategy, bias=True, heads=8):
        super(GATLayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index, strategy, bias)

        if self.layer_type == "in":
            self.conv = GATConv(self.in_channels, self.hid_channels, heads=heads, bias=self.bias)
        elif self.layer_type == "hid":
            self.conv = GATConv(self.hid_channels * heads, self.hid_channels, heads=heads, bias=self.bias)
        elif self.layer_type == "out":
            self.conv = GATConv(self.hid_channels * heads, self.out_channels, heads=1, bias=self.bias)


class SAGELayer(Layer):
    def __init__(self, in_channels, hid_channels, out_channels, layer_type, layer_index, strategy, bias=True):
        super(SAGELayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index, strategy,
                                        bias)

        if self.layer_type == "in":
            self.conv = SAGEConv(self.in_channels, self.hid_channels, bias=self.bias)
        elif self.layer_type == "hid":
            self.conv = SAGEConv(self.hid_channels, self.hid_channels, bias=self.bias)
        elif self.layer_type == "out":
            self.conv = SAGEConv(self.hid_channels, self.out_channels, bias=self.bias)


class APPNPLayer(Layer):
    def __init__(self, in_channels=1, hid_channels=1, out_channels=1, layer_type="in", layer_index=2, strategy=None,
                 bias=True):
        super(APPNPLayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index, strategy,
                                         bias)
        cached = True if self.strategy.name == "None" or self.strategy.name != "DropEdge" else False
        if self.layer_type == "out":
            self.conv = APPNP(K=layer_index, alpha=0.1, cached=cached)
        else:
            self.conv = lambda x, y: x


class GPRGNNLayer(Layer):
    def __init__(self, in_channels=1, hid_channels=1, out_channels=1, layer_type="in", layer_index=2, strategy=None,
                 bias=True):
        super(GPRGNNLayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index, strategy,
                                          bias)
        cached = True if self.strategy.name == "None" or self.strategy.name != "DropEdge" else False
        if self.layer_type == "out":
            self.conv = GPRProp(layer_index, alpha=0.1, cached=cached)
        else:
            self.conv = lambda x, y: x


class InceptLayer(Layer):
    def __init__(self, in_channels, hid_channels, out_channels, layer_type, layer_index, strategy, bias=True):
        super(InceptLayer, self).__init__(in_channels, hid_channels, out_channels, layer_index, layer_type, strategy,
                                          bias)

        conv_layer = Sequential('x, edge_index', [
            (GCNLayer(self.in_channels, self.hid_channels, self.hid_channels, layer_index=0, layer_type="in",
                      strategy=strategy, bias=self.bias),
             'x, edge_index -> x'),
            ReLU(inplace=True),
        ])
        for i in range(layer_index):
            conv_layer.add_module(f'GCN Layer {i}',
                                  GCNLayer(self.in_channels, self.hid_channels, self.hid_channels, layer_index=i + 1,
                                           layer_type="hid", strategy=strategy, bias=self.bias))
            if i < layer_index - 1:
                conv_layer.add_module(f'ReLu {i}', ReLU(inplace=True))

        self.conv = conv_layer


class GCNIILayer(Layer):
    def __init__(self, in_channels, hid_channels, out_channels, layer_type, layer_index, strategy, bias=True, alpha=0.1,
                 theta=0.5):
        super(GCNIILayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index, strategy)

        self.conv = GCN2Conv(channels=hid_channels, alpha=alpha, theta=theta, layer=layer_index + 1)


class GRANDLayer(Layer):
    def __init__(self, in_channels=1, hid_channels=1, out_channels=1, layer_type="in", layer_index=2, strategy=None,
                 bias=True):
        super(GRANDLayer, self).__init__(in_channels, hid_channels, out_channels, layer_type, layer_index, strategy,
                                         bias)
        cached = True if self.strategy.name == "None" or self.strategy.name != "DropEdge" else False
        self.conv = PlainConv(cached=cached)


class GPRProp(MessagePassing):
    '''
    propagation class for GPRGNN
    '''

    def __init__(self, K, alpha, cached=True, **kwargs):
        super(GPRProp, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.alpha = alpha
        # PPR-like
        TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
        TEMP[-1] = (1 - alpha) ** K

        self.cached = cached
        self.edge_index = None
        self.norm = None

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        if not self.cached or self.norm is None:
            self.edge_index, self.norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        edge_index = self.edge_index if self.edge_index is not None else edge_index
        norm = self.norm

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class PlainConv(MessagePassing):
    def __init__(self, cached=True):
        super(PlainConv, self).__init__()
        self.edge_weight = None
        self.edge_index = None
        self.cached = cached
        self.pt = ModelPretreatment()

    def forward(self, x: Tensor, edge_index: Adj):
        if self.cached and self.edge_weight is not None and self.edge_index is not None:
            edge_index = self.edge_index
            edge_weight = self.edge_weight
        else:
            self.edge_index, self.edge_weight = self.pt.pretreatment(x, edge_index)
            edge_index = self.edge_index
            edge_weight = self.edge_weight

        x = self.propagate(edge_index=edge_index, size=None, x=x)

        return x

    def message(self, x_j: Tensor):
        # normalize
        if self.edge_weight is not None:
            x_j = x_j * self.edge_weight.view(-1, 1)

        if not self.training:
            return x_j

        return x_j


class BbGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, drop_rate, add_self_loops: bool = True,
                 normalize: bool = True):
        super(BbGCN, self).__init__()
        self.pt = ModelPretreatment(add_self_loops, normalize)
        self.edge_weight = None

        self.lin = Linear(in_channels, out_channels)
        self.drop_rate = drop_rate

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj):
        edge_index, self.edge_weight = self.pt.pretreatment(x, edge_index)
        y = self.propagate(edge_index=edge_index, size=None, x=x)
        y = self.lin(y)
        return y

    def message(self, x_j: Tensor):
        # normalize
        if self.edge_weight is not None:
            x_j = x_j * self.edge_weight.view(-1, 1)

        if not self.training:
            return x_j

        # drop messages
        x_j = F.dropout(x_j, self.drop_rate)

        return x_j


class ModelPretreatment:
    def __init__(self, add_self_loops: bool = True, normalize: bool = True):
        super(ModelPretreatment, self).__init__()
        self.add_self_loops = add_self_loops
        self.normalize = normalize

    def pretreatment(self, x: Tensor, edge_index: Adj):
        # add self loop
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x.size(0)
                edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
                edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # normalize
        edge_weight = None
        if self.normalize:
            if isinstance(edge_index, Tensor):
                row, col = edge_index
            elif isinstance(edge_index, SparseTensor):
                row, col, _ = edge_index.coo()
            deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return edge_index, edge_weight
