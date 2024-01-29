import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GCNLayer, GATLayer, SAGELayer, InceptLayer, GCNIILayer, APPNPLayer, GPRGNNLayer, GRANDLayer
from torch_geometric.nn import JumpingKnowledge, Sequential
from torch.nn import Linear, ReLU, Dropout
import os
import numpy as np


class GNN(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_layers, dropout, layer_name="GCN",
                 grand_prop_times=1, strategy=None, bias=True):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_name = layer_name
        self.bias = bias

        self.jk = None
        self.init_lin = None
        self.out_lin = None

        # for JKNet and InceptGCN
        if layer_name in ("JKNet", "InceptGCN"):
            self.jk = JumpingKnowledge("cat")
            self.out_lin = Linear(num_layers * hid_channels, out_channels)

        if layer_name == "GCNII":
            self.init_lin = Linear(in_channels, hid_channels)
            self.out_lin = Linear(hid_channels, out_channels)

        if layer_name in ("APPNP", "GPRGNN"):
            self.init_lin = Sequential('x', [
                (Dropout(p=dropout), 'x -> x'),
                (Linear(self.in_channels, self.hid_channels), 'x -> x'),
                ReLU(inplace=True),
                (Dropout(p=dropout), 'x -> x'),
                (Linear(self.hid_channels, self.out_channels), 'x -> x'),
            ])

        if layer_name == "GRAND":
            self.out_lin = Sequential('x', [
                (Dropout(p=dropout), 'x -> x'),
                (Linear(self.in_channels, self.hid_channels), 'x -> x'),
                ReLU(inplace=True),
                (Dropout(p=dropout), 'x -> x'),
                (Linear(self.hid_channels, self.out_channels), 'x -> x'),
            ])

        self.strategy = strategy

        self.convs = nn.ModuleList()  # List to hold the layers
        for i in range(num_layers):
            if i == 0:
                conv = self.bulid_layer("in", i)
            elif 0 < i < num_layers - 1:
                conv = self.bulid_layer("hid", i)
            else:
                if self.layer_name == "JKNet":
                    conv = self.bulid_layer("hid")
                else:
                    conv = self.bulid_layer("out", i)
            self.convs.append(conv)

    def bulid_layer(self, layer_type, layer_index=1):
        conv_layer = None
        if self.layer_name in ("GCN", "ResGCN"):
            conv_layer = GCNLayer(self.in_channels, self.hid_channels, self.out_channels, layer_type, layer_index,
                                  self.strategy, bias=self.bias)
        elif self.layer_name == "GAT":
            conv_layer = GATLayer(self.in_channels, self.hid_channels, self.out_channels, layer_type, layer_index,
                                  self.strategy, bias=self.bias)
        elif self.layer_name == "SAGE":
            conv_layer = SAGELayer(self.in_channels, self.hid_channels, self.out_channels, layer_type, layer_index,
                                   self.strategy, bias=self.bias)
        elif self.layer_name == "JKNet":
            conv_layer = GCNLayer(self.in_channels, self.hid_channels, self.out_channels, layer_type, layer_index,
                                  self.strategy, bias=self.bias)
        elif self.layer_name == "InceptGCN":
            conv_layer = InceptLayer(self.in_channels, self.hid_channels, self.out_channels, layer_type, layer_index,
                                     self.strategy, bias=self.bias)
        elif self.layer_name == "GCNII":
            conv_layer = GCNIILayer(self.in_channels, self.hid_channels, self.out_channels, layer_type, layer_index,
                                    self.strategy, bias=self.bias)
        elif self.layer_name == "APPNP":
            conv_layer = APPNPLayer(layer_type=layer_type, layer_index=self.num_layers, strategy=self.strategy,
                                    bias=self.bias)
        elif self.layer_name == "GPRGNN":
            conv_layer = GPRGNNLayer(layer_type=layer_type, layer_index=self.num_layers, strategy=self.strategy,
                                     bias=self.bias)
        elif self.layer_name == "GRAND":
            conv_layer = GRANDLayer(layer_index=self.num_layers, strategy=self.strategy, bias=self.bias)
        else:
            raise ValueError(f"{self.layer_name} is not supported!")

        return conv_layer

    def forward(self, x, edge_index):
        x_input = torch.clone(x)

        x = self.init_lin(x) if self.layer_name in ("GCNII", "APPNP", "GPRGNN") else x
        x_0 = torch.clone(x) if self.layer_name == "GCNII" else None

        for i, conv in enumerate(self.convs):

            x_in = x_input if self.layer_name == "InceptGCN" else x

            if self.layer_name not in ("APPNP", "GPRGNN", "GRAND"):
                x_in = F.dropout(x_in, p=self.dropout, training=self.training)
            elif i == self.num_layers:
                x_in = F.dropout(x_in, p=self.dropout, training=self.training)

            x_out = conv(x_in, edge_index, x_0=x_0)

            if i < self.num_layers - 1 and self.layer_name not in ("APPNP", "GPRGNN", "GRAND"):
                x_out = F.relu(x_out)

            x = x_out

        if self.layer_name in ("JKNet", "InceptGCN"):
            x = self.jk([conv.get_hid() for conv in self.convs])
            x = self.out_lin(x)

        x = self.out_lin(x) if self.layer_name in ("GCNII", "GRAND") else x

        return F.log_softmax(x, dim=1)
