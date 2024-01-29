import torch
from torch_geometric.utils import dropout_adj
from torch_geometric.nn.norm import PairNorm as PM


class Strategy(torch.nn.Module):
    def __init__(self, param):
        super(Strategy, self).__init__()
        self.name = param["name"]
        self.drop_edge_ratio = param["drop_edge_ratio"]

        self.skip_node_ratio = param["skip_node_ratio"]
        self.skip_node_type = param["skip_node_type"]
        self.degree = None
        if self.name == "PairNorm":
            self.pairnorm_layer = PM(scale=param["pair_norm_scale"])

        self.drop_message_ratio = param["drop_message_ratio"]

        self.drop_node_ratio = param["drop_node_ratio"]

    def skipnode_forward(self, x_in, x_out, edge_index, layer_type="hid"):
        if x_in.size(1) != x_out.size(1):
            return x_in, x_out, edge_index
        if self.degree is None:
            from torch_geometric.utils import degree
            d = degree(edge_index[0])
            d = d / d.sum()
            self.degree = d
        num_nodes = x_in.size(0)
        num_sampled_nodes = int(self.skip_node_ratio * num_nodes)
        if self.skip_node_type == "u":
            sampled_indices = torch.randperm(num_nodes)[:num_sampled_nodes]
        elif self.skip_node_type == "b":
            sampled_indices = torch.multinomial(self.degree, num_samples=num_sampled_nodes, replacement=False)
        else:
            raise ValueError(f"Not Supported for this skip node type {self.skip_node_type}")

        # Apply SkipNode strategy on x using the sampled_indices
        x_out[sampled_indices] = x_in[sampled_indices]

        return x_in, x_out, edge_index

    def skipconncetion_forward(self, x_in, x_out, edge_index):
        if x_in.size(1) != x_out.size(0):
            return x_in, x_out, edge_index

        return x_in, 0.5*x_out + 0.5*x_in, edge_index

    def dropedge_forward(self, x_in, x_out, edge_index):
        aug_adj = edge_index.detach()
        if self.training:
            aug_adj = dropout_adj(edge_index=edge_index, p=self.drop_edge_ratio)[0]
        return x_in, x_out, aug_adj

    def dropnode_forward(self, x_in, x_out, edge_index):
        x_in = x_in * torch.bernoulli(torch.ones(x_in.size(0), 1) - self.drop_node_ratio).to(x_in.device)
        x_in = x_in / (1 - self.drop_node_ratio)

        return x_in, x_out, edge_index

    def dropmessage_forward(self, x_in, x_out, edge_index):
        return x_in, x_out, edge_index

    def pairnorm_forward(self, x_in, x_out, edge_index):
        return x_in, self.pairnorm_layer(x_out), edge_index

    def forward(self, x_in, x_out, edge_index, layer_type="hid"):
        if self.name == "SkipNode":
            return self.skipnode_forward(x_in, x_out, edge_index, layer_type="hid")
        elif self.name == "DropEdge":
            return self.dropedge_forward(x_in, x_out, edge_index)
        elif self.name == "DropNode":
            return self.dropnode_forward(x_in, x_out, edge_index)
        elif self.name == "DropMessage":
            return self.dropmessage_forward(x_in, x_out, edge_index)
        elif self.name == "SkipConnection":
            return self.skipconncetion_forward(x_in, x_out, edge_index)
        elif self.name == "PairNorm":
            return self.pairnorm_forward(x_in, x_out, edge_index)
        else:
            return x_in, x_out, edge_index