import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import WeightedRandomSampler, DataLoader


class SkipLayer(nn.Module):
    def __init__(self, skip_type, sampling_rate, adj, degree=None,
                 device=torch.device('cpu:0')):
        super(SkipLayer, self).__init__()
        self.skip_type = skip_type
        self.sampling_rate = sampling_rate
        self.adj = adj.to_dense()
        self.degree = degree.squeeze(dim=1)
        self.N = degree.shape[0]
        self.device = device

    def forward(self, is_training=True):
        mask = torch.FloatTensor([1.0 for _ in range(self.N)])

        if not is_training:
            return mask.unsqueeze(1).to(self.device)

        if self.skip_type == 'degree':
            rowsum = self.degree
            prob = rowsum / (rowsum.sum() + 1e-6)
        else:
            prob = np.ones(self.N)
            prob = prob / self.N

        index = torch.Tensor([i for i in range(self.N)]).to(self.device)
        size = int(self.N * self.sampling_rate)
        dataloader = DataLoader(dataset=index, batch_size=size,
                                sampler=WeightedRandomSampler(prob, size, replacement=False))
        sampled_idx = None
        for data in dataloader:
            sampled_idx = data
        sampled_idx = sampled_idx.to(torch.int64).cpu()
        mask = mask.index_fill_(0, sampled_idx, 0)
        mask = mask.unsqueeze(1).to(self.device)
        return mask


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, skiplayer=None, residual=False, useBN=False, stdv=0.):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.skiplayer = skiplayer
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.useBN = useBN
        self.stdv = stdv
        if self.useBN:
            self.bn = torch.nn.BatchNorm1d(self.out_features)
        self.reset_parameters()
        self.hid_feat = None

    def reset_parameters(self):
        if self.stdv == 0.:
            stdv = 1. / math.sqrt(self.out_features)
            self.weight.data.uniform_(-stdv, stdv)
        else:
            self.weight.data.uniform_(-self.stdv, self.stdv)

    def forward(self, input, p_mat, training=True):
        if training and self.skiplayer and self.skiplayer.skip_type != 'none':
            x = input
            support = torch.mm(x, self.weight)
            support = torch.spmm(p_mat, support)
            p = self.skiplayer(training)
            output = p * (support - input) + input
        else:
            support = torch.mm(input, self.weight)
            output = torch.spmm(p_mat, support)

        self.hid_feat = output

        if self.useBN:
            output = self.bn(output)
        if self.residual:
            output = output + input
        return output


class MultiGCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, skiplayer, pred_out=True, residual=False):
        super(MultiGCN, self).__init__()
        self.skiplayer = skiplayer

        self.convs = nn.ModuleList()
        self.convs.append(GraphConvolution(nfeat, nhidden, stdv=0.5))
        self.pred_out = pred_out
        self.residual = residual
        # default useBN for ResGCN
        self.useBN = residual
        for _ in range(nlayers - 2):
            self.convs.append(GraphConvolution(nhidden, nhidden, self.skiplayer, residual, useBN=self.useBN, stdv=0.5))
        if pred_out:
            self.convs.append(GraphConvolution(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, p_mat):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.convs[0](x, p_mat, training=self.training))
        for i, con in enumerate(self.convs[1:-1]):
            x_input = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(con(x_input, p_mat, training=self.training))
        if len(self.convs) > 1:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(self.convs[-1](x, p_mat, training=self.training))
        if self.pred_out:
            return F.log_softmax(x, dim=1)
        else:
            return x


class JKNet(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, skiplayer, residual=False):
        super(JKNet, self).__init__()
        self.skiplayer = skiplayer

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.convs.append(GraphConvolution(nfeat, nhidden))
        for _ in range(nlayers - 1):
            self.convs.append(GraphConvolution(nhidden, nhidden, self.skiplayer, residual=False))
        self.fcs = nn.Linear(nhidden * nlayers, nclass)
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, p_mat):
        hiddens = []
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.convs[0](x, p_mat, training=self.training))
        hiddens.append(x)
        for i, con in enumerate(self.convs[1:]):
            x_input = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(con(x_input, p_mat, training=self.training))
            hiddens.append(x)
        x = self.fcs(torch.cat(hiddens, -1))
        return F.log_softmax(x, dim=1)


class InceptGCN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, skiplayer, residual=False):
        super(InceptGCN, self).__init__()
        self.skiplayer = skiplayer

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for i in range(nlayers):
            self.convs.append(MultiGCN(nfeat, i + 2, nhidden, nhidden, dropout, skiplayer, pred_out=False))
        self.fcs = nn.Linear(nhidden * nlayers, nclass)
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, p_mat):
        input = x
        hiddens = []
        for i, con in enumerate(self.convs):
            # x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(con(input, p_mat))
            hiddens.append(x)
        output = self.fcs(torch.cat(hiddens, -1))
        return F.log_softmax(output, dim=1)
