from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from new_model import *
import os

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=16, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--model', type=str, default='GCN', help='which model to choose')
parser.add_argument('--residual', action='store_true', default=False, help='skip connection')

parser.add_argument('--sampling_rate', type=float, default=0., help='node sampling rate')
parser.add_argument('--skip_type', type=str, default='none', help='how to skip')
parser.add_argument('--edge_sampling_rate', type=float, default=0., help='edge sampling rate')

# for experimental setting
parser.add_argument('--task', type=str, default='semi', help='task type')  # 'semi', 'full'
args = parser.parse_args()


def train(model, features, train_p_mat, idx_train, labels, device, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(features, train_p_mat)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def validate(model, features, test_p_mat, idx_val, labels, device):
    model.eval()
    with torch.no_grad():
        output = model(features, test_p_mat)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(), acc_val.item()


def test(model, features, test_p_mat, idx_test, labels, device, checkpt_file):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, test_p_mat)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(), acc_test.item()


def run(splitstr=None):
    p_mat, features, labels, idx_train, idx_val, idx_test, adj, degree, pre_processed_adj = load_data(args.data)
    if splitstr is not None and args.task == 'full':
        idx_train, idx_val, idx_test = split_data(splitstr)

    cudaid = "cpu:0" if not torch.cuda.is_available() else "cuda:" + str(args.dev)
    device = torch.device(cudaid)
    features = features.to(device)
    train_p_mat = p_mat.to(device)
    test_p_mat = p_mat.to(device)
    adj = adj.to(device)

    seed = i + 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if not os.path.exists('pretrained'):
        os.mkdir('pretrained')
    checkpt_file = 'pretrained/{}_{}_{}_{}.pt'.format(args.model, args.data, args.layer, time.time())
    print(cudaid, checkpt_file)

    # choose the base model
    if args.model == 'GCN':
        BaseModel = MultiGCN
    elif args.model == 'ResGCN':
        BaseModel = MultiGCN
        args.residual = True
    elif args.model == 'JKNet':
        BaseModel = JKNet
    elif args.model == 'InceptGCN':
        BaseModel = InceptGCN
    else:
        raise ValueError("Undefined model {}!".format(args.model))

    # initialize the skiplayer
    skiplayer = SkipLayer(skip_type=args.skip_type, sampling_rate=args.sampling_rate,
                          adj=adj, degree=degree, device=device)

    model = BaseModel(nfeat=features.shape[1],
                      nlayers=args.layer,
                      nhidden=args.hidden,
                      nclass=int(labels.max()) + 1,
                      dropout=args.dropout,
                      skiplayer=skiplayer,
                      residual=args.residual,
                      ).to(device)

    optimizer = optim.Adam([
        {'params': model.params1, 'weight_decay': args.wd},
    ], lr=args.lr)

    best_acc = 0.
    for epoch in range(args.epochs):
        # using dropedge or not
        if args.edge_sampling_rate > 0.:
            adj, train_p_mat = sample_edge(args.edge_sampling_rate, pre_processed_adj)
            train_p_mat = train_p_mat.to(device)
        loss_tra, acc_tra = train(model, features, train_p_mat, idx_train, labels, device, optimizer)
        loss_val, acc_val = validate(model, features, test_p_mat, idx_val, labels, device)

        print('Epoch:{:04d}'.format(epoch + 1),
              'train',
              'loss:{:.3f}'.format(loss_tra),
              'acc:{:.2f}'.format(acc_tra * 100),
              '| val',
              'loss:{:.3f}'.format(loss_val),
              'acc:{:.2f}'.format(acc_val * 100))

        if acc_val > best_acc:
            best_acc = acc_val
            torch.save(model.state_dict(), checkpt_file)

    acc = test(model, features, test_p_mat, idx_test, labels, device, checkpt_file)[1]
    os.remove(checkpt_file)

    return acc


acc_list = []
args.round = 10 if args.task == "full" else 1

for i in range(args.round):
    if args.task == 'semi':
        acc = run()
    elif args.task == 'full':
        acc = run('splits/' + args.data + '_split_0.6_0.2_' + str(i) + '.npz')
    acc_list.append(acc)
    print("model: {} data:{} ".format(args.model, args.data))
    print("Test acc.:{:.1f}".format(acc * 100))
print("average accuracy: {}".format(np.mean(acc_list)))

