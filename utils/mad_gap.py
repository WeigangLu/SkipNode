import torch
import torch.nn.functional as F


def compute_mad_value(intensor, neb_mask, rmt_mask, target_idx):
    node_num, feat_num = intensor.size()

    input1 = intensor.expand(node_num, node_num, feat_num)
    input2 = input1.transpose(0, 1)

    input1 = input1.contiguous().view(-1, feat_num)
    input2 = input2.contiguous().view(-1, feat_num)

    simi_tensor = F.cosine_similarity(input1, input2, dim=1, eps=1e-8).view(node_num, node_num)
    dist_tensor = 1 - simi_tensor

    neb_dist = torch.mul(dist_tensor, neb_mask)
    rmt_dist = torch.mul(dist_tensor, rmt_mask)

    divide_neb = (neb_dist != 0).sum(1).type(torch.FloatTensor) + 1e-8
    divide_rmt = (rmt_dist != 0).sum(1).type(torch.FloatTensor) + 1e-8
    divide_neb = divide_neb.to(intensor.device)
    divide_rmt = divide_rmt.to(intensor.device)

    neb_mean_list = neb_dist.sum(1) / divide_neb
    rmt_mean_list = rmt_dist.sum(1) / divide_rmt
    neb_mean_list = neb_mean_list.squeeze(0)
    rmt_mean_list = rmt_mean_list.squeeze(0)
    neb_mad = torch.mean(neb_mean_list[target_idx])
    rmt_mad = torch.mean(rmt_mean_list[target_idx])

    mad_gap = rmt_mad - neb_mad

    return mad_gap
