import torch
from utils.eval import run_evaluation, get_accuracy
import os
import time


def consis_loss(logps, temp=0.5):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)
    # p2 = torch.exp(logp2)

    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return loss


def train(model, optimizer, criterion, data, times=1):
    mask = data.train_mask

    model.train()

    optimizer.zero_grad()
    loss = 0
    out_list = []
    for i in range(times):
        out = model(data.x, data.edge_index)
        out_list.append(out)
        loss += criterion(out[mask], data.y[mask])
    loss /= times

    if times > 1:
        loss_consis = consis_loss(out_list)
        loss += 0.5 * loss_consis

    loss.backward()
    optimizer.step()

    acc = get_accuracy(out_list[0][mask], data.y[mask])
    return loss, acc


def run_training(model, optimizer, criterion, data, args):
    es = 0
    best_acc = 0
    best_val_acc = 0

    num_epochs = args.epochs
    times = args.grand_prop_times

    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, optimizer, criterion, data, times)

        val_loss, val_acc, test_loss, test_acc = run_evaluation(model, criterion, data)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f} | Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc * 100:.2f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f} ')

        # select the result based on valdation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_acc = test_acc
            es = 0
        else:
            es += 1

        if es == 50:
            print("Early stopping!")
            break
    end_time = time.time()
    run_time = (end_time - start_time) / num_epochs

    return best_acc * 100, run_time
