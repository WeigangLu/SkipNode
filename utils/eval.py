import torch
from torch.utils.data import DataLoader


def evaluate(model, criterion, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        total_loss = criterion(out[mask], data.y[mask]).item()  # You might need to adjust this based on your task
        acc = get_accuracy(out[mask], data.y[mask])

        return total_loss, acc


def run_evaluation(model, criterion, data):
    if data.val_mask.sum(0) == 0:
        val_loss, val_acc = 0, 0
    else:
        val_loss, val_acc = evaluate(model, criterion, data, data.val_mask)

    if data.test_mask.sum(0) == 0:
        test_loss, test_acc = 0, 0
    else:
        test_loss, test_acc = evaluate(model, criterion, data, data.test_mask)

    return val_loss, val_acc, test_loss, test_acc


def get_accuracy(out, labels):
    pred = out.argmax(1)
    return pred.eq(labels).float().mean().item()
