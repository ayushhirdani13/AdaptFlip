import torch
import torch.nn.functional as F

def truncated_loss(y, label, drop_rate):
    loss = F.binary_cross_entropy_with_logits(y, label, reduction='none')
    loss_mul = loss * label
    ind_sorted = torch.argsort(loss_mul)
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], label[ind_update])
    return loss_update

def truncated_loss_cdae(y, label, drop_rate):
    y_flat = y.view(-1)
    label_flat = label.view(-1)
    loss_mat = F.binary_cross_entropy_with_logits(y, label, reduction="none")
    loss = loss_mat.view(-1)

    loss_mul = loss * label_flat
    ind_sorted = torch.argsort(loss_mul.data)
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y_flat[ind_update], label_flat[ind_update])

    return loss_update