import torch
import torch.nn.functional as F

def cal_loss(pred, target, eps=0.0, smoothing=False, reduction='mean'):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    target = target.contiguous().view(-1)

    if smoothing:
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)
        if (reduction == 'mean'):
            loss = loss.mean()
        else:
            loss = loss.sum()
    else:
        loss = F.cross_entropy(pred, target, reduction=reduction)
    return loss