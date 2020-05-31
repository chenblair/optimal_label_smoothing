import torch
import torch.nn.functional as F
import pdb

def nll_loss(pred, target, **kwargs):
    loss = F.cross_entropy(pred, target, reduction='none')
    return loss

def smoothing_loss(pred, target, p=1.0, **kwargs):
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
    one_hot = one_hot * p + (1 - one_hot) * (1 - p) / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1)

    return loss

def fsmoothing_loss(pred, target, p=1.0, **kwargs):
    n_class = pred.size(1)

    one_hot = torch.eye(n_class, n_class).to(pred.get_device())
    one_hot = one_hot * p + (1 - one_hot) * (1 - p) / (n_class - 1)
    
    pred = F.softmax(pred, dim=1)
    pred = torch.matmul(pred, one_hot)
    pred = pred.log()
    
    loss = F.nll_loss(pred, target)
    return loss

def lq_loss(pred, target, q=0.7, **kwargs):
    pred = F.softmax(pred, dim=1)
    loss = -1 * F.nll_loss(pred, target, reduction='none')
    loss = (1 - (loss + 1e-10).pow(q)) / q
    return loss

def cal_loss(pred, target, method='nll', reduction='mean', **kwargs):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    target = target.contiguous().view(-1)

    loss = globals()["{}_loss".format(method)](pred, target, **kwargs)
    if (reduction == 'mean'):
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss