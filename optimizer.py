import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer


class LaProp(Optimizer):
    def __init__(self,
                 params,
                 lr=4e-4,
                 betas=(0.9, 0.9),
                 eps=1e-15,
                 weight_decay=0,
                 amsgrad=False,
                 centered=False,
                 auto=False):
        self.centered = centered
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad)
        super(LaProp, self).__init__(params, defaults)
        self.bias_correction1 = 0
        self.beta2 = 0
        self.auto = auto

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_mean_avg_sq'] = torch.zeros_like(p.data)
                    state['correction2'] = 1
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq, exp_mean_avg_sq = state['exp_avg'], state[
                    'exp_avg_sq'], state['exp_mean_avg_sq']
                correction2 = state['correction2']

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if self.auto:
                    self.beta2 = 1 - min(1, (exp_avg**2).mean())
                    if state['step'] > 5:
                        beta2 = self.beta2
                state['correction2'] = correction2 * beta2
                bias_correction2 = 1 - state['correction2']

                # Decay the first and second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                bias_correction1 = 1 - beta1**state['step']

                denom = exp_avg_sq
                if self.centered:
                    exp_mean_avg_sq.mul_(beta2).add_(1 - beta2, grad)
                    if state['step'] > 5:
                        mean = exp_mean_avg_beta2**2
                        denom = denom - mean
                if amsgrad:
                    if not (self.centered and state['step'] <= 5):
                        # Maintains the maximum of all (centered) 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, denom, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq

                denom = denom.abs().div(bias_correction2).sqrt_().add_(
                    group['eps'])
                step_of_this_grad = grad / denom
                exp_avg.mul_(beta1).add_(1 - beta1, step_of_this_grad)

                step_size = group['lr'] / bias_correction1

                p.data.add_(-step_size, exp_avg)
                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'], p.data)
        return loss