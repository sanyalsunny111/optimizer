"""
Template for a new optimizer: SGD with (Nesterov) momentum and decoupled weight decay.

It is a complete, working optimizer (`--optimizer=sgd`) meant to be copied:

    cp optimizers/template.py optimizers/myopt.py
    # 1. rename the class and change `step`
    # 2. change the name in @register(...) and the hyperparameters read from `cfg`
    # 3. add `from . import myopt` to optimizers/__init__.py
    # 4. add any new hyperparameters (e.g. myopt_rho=0.04) to `defaults` in train.py
    python train.py config/gpt2_small_adamw.py --optimizer=myopt --run_name=gpt2_small_myopt

Contract with train.py:
  * the builder receives the plain nn.Module (not compiled, not DDP-wrapped), the run
    configuration `cfg` (attribute access to every key of `defaults` in train.py) and
    the device type ('cuda' or 'cpu');
  * every param group needs an 'lr': train.py remembers it as that group's peak LR and
    multiplies it by the schedule (warmup + decay) before every step;
  * under DDP every process holds the same averaged gradients when `step` is called,
    after gradient clipping (and unscaling when dtype=float16);
  * `state_dict` / `load_state_dict` are used for checkpoints, so all optimizer state
    must live in `self.state[p]` (tensors or Python numbers).
"""

import torch

from .registry import register, split_params


class SGDMomentum(torch.optim.Optimizer):
    """
    m <- momentum * m + g
    u <- g + momentum * m     (Nesterov; otherwise u <- m)
    w <- (1 - lr * weight_decay) * w - lr * u
    """

    def __init__(self, params, lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.0):
        if lr < 0 or not 0 <= momentum < 1 or weight_decay < 0:
            raise ValueError("invalid hyperparameters")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum, weight_decay = group['lr'], group['momentum'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(p.grad)
                update = p.grad.add(buf, alpha=momentum) if group['nesterov'] else buf
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)
        return loss


@register('sgd')
def build_sgd(model, cfg, device_type):
    # learning_rate / weight_decay / beta1 are the shared hyperparameters of train.py;
    # a new optimizer can equally add its own keys to `defaults` in train.py
    groups = split_params(model)
    return SGDMomentum([
        dict(params=groups['hidden_matrices'] + groups['embeddings'] + groups['head'],
             weight_decay=cfg.weight_decay),
        dict(params=groups['gains_biases'], weight_decay=0.0),
    ], lr=cfg.learning_rate, momentum=cfg.beta1)
