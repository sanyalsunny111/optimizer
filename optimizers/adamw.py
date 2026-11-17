"""AdamW baseline (Loshchilov & Hutter, 2019), using torch.optim.AdamW."""

import torch

from .registry import register, split_params


@register('adamw')
def build_adamw(model, cfg, device_type):
    """
    AdamW with decoupled weight decay on every >=2-D parameter (weight matrices and
    embeddings) and no weight decay on gains and biases, as in nanoGPT / Sophia.
    """
    groups = split_params(model)
    decay = groups['hidden_matrices'] + groups['embeddings'] + groups['head']
    no_decay = groups['gains_biases']
    param_groups = [
        {'params': decay, 'weight_decay': cfg.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(
        [g for g in param_groups if g['params']],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=(device_type == 'cuda'),  # fused CUDA kernel; numerically equivalent
    )
