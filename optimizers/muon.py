"""
Muon: MomentUm Orthogonalized by Newton-Schulz.

    K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, J. Bernstein.
    "Muon: An optimizer for hidden layers in neural networks", 2024.
    https://kellerjordan.github.io/posts/muon/

This follows the reference implementation (https://github.com/KellerJordan/Muon,
`SingleDeviceMuonWithAuxAdam`). Muon is only meant for the >=2-D weights of hidden
layers; embeddings, the output head and gains/biases are optimized by AdamW inside the
same optimizer object ("auxiliary Adam"), selected per param group with `use_muon`.

Under DistributedDataParallel every process holds identical, already averaged
gradients, so this single-device implementation is also the correct one for
multi-GPU training.
"""

import torch

from .registry import register, split_params


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Approximately orthogonalize G, i.e. map G = U S V^T to roughly U V^T, with a quintic
    Newton-Schulz iteration. The coefficients maximize the slope at zero, so the result
    is U S' V^T with singular values S' spread around 1 (roughly in [0.7, 1.2]) rather
    than exactly 1, which does not hurt optimization in practice. Singular values that
    are tiny relative to the largest one (below ~1e-3 of the Frobenius norm) are not
    fully lifted to 1 within 5 steps.

    As in the reference, the iteration runs in bfloat16 on CUDA; on other devices it
    runs in float32.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.is_cuda else G.float()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    # make the spectral norm at most 1 (Frobenius norm upper-bounds the spectral norm)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon for hidden weight matrices with auxiliary AdamW for all other parameters.

    Args:
        param_groups: list of dicts. Every group needs a boolean `use_muon`.
            use_muon=True  (all params must have >= 2 dims):
                lr (0.02), momentum (0.95), nesterov (True), ns_steps (5),
                weight_decay (0.0)
            use_muon=False (AdamW):
                lr (3e-4), betas ((0.9, 0.95)), eps (1e-10), weight_decay (0.0)

    Muon update for a weight W of shape (fan_out, fan_in) with gradient G:
        M <- momentum * M + (1 - momentum) * G
        U <- (1 - momentum) * G + momentum * M      (Nesterov; otherwise U <- M)
        O <- NewtonSchulz(U)                        (approx. nearest semi-orthogonal matrix)
        W <- (1 - lr * weight_decay) * W - lr * sqrt(max(1, fan_out / fan_in)) * O
    """

    def __init__(self, param_groups):
        groups = []
        for group in param_groups:
            group = dict(group)
            group['params'] = list(group['params'])
            if 'use_muon' not in group:
                raise ValueError("every param group needs a boolean 'use_muon' entry")
            if group['use_muon']:
                for p in group['params']:
                    if p.ndim < 2:
                        raise ValueError("Muon only supports parameters with >= 2 dimensions; "
                                         "put gains and biases in a use_muon=False group")
                group.setdefault('lr', 0.02)
                group.setdefault('momentum', 0.95)
                group.setdefault('nesterov', True)
                group.setdefault('ns_steps', 5)
                group.setdefault('weight_decay', 0.0)
            else:
                group.setdefault('lr', 3e-4)
                group.setdefault('betas', (0.9, 0.95))
                group.setdefault('eps', 1e-10)
                group.setdefault('weight_decay', 0.0)
            groups.append(group)
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, weight_decay = group['lr'], group['weight_decay']
            if group['use_muon']:
                momentum = group['momentum']
                for p in group['params']:
                    grad = p.grad
                    if grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(grad)
                    buf = state['momentum_buffer']
                    buf.lerp_(grad, 1 - momentum)
                    update = grad.lerp(buf, momentum) if group['nesterov'] else buf
                    update = update.reshape(update.size(0), -1)  # conv filters -> matrix
                    scale = max(1.0, update.size(-2) / update.size(-1)) ** 0.5
                    update = zeropower_via_newtonschulz5(update, steps=group['ns_steps'])
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)
                    p.add_(update.reshape(p.shape).to(p.dtype), alpha=-lr * scale)
            else:
                beta1, beta2 = group['betas']
                eps = group['eps']
                for p in group['params']:
                    grad = p.grad
                    if grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(grad)
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)
                    p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)
        return loss


@register('muon')
def build_muon(model, cfg, device_type):
    """
    Muon on all hidden weight matrices; AdamW on embeddings, output head, gains and
    biases with the same hyperparameters as the AdamW baseline (learning_rate, beta1,
    beta2, eps, weight_decay on >=2-D parameters only).
    """
    groups = split_params(model)
    adam = dict(use_muon=False, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    param_groups = [
        dict(params=groups['hidden_matrices'], use_muon=True, lr=cfg.muon_lr,
             momentum=cfg.muon_momentum, nesterov=cfg.muon_nesterov,
             ns_steps=cfg.muon_ns_steps, weight_decay=cfg.muon_weight_decay),
        dict(params=groups['embeddings'] + groups['head'], weight_decay=cfg.weight_decay, **adam),
        dict(params=groups['gains_biases'], weight_decay=0.0, **adam),
    ]
    return Muon([g for g in param_groups if g['params']])
