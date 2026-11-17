"""
Optimizer registry.

An optimizer is added to the repo by registering a *builder*: a function that takes
the (unwrapped) model, the run configuration and the device type, and returns a
torch.optim.Optimizer. train.py then selects it with `--optimizer=<name>`.

    from .registry import register, split_params

    @register('myopt')
    def build_myopt(model, cfg, device_type):
        groups = split_params(model)
        return MyOpt(groups['hidden_matrices'] + ..., lr=cfg.learning_rate)

The learning-rate schedule in train.py multiplies the `lr` every param group was
constructed with, so groups may use different peak learning rates.
"""

import torch.nn as nn

_BUILDERS = {}


def register(name):
    def decorator(fn):
        if name in _BUILDERS:
            raise ValueError(f"optimizer {name!r} is already registered")
        _BUILDERS[name] = fn
        return fn
    return decorator


def available_optimizers():
    return sorted(_BUILDERS)


def build_optimizer(name, model, cfg, device_type):
    if name not in _BUILDERS:
        raise ValueError(f"unknown optimizer {name!r}; available: {available_optimizers()}")
    return _BUILDERS[name](model, cfg, device_type)


def split_params(model):
    """
    Partition the trainable parameters of a language model into four disjoint lists:

      hidden_matrices: >=2-D weights inside the Transformer blocks (attention / MLP)
      embeddings:      weights of nn.Embedding modules (including an embedding matrix
                       that is tied to the output head)
      head:            the output projection `model.lm_head.weight`, if it is not tied
      gains_biases:    all parameters with fewer than 2 dimensions (norm gains, biases)

    Tied parameters appear exactly once.
    """
    embedding_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    head = getattr(model, 'lm_head', None)
    head_ids = {id(head.weight)} if head is not None else set()

    groups = {'hidden_matrices': [], 'embeddings': [], 'head': [], 'gains_biases': []}
    for p in model.parameters():  # yields shared (tied) tensors only once
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            groups['gains_biases'].append(p)
        elif id(p) in embedding_ids:
            groups['embeddings'].append(p)
        elif id(p) in head_ids:
            groups['head'].append(p)
        else:
            groups['hidden_matrices'].append(p)
    return groups
