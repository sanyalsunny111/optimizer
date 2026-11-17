"""
LLaMA-style language model: decoder-only Transformer with RMSNorm pre-normalization,
rotary position embeddings (RoPE), SwiGLU feed-forward layers and no biases.

Follows the architecture of the reference implementation
(https://github.com/meta-llama/llama, model.py). RoPE is implemented with real
arithmetic (cos/sin tables) instead of complex numbers so that it works with
torch.compile and mixed precision; tests/test_models.py checks that it matches the
complex-number formulation.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class LlamaConfig:
    block_size: int = 1024  # maximum sequence length
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    mlp_hidden_dim: Optional[int] = None  # None: 8/3 * n_embd rounded up to `multiple_of`
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_weights: bool = False  # LLaMA does not tie the input embedding and output head


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # normalize in float32 for numerical stability under mixed precision
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def precompute_rope(head_dim, seq_len, theta=10000.0):
    """cos/sin tables of shape (seq_len, head_dim // 2)."""
    assert head_dim % 2 == 0, "RoPE needs an even head dimension"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Rotate consecutive feature pairs of x (B, n_head, T, head_dim) by position angles."""
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)  # each (B, nh, T, hd/2)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q = self.wq(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y))


class FeedForward(nn.Module):
    """SwiGLU: w2(silu(w1 x) * w3 x)."""

    def __init__(self, config):
        super().__init__()
        hidden = config.mlp_hidden_dim
        if hidden is None:
            hidden = int(2 * (4 * config.n_embd) / 3)
            hidden = config.multiple_of * ((hidden + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.attention_norm(x), cos, sin)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.tok_embeddings.weight = self.lm_head.weight

        cos, sin = precompute_rope(config.n_embd // config.n_head, config.block_size, config.rope_theta)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for the residual-stream output projections, as for GPT-2
        for name, p in self.named_parameters():
            if name.endswith('wo.weight') or name.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """Number of unique parameters (a tied embedding/head matrix is counted once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """idx, targets: (B, T) int64. Returns (logits (B, T, V), loss or None)."""
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        x = self.drop(self.tok_embeddings(idx))
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
