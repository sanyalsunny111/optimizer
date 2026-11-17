import math

import pytest
import torch

from models import build_model
from models.llama import apply_rope, precompute_rope

TINY = dict(n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=128)


@pytest.mark.parametrize('model_type', ['gpt2', 'llama'])
def test_forward_shapes_and_initial_loss(model_type):
    torch.manual_seed(0)
    model = build_model(model_type, **TINY)
    idx = torch.randint(0, 128, (3, 32))
    targets = torch.randint(0, 128, (3, 32))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 32, 128)
    # a freshly initialized model predicts close to uniformly: loss ~ ln(vocab_size)
    assert abs(loss.item() - math.log(128)) < 0.2
    logits_only, no_loss = model(idx)
    assert no_loss is None
    assert torch.allclose(logits, logits_only)


@pytest.mark.parametrize('model_type', ['gpt2', 'llama'])
def test_causality(model_type):
    """Changing a token must not change the logits at earlier positions."""
    torch.manual_seed(0)
    model = build_model(model_type, **TINY).eval()
    idx = torch.randint(0, 128, (2, 32))
    changed = idx.clone()
    changed[:, 20] = (changed[:, 20] + 1) % 128
    a, _ = model(idx)
    b, _ = model(changed)
    assert torch.allclose(a[:, :20], b[:, :20], atol=1e-5)
    assert not torch.allclose(a[:, 20:], b[:, 20:], atol=1e-5)


@pytest.mark.parametrize('model_type', ['gpt2', 'llama'])
def test_rejects_too_long_sequences(model_type):
    model = build_model(model_type, **TINY)
    with pytest.raises(AssertionError):
        model(torch.zeros(1, 33, dtype=torch.long))


def test_gpt2_small_parameter_count():
    model = build_model('gpt2', n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=50257, bias=True)
    assert model.get_num_params() == 124_439_808  # the published GPT-2 small size


def test_weight_tying():
    gpt = build_model('gpt2', **TINY)
    assert gpt.transformer.wte.weight is gpt.lm_head.weight
    untied = build_model('llama', **TINY)
    assert untied.tok_embeddings.weight is not untied.lm_head.weight
    tied = build_model('llama', tie_weights=True, **TINY)
    assert tied.tok_embeddings.weight is tied.lm_head.weight
    assert untied.get_num_params() - tied.get_num_params() == 128 * 64


def test_llama_default_hidden_dim():
    model = build_model('llama', n_layer=1, n_head=12, n_embd=768, block_size=8, vocab_size=16)
    assert model.layers[0].feed_forward.w1.out_features == 2048  # 8/3 * 768
    model = build_model('llama', n_layer=1, n_head=16, n_embd=1024, block_size=8, vocab_size=16)
    assert model.layers[0].feed_forward.w1.out_features == 2816  # 2730.67 rounded up to 256


def test_rope_matches_complex_reference():
    """apply_rope must equal the complex-number formulation of the LLaMA reference code."""
    torch.manual_seed(0)
    B, nh, T, hd = 2, 3, 17, 16
    x = torch.randn(B, nh, T, hd)
    cos, sin = precompute_rope(hd, T, theta=10000.0)

    freqs = 1.0 / (10000.0 ** (torch.arange(0, hd, 2)[: hd // 2].float() / hd))
    freqs_cis = torch.polar(torch.ones(T, hd // 2), torch.outer(torch.arange(T).float(), freqs))
    xc = torch.view_as_complex(x.reshape(B, nh, T, hd // 2, 2))
    expected = torch.view_as_real(xc * freqs_cis).flatten(3)

    assert torch.allclose(apply_rope(x, cos, sin), expected, atol=1e-5)


def test_rope_is_relative():
    """<rope(q, m), rope(k, n)> only depends on the offset m - n."""
    torch.manual_seed(0)
    hd, T = 16, 12
    cos, sin = precompute_rope(hd, T)
    q = torch.randn(hd).expand(1, 1, T, hd)
    k = torch.randn(hd).expand(1, 1, T, hd)
    scores = apply_rope(q, cos, sin)[0, 0] @ apply_rope(k, cos, sin)[0, 0].T  # (T, T)
    for offset in range(-3, 4):
        diagonal = torch.diagonal(scores, offset)
        assert torch.allclose(diagonal, diagonal[0].expand_as(diagonal), atol=1e-4)


@pytest.mark.parametrize('model_type', ['gpt2', 'llama'])
def test_same_seed_same_init(model_type):
    torch.manual_seed(123)
    a = build_model(model_type, **TINY)
    torch.manual_seed(123)
    b = build_model(model_type, **TINY)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
