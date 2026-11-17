import copy
from types import SimpleNamespace

import pytest
import torch

from models import build_model
from optimizers import available_optimizers, build_optimizer, split_params
from optimizers.muon import Muon, zeropower_via_newtonschulz5

TINY = dict(n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=128)
CFG = SimpleNamespace(
    learning_rate=1e-3, weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8,
    muon_lr=0.02, muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=5, muon_weight_decay=0.01,
)


def test_registry():
    assert {'adamw', 'muon', 'sgd'} <= set(available_optimizers())
    with pytest.raises(ValueError):
        build_optimizer('does_not_exist', build_model('gpt2', **TINY), CFG, 'cpu')


@pytest.mark.parametrize('model_type,kwargs', [('gpt2', {}), ('gpt2', {'bias': True}),
                                               ('llama', {}), ('llama', {'tie_weights': True})])
def test_split_params_is_a_partition(model_type, kwargs):
    model = build_model(model_type, **TINY, **kwargs)
    groups = split_params(model)
    ids = [id(p) for params in groups.values() for p in params]
    assert len(ids) == len(set(ids))  # disjoint
    assert set(ids) == {id(p) for p in model.parameters()}  # complete
    assert all(p.ndim >= 2 for key in ('hidden_matrices', 'embeddings', 'head') for p in groups[key])
    assert all(p.ndim < 2 for p in groups['gains_biases'])

    names = {id(p): n for n, p in model.named_parameters()}
    embedding_names = {names[id(p)] for p in groups['embeddings']}
    if model_type == 'gpt2':
        assert embedding_names == {'transformer.wte.weight', 'transformer.wpe.weight'}
        assert groups['head'] == []  # tied to wte
        assert len(groups['hidden_matrices']) == 4 * TINY['n_layer']
    else:
        assert embedding_names == {'tok_embeddings.weight'}
        assert len(groups['head']) == (0 if kwargs.get('tie_weights') else 1)
        assert len(groups['hidden_matrices']) == 7 * TINY['n_layer']


@pytest.mark.parametrize('model_type', ['gpt2', 'llama'])
@pytest.mark.parametrize('name', ['adamw', 'muon', 'sgd'])
def test_builders_cover_every_parameter_once(model_type, name):
    model = build_model(model_type, **TINY)
    optimizer = build_optimizer(name, model, CFG, 'cpu')
    ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters()}
    for group in optimizer.param_groups:
        if all(p.ndim < 2 for p in group['params']):
            assert group['weight_decay'] == 0.0  # no weight decay on gains and biases
    if name == 'muon':
        hidden = {id(p) for p in split_params(model)['hidden_matrices']}
        muon_ids = {id(p) for g in optimizer.param_groups if g['use_muon'] for p in g['params']}
        assert muon_ids == hidden
        assert [g['lr'] for g in optimizer.param_groups] == [CFG.muon_lr, CFG.learning_rate, CFG.learning_rate]


@pytest.mark.parametrize('shape', [(64, 64), (32, 128), (128, 32)])
def test_newton_schulz_orthogonalizes(shape):
    torch.manual_seed(0)
    G = torch.randn(*shape)
    X = zeropower_via_newtonschulz5(G, steps=5)
    assert X.shape == G.shape
    # singular values end up near 1 (the quintic iteration is deliberately not exact);
    # only directions whose singular value was tiny to begin with may stay below
    singular_values = torch.linalg.svdvals(X.float())
    assert singular_values.max() < 1.3
    assert (singular_values > 0.5).float().mean() >= 0.95
    # and X is close to the exact orthogonal polar factor U V^T of G
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    cosine = (X.float() * (U @ Vh)).sum() / (X.float().norm() * (U @ Vh).norm())
    assert cosine > 0.95
    # invariant to the scale of the input
    assert torch.allclose(zeropower_via_newtonschulz5(1000.0 * G), X, atol=1e-3)


def test_muon_first_step_matches_definition():
    torch.manual_seed(0)
    for shape in [(48, 16), (16, 48)]:
        w = torch.nn.Parameter(torch.randn(*shape))
        w0 = w.detach().clone()
        w.grad = torch.randn(*shape)
        lr, wd = 0.05, 0.1
        Muon([dict(params=[w], use_muon=True, lr=lr, weight_decay=wd)]).step()
        # with zero-initialized momentum the Nesterov update direction is proportional to the gradient
        ortho = zeropower_via_newtonschulz5(w.grad)
        scale = max(1.0, shape[0] / shape[1]) ** 0.5
        expected = w0 * (1 - lr * wd) - lr * scale * ortho
        assert torch.allclose(w, expected, atol=1e-6)


def test_muon_momentum_recursion():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = Muon([dict(params=[w], use_muon=True, momentum=0.9, nesterov=False)])
    expected = torch.zeros(8, 8)
    for _ in range(3):
        w.grad = torch.randn(8, 8)
        optimizer.step()
        expected = 0.9 * expected + 0.1 * w.grad
    assert torch.allclose(optimizer.state[w]['momentum_buffer'], expected, atol=1e-6)


def test_muon_does_not_modify_gradients():
    w = torch.nn.Parameter(torch.randn(8, 4))
    w.grad = torch.randn(8, 4)
    grad = w.grad.clone()
    optimizer = Muon([dict(params=[w], use_muon=True)])
    optimizer.step()
    optimizer.step()
    assert torch.equal(w.grad, grad)


def test_muon_validates_param_groups():
    with pytest.raises(ValueError):
        Muon([dict(params=[torch.nn.Parameter(torch.zeros(4))], use_muon=True)])
    with pytest.raises(ValueError):
        Muon([dict(params=[torch.nn.Parameter(torch.zeros(4, 4))])])


def test_auxiliary_adam_matches_torch_adamw():
    """The use_muon=False branch must be exactly AdamW."""
    torch.manual_seed(0)
    a = [torch.nn.Parameter(torch.randn(10, 5, dtype=torch.float64)),
         torch.nn.Parameter(torch.randn(7, dtype=torch.float64))]
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    hyper = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ours = Muon([dict(params=a, use_muon=False, **hyper)])
    reference = torch.optim.AdamW(b, **hyper)
    for _ in range(20):
        grads = [torch.randn_like(p) for p in a]
        for pa, pb, g in zip(a, b, grads):
            pa.grad, pb.grad = g.clone(), g.clone()
        ours.step()
        reference.step()
    for pa, pb in zip(a, b):
        assert torch.allclose(pa, pb, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('model_type', ['gpt2', 'llama'])
@pytest.mark.parametrize('name', ['adamw', 'muon'])
def test_optimizers_reduce_the_loss(model_type, name):
    torch.manual_seed(0)
    model = build_model(model_type, **TINY)
    optimizer = build_optimizer(name, model, CFG, 'cpu')
    x = torch.randint(0, 128, (8, 32))
    y = torch.randint(0, 128, (8, 32))
    _, first = model(x, y)
    for _ in range(30):
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    _, last = model(x, y)
    assert last.item() < first.item() - 0.5


@pytest.mark.parametrize('name', ['adamw', 'muon', 'sgd'])
def test_state_dict_roundtrip_resumes_exactly(name):
    """Saving and reloading model + optimizer must not change the training trajectory."""
    torch.manual_seed(0)
    batches = [(torch.randint(0, 128, (4, 32)), torch.randint(0, 128, (4, 32))) for _ in range(6)]

    def train(model, optimizer, data):
        for x, y in data:
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    torch.manual_seed(1)
    model = build_model('gpt2', **TINY)
    optimizer = build_optimizer(name, model, CFG, 'cpu')
    train(model, optimizer, batches[:3])
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    train(model, optimizer, batches[3:])

    resumed = build_model('gpt2', **TINY)
    resumed.load_state_dict(model_state)
    resumed_optimizer = build_optimizer(name, resumed, CFG, 'cpu')
    resumed_optimizer.load_state_dict(optimizer_state)
    train(resumed, resumed_optimizer, batches[3:])

    for p, q in zip(model.parameters(), resumed.parameters()):
        assert torch.allclose(p, q, atol=1e-7)


def test_template_sgd_matches_torch_sgd():
    """The template optimizer without weight decay equals torch.optim.SGD (Nesterov)."""
    from optimizers.template import SGDMomentum
    torch.manual_seed(0)
    a = [torch.nn.Parameter(torch.randn(6, 3, dtype=torch.float64))]
    b = [torch.nn.Parameter(a[0].detach().clone())]
    ours = SGDMomentum(a, lr=0.1, momentum=0.9, nesterov=True)
    reference = torch.optim.SGD(b, lr=0.1, momentum=0.9, nesterov=True)
    for _ in range(10):
        g = torch.randn(6, 3, dtype=torch.float64)
        a[0].grad, b[0].grad = g.clone(), g.clone()
        ours.step()
        reference.step()
    assert torch.allclose(a[0], b[0], rtol=1e-12)
