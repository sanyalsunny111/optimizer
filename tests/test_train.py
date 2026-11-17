"""End-to-end tests: run train.py as a subprocess on a tiny synthetic dataset (CPU)."""

import glob
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train
from configurator import load_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = ['--n_layer=2', '--n_head=2', '--n_embd=32', '--block_size=16', '--batch_size=4',
        '--gradient_accumulation_steps=2', '--warmup_iters=2', '--eval_interval=5', '--eval_iters=2',
        '--log_interval=1', '--device=cpu', '--compile=False', '--dataset=synthetic']


@pytest.fixture
def data_dir(tmp_path):
    """A learnable token stream: a short pattern repeated many times."""
    directory = tmp_path / 'data' / 'synthetic'
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 100, size=37)
    np.tile(pattern, 200).astype(np.uint16).tofile(directory / 'train.bin')
    np.tile(pattern, 20).astype(np.uint16).tofile(directory / 'val.bin')
    (directory / 'meta.json').write_text(json.dumps({'vocab_size': 100, 'dtype': 'uint16'}))
    return str(tmp_path / 'data')


def run(args, launcher=()):
    cmd = [sys.executable, *launcher, 'train.py', *TINY, *args]
    env = {k: v for k, v in os.environ.items() if k not in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')}
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def read_log(out_dir):
    with open(os.path.join(str(out_dir), 'log.jsonl')) as f:
        return [json.loads(line) for line in f]


@pytest.mark.parametrize('model_type', ['gpt2', 'llama'])
@pytest.mark.parametrize('optimizer', ['adamw', 'muon'])
def test_train_and_resume(tmp_path, data_dir, model_type, optimizer):
    out_dir = str(tmp_path / 'out')
    common = [f'--data_dir={data_dir}', f'--out_dir={out_dir}', f'--model_type={model_type}',
              f'--optimizer={optimizer}', '--learning_rate=3e-3']
    run(common + ['--max_iters=10'])

    records = read_log(out_dir)
    evals = [r for r in records if 'val_loss' in r]
    steps = [r for r in records if 'train_loss' in r]
    assert [r['iter'] for r in evals] == [0, 5, 10]
    assert [r['iter'] for r in steps] == list(range(10))  # exactly max_iters optimizer steps
    assert abs(evals[0]['val_loss'] - np.log(128)) < 0.3  # vocab 100 is padded to 128
    assert evals[-1]['val_loss'] < evals[0]['val_loss'] - 0.2  # it learns
    assert steps[0]['lr_mult'] == pytest.approx(1 / 3) and steps[1]['lr_mult'] == pytest.approx(2 / 3)
    assert all(r['grad_norm'] > 0 for r in steps)

    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), weights_only=True)
    assert checkpoint['iter_num'] == 10
    assert checkpoint['model_type'] == model_type
    assert checkpoint['model_args']['vocab_size'] == 128
    assert not any(k.startswith('_orig_mod') or k.startswith('module.') for k in checkpoint['model'])
    assert not glob.glob(os.path.join(out_dir, '*.tmp'))

    # resume and train further; the log continues without duplicated iterations
    run(common + ['--max_iters=15', '--init_from=resume'])
    records = read_log(out_dir)
    assert [r['iter'] for r in records if 'val_loss' in r] == [0, 5, 10, 15]
    assert [r['iter'] for r in records if 'train_loss' in r] == list(range(15))
    assert torch.load(os.path.join(out_dir, 'ckpt.pt'), weights_only=True)['iter_num'] == 15

    # evaluation only: reproduces the last logged validation loss and leaves the log untouched
    stdout = run([f'--data_dir={data_dir}', f'--out_dir={out_dir}', f'--optimizer={optimizer}',
                  '--init_from=resume', '--eval_only=True'])
    final = [r for r in records if 'val_loss' in r][-1]
    assert f"val loss {final['val_loss']:.4f}" in stdout
    assert read_log(out_dir) == records


def test_same_seed_is_reproducible_and_seeds_differ(tmp_path, data_dir):
    losses = []
    for name, seed in (('a', 1), ('b', 1), ('c', 2)):
        out_dir = str(tmp_path / name)
        run([f'--data_dir={data_dir}', f'--out_dir={out_dir}', '--max_iters=5', f'--seed={seed}'])
        losses.append([r['train_loss'] for r in read_log(out_dir) if 'train_loss' in r])
    assert losses[0] == losses[1]
    assert losses[0] != losses[2]


@pytest.mark.parametrize('optimizer', ['adamw', 'muon'])
def test_ddp_matches_single_process(tmp_path, data_dir, optimizer):
    """
    2 processes x 2 accumulation steps must compute the same optimizer steps as
    1 process x 4 accumulation steps: same data, same gradient averaging.
    """
    common = [f'--data_dir={data_dir}', '--max_iters=10', f'--optimizer={optimizer}',
              '--learning_rate=3e-3', '--gradient_accumulation_steps=4']
    run(common + [f'--out_dir={tmp_path / "single"}'])
    launcher = ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    stdout = run(common + [f'--out_dir={tmp_path / "ddp"}'], launcher)
    assert 'processes: 2' in stdout
    assert stdout.count('step 10:') == 1  # only rank 0 prints and logs

    single, ddp = read_log(tmp_path / 'single'), read_log(tmp_path / 'ddp')
    assert [r['iter'] for r in single] == [r['iter'] for r in ddp]
    for a, b in zip(single, ddp):
        for key in ('train_loss', 'val_loss', 'eval_train_loss', 'grad_norm'):
            if key in a:
                assert b[key] == pytest.approx(a[key], rel=1e-4, abs=1e-5), (a, b)
    evals = [r for r in ddp if 'val_loss' in r]
    assert evals[-1]['val_loss'] < evals[0]['val_loss'] - 0.2


def test_resume_matches_uninterrupted_run(tmp_path, data_dir):
    common = [f'--data_dir={data_dir}', '--optimizer=muon', '--learning_rate=3e-3']
    run(common + [f'--out_dir={tmp_path / "full"}', '--max_iters=10'])
    # stop_iter keeps the LR schedule of max_iters=10; 7 is not an eval/checkpoint step
    run(common + [f'--out_dir={tmp_path / "resumed"}', '--max_iters=10', '--stop_iter=7'])
    assert torch.load(tmp_path / 'resumed' / 'ckpt.pt', weights_only=True)['iter_num'] == 7
    run(common + [f'--out_dir={tmp_path / "resumed"}', '--max_iters=10', '--init_from=resume'])
    full, resumed = read_log(tmp_path / 'full'), read_log(tmp_path / 'resumed')
    # the resumed log has one extra evaluation, at the stopping point
    resumed = [r for r in resumed if not (r['iter'] == 7 and 'val_loss' in r)]
    assert [r['iter'] for r in full] == [r['iter'] for r in resumed]
    for a, b in zip(full, resumed):
        for key in ('train_loss', 'val_loss', 'grad_norm'):
            if key in a:
                assert b[key] == pytest.approx(a[key], rel=1e-5, abs=1e-6), (a, b)


def test_ddp_requires_divisible_accumulation_steps(tmp_path, data_dir):
    launcher = ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    cmd = [sys.executable, *launcher, 'train.py', *TINY, f'--data_dir={data_dir}',
           f'--out_dir={tmp_path / "out"}', '--gradient_accumulation_steps=3']
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0
    assert 'must be divisible' in result.stderr


def test_missing_data_gives_a_helpful_error(tmp_path):
    cmd = [sys.executable, 'train.py', *TINY, f'--data_dir={tmp_path}']
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0
    assert 'data/prepare.py' in result.stderr


@pytest.mark.parametrize('schedule', ['cosine', 'linear', 'constant'])
def test_lr_multiplier(schedule):
    cfg = SimpleNamespace(lr_schedule=schedule, warmup_iters=10, max_iters=110, min_lr_ratio=0.1)
    values = [train.lr_multiplier(it, cfg) for it in range(110)]
    assert values[0] == pytest.approx(1 / 11)  # never zero, so the first step is not wasted
    assert all(b > a for a, b in zip(values[:10], values[1:11]))  # linear warmup
    assert values[10] == pytest.approx(1.0)
    assert max(values) == pytest.approx(1.0)
    if schedule == 'constant':
        assert all(v == 1.0 for v in values[10:])
    else:
        assert all(b < a for a, b in zip(values[10:], values[11:]))  # monotone decay
        assert train.lr_multiplier(110, cfg) == pytest.approx(0.1)
        midpoint = train.lr_multiplier(60, cfg)
        assert midpoint == pytest.approx(0.55)  # both schedules are halfway at the midpoint


def test_shipped_configs_are_valid():
    """Every file in config/ only uses known keys with the right types."""
    paths = sorted(glob.glob(os.path.join(REPO_ROOT, 'config', '*.py')))
    assert paths
    for path in paths:
        config = load_config(train.defaults, [path])
        assert config['run_name'] == os.path.splitext(os.path.basename(path))[0]
        assert config['n_embd'] % config['n_head'] == 0
        assert (config['n_embd'] // config['n_head']) % 2 == 0  # RoPE needs an even head dim
        assert config['batch_size'] * config['gradient_accumulation_steps'] == 480


@pytest.mark.parametrize('arg', ['--min_lr_ratio=1.5', "--vocab_size='abc'", '--max_iters=0',
                                 '--eval_iters=0', '--seed=-1', '--optimizer=nope', '--dtype=float64'])
def test_invalid_config_values_are_rejected(tmp_path, data_dir, arg):
    cmd = [sys.executable, 'train.py', *TINY, f'--data_dir={data_dir}', f'--out_dir={tmp_path}', arg]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0
    assert 'ValueError' in result.stderr


def test_more_processes_than_eval_batches(tmp_path, data_dir):
    """eval_iters=1 with 2 processes: one rank evaluates nothing, the mean must still be right."""
    common = [f'--data_dir={data_dir}', '--max_iters=5', '--eval_iters=1', '--gradient_accumulation_steps=2']
    run(common + [f'--out_dir={tmp_path / "single"}'])
    launcher = ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    run(common + [f'--out_dir={tmp_path / "ddp"}'], launcher)
    a = [r['val_loss'] for r in read_log(tmp_path / 'single') if 'val_loss' in r]
    b = [r['val_loss'] for r in read_log(tmp_path / 'ddp') if 'val_loss' in r]
    assert b == pytest.approx(a, rel=1e-4)
