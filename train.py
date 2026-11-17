"""
Train a GPT-2 or LLaMA language model from scratch with a pluggable optimizer.

Single GPU (or CPU):
    python train.py config/gpt2_small_adamw.py

Multiple GPUs on one node:
    torchrun --standalone --nproc_per_node=8 train.py config/gpt2_small_adamw.py

Multiple nodes (run on every node, see README):
    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_endpoint=<host>:<port> \
        train.py config/gpt2_small_adamw.py

Every key in `defaults` below can be overridden by a config file and/or --key=value.
The training loop follows nanoGPT (https://github.com/karpathy/nanoGPT) and Sophia
(https://github.com/Liuhong99/Sophia).
"""

import json
import math
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from configurator import load_config
from models import build_model
from optimizers import available_optimizers, build_optimizer

# -----------------------------------------------------------------------------
# Default configuration: GPT-2 small (124M) on OpenWebText with AdamW.
defaults = dict(
    # I/O
    out_dir='',  # '' -> out/<dataset>/<run_name>
    run_name='gpt2_small_adamw',  # also the wandb run name
    init_from='scratch',  # 'scratch' or 'resume' (continue from <out_dir>/ckpt.pt)
    eval_interval=1000,  # evaluate (and checkpoint) every this many iterations
    eval_iters=200,  # batches per split per evaluation
    log_interval=10,
    eval_only=False,  # evaluate once and exit (use with init_from='resume')
    always_save_checkpoint=True,  # False: only checkpoint when the val loss improves
    # wandb logging (optional; a JSON-lines log is always written to <out_dir>/log.jsonl)
    wandb_log=False,
    wandb_project='optimizer',
    # data
    data_dir='data',  # tokens are read from <data_dir>/<dataset>/{train,val}.bin
    dataset='openwebtext',
    batch_size=12,  # sequences per forward pass on each GPU
    block_size=1024,  # context length in tokens
    gradient_accumulation_steps=40,  # forward passes per optimizer step, summed over all GPUs
    # model
    model_type='gpt2',  # 'gpt2' or 'llama'
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=False,  # gpt2 only: biases in Linear and LayerNorm
    tie_weights=False,  # llama only: tie input embedding and output head (gpt2 always ties)
    vocab_size=None,  # None: tokenizer vocab size from meta.json, rounded up to a multiple of 64
    # optimization
    optimizer='adamw',  # a name registered in optimizers/
    max_iters=100000,  # number of optimizer steps (also the length of the LR schedule)
    stop_iter=-1,  # >0: evaluate, checkpoint and exit at this iteration (resume later with
                   # init_from='resume'); for jobs with a time limit. -1: run to max_iters
    learning_rate=6e-4,  # peak LR of AdamW (for muon: of the auxiliary AdamW)
    weight_decay=1e-1,  # AdamW decoupled weight decay (applied to >=2-D parameters)
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
    grad_clip=1.0,  # clip the global gradient norm at this value; 0.0 disables clipping
    # Muon (only used with optimizer='muon')
    muon_lr=0.02,  # peak LR of Muon for the hidden weight matrices
    muon_momentum=0.95,
    muon_nesterov=True,
    muon_ns_steps=5,  # Newton-Schulz iterations
    muon_weight_decay=0.01,
    # learning-rate schedule: linear warmup, then decay to min_lr_ratio * peak at max_iters.
    # It scales the peak LR of every param group, so it applies to all optimizers.
    lr_schedule='cosine',  # 'cosine', 'linear' or 'constant'
    warmup_iters=2000,
    min_lr_ratio=0.1,
    # system
    device='cuda' if torch.cuda.is_available() else 'cpu',  # 'cuda', 'cuda:0', 'cpu', ...
    dtype='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
    compile=True,  # torch.compile the model
    seed=1337,
)
# -----------------------------------------------------------------------------

LR_SCHEDULES = ('cosine', 'linear', 'constant')


def lr_multiplier(it, cfg):
    """Factor in (0, 1] by which every param group's peak LR is scaled at iteration `it`."""
    if it < cfg.warmup_iters:
        return (it + 1) / (cfg.warmup_iters + 1)
    if cfg.lr_schedule == 'constant':
        return 1.0
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    progress = min(max(progress, 0.0), 1.0)
    if cfg.lr_schedule == 'cosine':
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:  # linear
        coeff = 1.0 - progress
    return cfg.min_lr_ratio + coeff * (1.0 - cfg.min_lr_ratio)


def model_args_from_config(cfg, vocab_size):
    args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                block_size=cfg.block_size, vocab_size=vocab_size, dropout=cfg.dropout)
    if cfg.model_type == 'gpt2':
        args['bias'] = cfg.bias
    elif cfg.model_type == 'llama':
        args['tie_weights'] = cfg.tie_weights
    return args


def truncate_log(path, iter_num):
    """Keep only log records from before `iter_num` (used when resuming a run)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        records = [line for line in f if line.strip() and json.loads(line)['iter'] < iter_num]
    with open(path, 'w') as f:
        f.writelines(records)


def main():
    cfg_dict = load_config(defaults)
    cfg = SimpleNamespace(**cfg_dict)
    if cfg.init_from not in ('scratch', 'resume'):
        raise ValueError(f"init_from must be 'scratch' or 'resume', got {cfg.init_from!r}")
    if cfg.lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {cfg.lr_schedule!r}")
    if cfg.dtype not in ('float32', 'bfloat16', 'float16'):
        raise ValueError(f"dtype must be float32, bfloat16 or float16, got {cfg.dtype!r}")
    if cfg.optimizer not in available_optimizers():
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}; available: {available_optimizers()}")
    if not 0.0 <= cfg.min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {cfg.min_lr_ratio}")
    if cfg.vocab_size is not None and not (isinstance(cfg.vocab_size, int) and cfg.vocab_size > 0):
        raise ValueError(f"vocab_size must be None or a positive int, got {cfg.vocab_size!r}")
    for key in ('max_iters', 'batch_size', 'block_size', 'gradient_accumulation_steps',
                'eval_interval', 'eval_iters', 'log_interval'):
        if getattr(cfg, key) <= 0:
            raise ValueError(f"{key} must be positive, got {getattr(cfg, key)}")
    if cfg.seed < 0:
        raise ValueError(f"seed must be non-negative, got {cfg.seed}")

    # ---- distributed setup --------------------------------------------------
    ddp = int(os.environ.get('RANK', -1)) != -1  # set by torchrun
    if ddp:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if cfg.device.startswith('cuda'):
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(f"local rank {local_rank} has no GPU: only "
                                   f"{torch.cuda.device_count()} visible on this node")
            device = f'cuda:{local_rank}'
            torch.cuda.set_device(device)
        else:
            device = cfg.device
        # generous timeout: rank 0 alone writes checkpoints while the others wait
        init_process_group(backend='nccl' if device.startswith('cuda') else 'gloo',
                           timeout=timedelta(minutes=60))
    else:
        rank, world_size, device = 0, 1, cfg.device
    master_process = rank == 0
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'

    # gradient_accumulation_steps counts forward passes over all GPUs, so the effective
    # batch size (and thus the optimization problem) does not depend on the GPU count
    if cfg.gradient_accumulation_steps % world_size != 0:
        raise ValueError(f"gradient_accumulation_steps ({cfg.gradient_accumulation_steps}) "
                         f"must be divisible by the number of processes ({world_size})")
    accum_steps = cfg.gradient_accumulation_steps // world_size

    out_dir = cfg.out_dir or os.path.join('out', cfg.dataset, cfg.run_name)
    cfg.out_dir = cfg_dict['out_dir'] = out_dir
    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        if master_process:
            print(msg, flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_autocast = device_type == 'cuda' and cfg.dtype != 'float32'
    ctx = (torch.autocast(device_type='cuda', dtype=getattr(torch, cfg.dtype))
           if use_autocast else nullcontext())

    # ---- data ---------------------------------------------------------------
    data_path = os.path.join(cfg.data_dir, cfg.dataset)
    bin_paths = {split: os.path.join(data_path, f'{split}.bin') for split in ('train', 'val')}
    for path in bin_paths.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; run `python data/prepare.py "
                                    f"{cfg.dataset}` first (see README)")
    meta_path = os.path.join(data_path, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        data_dtype, data_vocab_size = np.dtype(meta['dtype']), meta['vocab_size']
    else:  # plain nanoGPT / Sophia data directory: GPT-2 BPE tokens stored as uint16
        log(f"no meta.json in {data_path}; assuming uint16 GPT-2 BPE tokens")
        data_dtype, data_vocab_size = np.dtype(np.uint16), 50257

    # ---- model --------------------------------------------------------------
    iter_num, best_val_loss, train_time = 0, float('inf'), 0.0
    checkpoint = None
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if cfg.init_from == 'scratch':
        vocab_size = cfg.vocab_size
        if vocab_size is None:
            vocab_size = 64 * math.ceil(data_vocab_size / 64)
        model_type, model_args = cfg.model_type, model_args_from_config(cfg, vocab_size)
    else:
        log(f"resuming from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        if checkpoint['config']['optimizer'] != cfg.optimizer:
            raise ValueError(f"checkpoint was trained with optimizer "
                             f"{checkpoint['config']['optimizer']!r}, not {cfg.optimizer!r}")
        # the architecture always comes from the checkpoint
        model_type, model_args = checkpoint['model_type'], checkpoint['model_args']
        cfg.model_type = cfg_dict['model_type'] = model_type
        cfg.block_size = cfg_dict['block_size'] = model_args['block_size']
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        train_time = checkpoint['train_time']
    if model_args['vocab_size'] < data_vocab_size:
        raise ValueError(f"model vocab_size {model_args['vocab_size']} is smaller than the "
                         f"tokenizer vocab size {data_vocab_size} of {data_path}")

    torch.manual_seed(cfg.seed)  # same initialization on every rank and for every optimizer
    model = build_model(model_type, **model_args)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    raw_model = model  # the plain nn.Module: used for the optimizer and for checkpoints
    torch.manual_seed(cfg.seed + 1 + rank)  # dropout masks differ across ranks

    # ---- optimizer ----------------------------------------------------------
    optimizer = build_optimizer(cfg.optimizer, raw_model, cfg, device_type)
    # peak learning rates; the schedule rescales them every iteration
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler(device_type, enabled=(use_autocast and cfg.dtype == 'float16'))
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
    checkpoint = None  # free memory

    tokens_per_iter = cfg.gradient_accumulation_steps * cfg.batch_size * cfg.block_size
    log(f"model: {model_type} {model_args}")
    log(f"parameters: {raw_model.get_num_params() / 1e6:.2f}M")
    log(f"optimizer: {cfg.optimizer}, param groups: " + ", ".join(
        f"[{len(g['params'])} tensors, {sum(p.numel() for p in g['params']) / 1e6:.2f}M params, "
        f"peak lr {lr:g}, wd {g.get('weight_decay', 0):g}]"
        for g, lr in zip(optimizer.param_groups, base_lrs)))
    log(f"processes: {world_size}, device: {device}, dtype: {cfg.dtype if use_autocast else 'float32'}")
    log(f"tokens per optimizer step: {tokens_per_iter:,} "
        f"({world_size} procs x {accum_steps} accumulation steps x {cfg.batch_size} x {cfg.block_size})")

    if cfg.compile:
        log("compiling the model (takes about a minute)...")
        model = torch.compile(model)
    eval_model = model  # compiled, but not wrapped in DDP
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device_type == 'cuda' else None)

    # ---- batches ------------------------------------------------------------
    for split, path in bin_paths.items():
        n_tokens = os.path.getsize(path) // data_dtype.itemsize
        if n_tokens <= cfg.block_size + 1:
            raise ValueError(f"{path} has only {n_tokens} tokens; need more than block_size + 1")
        log(f"{split} data: {n_tokens:,} tokens ({path})")

    def get_batch(split, key):
        """
        `batch_size` random windows of `split`. The windows are a deterministic function
        of `key` alone, so a batch does not depend on which process loads it or on
        anything that happened before (in particular not on resuming).
        """
        seed = int(np.random.SeedSequence([cfg.seed, *key]).generate_state(1)[0])
        generator = torch.Generator().manual_seed(seed)
        # np.memmap is re-created for every batch to avoid a memory leak (see nanoGPT)
        data = np.memmap(bin_paths[split], dtype=data_dtype, mode='r')
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,), generator=generator).tolist()
        x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device)

    # An optimizer step consumes `gradient_accumulation_steps` micro-batches, numbered
    # 0..gradient_accumulation_steps-1; process `rank` handles numbers
    # rank*accum_steps ... (rank+1)*accum_steps-1. Micro-batch j of step `it` has key
    # (0, it, j), so a step sees exactly the same sequences for any number of GPUs.
    def train_key(it, micro_step):
        return (0, it, rank * accum_steps + micro_step)

    def all_reduce_mean(value):
        if not ddp:
            return value
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(tensor)
        return tensor.item() / world_size

    @torch.no_grad()
    def estimate_loss():
        """
        Mean loss over `eval_iters` fixed batches per split. The same batches are used at
        every evaluation and in every run with the same seed, batch_size and block_size;
        the work is divided over all processes.
        """
        out = {}
        eval_model.eval()
        for split_id, split in enumerate(('train', 'val')):
            total = torch.zeros((), dtype=torch.float64, device=device)
            for k in range(rank, cfg.eval_iters, world_size):
                x, y = get_batch(split, (1, split_id, k))
                with ctx:
                    _, loss = eval_model(x, y)
                total += loss.double()
            if ddp:
                torch.distributed.all_reduce(total)
            out[split] = total.item() / cfg.eval_iters
        eval_model.train()
        return out

    # ---- logging ------------------------------------------------------------
    log_file = None
    if master_process and not cfg.eval_only:
        with open(os.path.join(out_dir, 'config.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=2)
        log_path = os.path.join(out_dir, 'log.jsonl')
        if cfg.init_from == 'resume':
            truncate_log(log_path, iter_num)
        log_file = open(log_path, 'a' if cfg.init_from == 'resume' else 'w', buffering=1)
    use_wandb = cfg.wandb_log and master_process and not cfg.eval_only
    if use_wandb:
        import wandb
        # a resumed run continues the same wandb run; a run from scratch always starts a new one
        wandb_id_path = os.path.join(out_dir, 'wandb_id.txt')
        wandb_id = None
        if cfg.init_from == 'resume' and os.path.exists(wandb_id_path):
            with open(wandb_id_path) as f:
                wandb_id = f.read().strip()
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg_dict,
                   id=wandb_id, resume='allow' if wandb_id else None)
        with open(wandb_id_path, 'w') as f:
            f.write(wandb.run.id)

    def write_log(record):
        if log_file is not None:
            log_file.write(json.dumps(record) + '\n')
        if use_wandb:
            wandb.log({k: v for k, v in record.items() if k != 'iter'}, step=record['iter'])

    # ---- training loop ------------------------------------------------------
    # `train_time` is the cumulative wall-clock time of the training iterations;
    # evaluation and checkpointing are excluded (compilation is included).
    start_iter = iter_num
    model.train()
    x, y = get_batch('train', train_key(iter_num, 0))
    loss_sum = torch.zeros((), device=device)  # sum of micro-batch losses since the last log line
    last_log_time, iters_since_log = train_time, 0
    t0 = time.perf_counter()
    while True:
        # evaluate and checkpoint (every process evaluates; rank 0 writes)
        stopping = iter_num >= cfg.max_iters or iter_num == cfg.stop_iter or cfg.eval_only
        if iter_num % cfg.eval_interval == 0 or stopping:
            if device_type == 'cuda':
                torch.cuda.synchronize()
            train_time += time.perf_counter() - t0
            losses = estimate_loss()
            log(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                f"val ppl {math.exp(losses['val']):.2f}, train time {train_time / 3600:.2f}h")
            if not cfg.eval_only:
                write_log({'iter': iter_num, 'eval_train_loss': losses['train'],
                           'val_loss': losses['val'], 'train_time': train_time})
                improved = losses['val'] < best_val_loss
                best_val_loss = min(best_val_loss, losses['val'])
                if master_process and iter_num > start_iter and (improved or cfg.always_save_checkpoint):
                    ckpt = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'model_type': model_type,
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'train_time': train_time,
                        'config': cfg_dict,
                    }
                    try:
                        torch.save(ckpt, ckpt_path + '.tmp')
                        os.replace(ckpt_path + '.tmp', ckpt_path)  # atomic: never a half-written file
                    except BaseException:
                        if os.path.exists(ckpt_path + '.tmp'):
                            os.remove(ckpt_path + '.tmp')  # e.g. disk full: keep the previous checkpoint
                        raise
                    log(f"saved checkpoint to {ckpt_path}")
                    del ckpt
            t0 = time.perf_counter()  # restart the clock: evaluation is not training time
        if stopping:
            break

        # learning rate for this iteration
        lr_mult = lr_multiplier(iter_num, cfg)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = base_lr * lr_mult

        # forward/backward, accumulating gradients over `accum_steps` micro-batches
        for micro_step in range(accum_steps):
            if ddp:
                # only synchronize (average) gradients across ranks at the last micro-step
                model.require_backward_grad_sync = (micro_step == accum_steps - 1)
            with ctx:
                _, loss = model(x, y)
            loss_sum += loss.detach()
            # the accumulated gradient is the mean over all micro-batches (DDP averages
            # over processes, the division by accum_steps averages within a process)
            scaled_loss = loss / accum_steps
            # fetch the next batch while the GPU is busy with the forward pass
            if micro_step + 1 < accum_steps:
                x, y = get_batch('train', train_key(iter_num, micro_step + 1))
            else:
                x, y = get_batch('train', train_key(iter_num + 1, 0))
            scaler.scale(scaled_loss).backward()
        # unscale (no-op unless float16), then clip the global gradient norm
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float('inf'))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        iters_since_log += 1

        # logging: every process takes part in the loss all-reduce
        if iter_num % cfg.log_interval == 0:
            # mean loss over all micro-batches of all processes since the last log line
            train_loss = all_reduce_mean(loss_sum.item() / (iters_since_log * accum_steps))
            loss_sum.zero_()
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"non-finite training loss at iteration {iter_num}")
            t1 = time.perf_counter()
            train_time += t1 - t0
            t0 = t1
            dt = (train_time - last_log_time) / iters_since_log  # seconds per iteration
            last_log_time, iters_since_log = train_time, 0
            mem = (f", peak mem {torch.cuda.max_memory_allocated(device) / 2**30:.1f}GiB"
                   if device_type == 'cuda' else '')
            log(f"iter {iter_num}: loss {train_loss:.4f}, grad norm {grad_norm.item():.3f}, "
                f"lr x{lr_mult:.4f}, {dt * 1000:.0f} ms/iter, {tokens_per_iter / dt / 1e3:.0f}k tok/s{mem}")
            write_log({'iter': iter_num, 'train_loss': train_loss, 'grad_norm': grad_norm.item(),
                       'lr_mult': lr_mult, 'train_time': train_time})
        iter_num += 1

    if log_file is not None:
        log_file.close()
    if use_wandb:
        wandb.finish()
    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    main()
