# optimizer

A small, end-to-end research code base for **comparing optimizers on language-model
pretraining**. It trains GPT-2 and LLaMA-style models from scratch on OpenWebText,
FineWeb or C4, on one GPU or many, with **AdamW** and **Muon** as baselines. A new
optimizer is added with one file.

The design follows the [Sophia](https://github.com/Liuhong99/Sophia) repository, which in
turn builds on [nanoGPT](https://github.com/karpathy/nanoGPT): plain PyTorch, no trainer
framework, one training script, Python config files. Several things are changed so that
optimizer comparisons are controlled and reproducible:

* **Every optimizer sees the same data and the same initial weights.** Model initialization
  depends only on `seed`. Each training micro-batch is a deterministic function of
  `(seed, iteration, micro-batch index)`, so a step consumes exactly the same sequences
  whatever optimizer, number of GPUs or resume history the run has.
* **The number of GPUs does not change the optimization problem.**
  `gradient_accumulation_steps` is the total number of micro-batches per step, summed over
  all GPUs. A run on 8 GPUs computes the same steps as on 1 GPU (a test checks this).
* **Evaluation is identical across runs.** Validation loss is measured on a fixed set of
  batches, split over all GPUs.
* **Resuming is exact.** A run that is stopped and resumed produces the same log as an
  uninterrupted one (a test checks this). `stop_iter` lets you split a run into jobs with a
  time limit.
* **The logs are suitable for a paper.** `log.jsonl` records the loss averaged over the
  whole batch, the gradient norm, the LR and the training wall-clock time (excluding
  evaluation). `plot.py` plots against steps, tokens or time.
* **Configuration errors fail loudly.** A misspelled hyperparameter name or a wrong type
  raises an error instead of being silently ignored.

---

## Contents

1. [Repository layout](#repository-layout)
2. [Installation](#installation)
3. [Data preparation](#data-preparation)
4. [Training](#training) (single GPU, multiple GPUs, multiple nodes, SLURM)
5. [Models and configs](#models-and-configs)
6. [Baseline optimizers](#baseline-optimizers)
7. [Adding your own optimizer](#adding-your-own-optimizer)
8. [Logging, evaluation and plotting](#logging-evaluation-and-plotting)
9. [Running a fair comparison](#running-a-fair-comparison)
10. [Tests](#tests)
11. [Configuration reference](#configuration-reference)
12. [Acknowledgements](#acknowledgements)

---

## Repository layout

```
optimizer/
├── train.py                 training loop (single GPU, DDP, multi-node)
├── configurator.py          config files + --key=value overrides, with type checking
├── plot.py                  loss curves from one or more runs
├── models/
│   ├── gpt.py               GPT-2 (learned positions, LayerNorm, GELU, tied embeddings)
│   └── llama.py             LLaMA (RoPE, RMSNorm, SwiGLU, no biases)
├── optimizers/
│   ├── registry.py          @register decorator, parameter grouping (split_params)
│   ├── adamw.py             AdamW baseline
│   ├── muon.py              Muon baseline (+ auxiliary AdamW)
│   └── template.py          SGD with momentum: a template for your own optimizer
├── config/                  12 experiment configs: {gpt2,llama}_{small,medium,large}_{adamw,muon}.py
├── data/prepare.py          download + tokenize OpenWebText / FineWeb / C4 / Shakespeare
├── scripts/
│   ├── run_baselines.sh     AdamW and Muon for one model and dataset
│   └── slurm_multinode.sh   multi-node job for SLURM clusters
└── tests/                   unit and end-to-end tests (pytest)
```

---

## Installation

You need Python ≥ 3.10 and PyTorch ≥ 2.1 with CUDA. The code was developed and tested with
PyTorch 2.8 on an A100 80GB.

```bash
cd optimizer
# install PyTorch for your CUDA version first, see https://pytorch.org/get-started/locally/
pip install -r requirements.txt
# optional
pip install wandb          # --wandb_log=True
pip install transformers   # data/prepare.py --tokenizer=hf:<repo>
```

GPUs with bfloat16 support (A100, H100, RTX 30xx/40xx and newer) train in bfloat16. On
older GPUs (V100, T4) train.py automatically falls back to float16 with loss scaling.

---

## Data preparation

`data/prepare.py` downloads a corpus from the Hugging Face Hub, tokenizes every document
with the GPT-2 BPE tokenizer, appends the end-of-text token (id 50256) to each, and writes
the concatenated token stream to flat binary files:

```
data/<dataset>/train.bin    uint16 token ids (the nanoGPT / Sophia format)
data/<dataset>/val.bin
data/<dataset>/meta.json    tokenizer, vocab size, dtype, number of documents and tokens
```

During training, `train.py` reads these files with `np.memmap`, so the dataset never has
to fit in RAM.

### Commands

Run from the repository root:

```bash
python data/prepare.py openwebtext     # OpenWebText, all 80 shards
python data/prepare.py fineweb         # FineWeb sample-10BT, all 15 shards
python data/prepare.py c4              # C4 English, first 64 of 1024 training shards
python data/prepare.py shakespeare     # 1 MB tiny-shakespeare: a quick smoke test
```

| dataset | source on the Hub | download | train tokens | `train.bin` | validation set |
|---|---|---|---|---|---|
| `openwebtext` | `Skylion007/openwebtext` | 24 GB | ≈ 9.0 B | ≈ 18 GB | 0.05% of documents held out (≈ 4.5 M tokens) |
| `fineweb` | `HuggingFaceFW/fineweb`, `sample/10BT` | 31 GB | ≈ 10 B | ≈ 20 GB | 0.05% of documents held out (≈ 5 M tokens) |
| `c4` | `allenai/c4`, `en` (64 shards) | 21 GB | ≈ 10.9 B | ≈ 22 GB | official validation shard 0 (21.6 M tokens) |
| `c4 --num_train_shards 0` | all 1024 shards | 327 GB | ≈ 175 B | ≈ 350 GB | same |

(One C4 shard has 170.6 M tokens, one OpenWebText shard 113 M, one FineWeb-10BT shard 724 M.
These were measured while developing this repo.)

The raw shards are cached in the Hugging Face Hub cache (`$HF_HOME/hub`, by default
`~/.cache/huggingface/hub`). After preparation you can delete them with
`huggingface-cli delete-cache`. Point `HF_HOME` at a large disk if your home directory is
small. Downloads resume if they are interrupted. Setting the `HF_TOKEN` environment variable
avoids anonymous rate limits.

Each shard is tokenized by its own worker process (`--num_proc`, default: half the CPU
cores). One shard takes about a minute (C4, OpenWebText) to four minutes (FineWeb) per
process, so with enough cores preparation is limited mostly by download speed.
**The output does not depend on `--num_proc`**: documents stay in shard order, and the
train/validation split of OpenWebText and FineWeb is a seeded function of each document's
position.

### Options

```bash
python data/prepare.py fineweb --fineweb_sample 100BT            # FineWeb 100BT sample (303 GB)
python data/prepare.py fineweb --num_train_shards 5              # only the first 5 shards
python data/prepare.py c4 --num_train_shards 128                 # ~22B tokens of C4
python data/prepare.py c4 --num_val_shards 0                     # all 8 C4 validation shards
python data/prepare.py c4 --c4_variant realnewslike              # other C4 variants
python data/prepare.py openwebtext --out_dir /scratch/data/openwebtext
python data/prepare.py fineweb --tokenizer hf:meta-llama/Llama-2-7b-hf --out_dir data/fineweb_llama
```

* `--out_dir`: where the files go. If it is not `data/<dataset>`, pass the parent directory
  to training with `--data_dir` and the folder name with `--dataset`. For example,
  `--data_dir=/scratch/data --dataset=openwebtext`.
* `--tokenizer`: any [tiktoken](https://github.com/openai/tiktoken) encoding (default `gpt2`),
  or `hf:<repo>` for a Hugging Face tokenizer. Vocabularies larger than 65,536 entries (e.g.
  Llama 3) are stored as uint32 automatically. `train.py` reads the vocabulary size and dtype
  from `meta.json` and sizes the model's embedding to match, rounded up to a multiple of 64.
  The table above assumes the GPT-2 tokenizer. All baselines in this repo use GPT-2 BPE for
  both GPT-2 and LLaMA models, so the two architectures are compared on the same tokens.

### How many tokens do you need?

One optimizer step of the shipped configs processes 480 sequences × 1024 tokens
= 491,520 tokens. That is the batch size used by Sophia and nanoGPT. The configs run
100,000 steps (≈ 49 B tokens), as in Sophia. On a ≈ 9–11 B-token dataset this is 4.5–5.5 epochs.
For single-epoch training, either lower `max_iters` (e.g. `--max_iters=20000` ≈ 9.8 B tokens)
or prepare more data (`fineweb --fineweb_sample 100BT`, or more C4 shards). Training samples
random windows from `train.bin`, so it does not need the data to be shuffled.

---

## Training

All examples below assume the data has been prepared under `data/<dataset>/`.

### Single GPU

```bash
python train.py config/gpt2_small_adamw.py                          # OpenWebText (default)
python train.py config/gpt2_small_muon.py --dataset=fineweb
python train.py config/llama_small_adamw.py --dataset=c4
python train.py config/llama_medium_muon.py --dataset=c4 --wandb_log=True
```

Arguments are applied left to right: first the defaults in `train.py`, then the config file,
then `--key=value` overrides. Outputs go to `out/<dataset>/<run_name>/`, or to `--out_dir`
if you set it.

**Memory.** Peak GPU memory, measured on an A100 with `torch.compile`:

| config | parameters | `batch_size` | peak memory |
|---|---|---|---|
| `gpt2_small_*` | 124 M | 12 | ≈ 10 GiB |
| `llama_small_*` | 162 M | 12 | ≈ 10 GiB |
| `gpt2_medium_*` | 355 M | 12 | ≈ 20 GiB |
| `llama_medium_*` | 411 M | 12 | ≈ 20 GiB |
| `gpt2_large_*` | 774 M | 6 | ≈ 25 GiB |
| `llama_large_*` | 860 M | 6 | ≈ 27 GiB |

All configs fit on a 40 GB GPU. If you run out of memory, halve `batch_size` and double
`gradient_accumulation_steps`. The effective batch, and so the experiment, stays the same.

**Throughput.** One A100 80GB trains GPT-2 small at ≈ 200 k tokens/s, which is
≈ 2.4 s per step of 491,520 tokens. The full 100,000-step run therefore takes ≈ 68 hours on
one GPU, or ≈ 9 hours on 8 GPUs. For quick experiments, lower `max_iters` (the LR schedule
adapts to it) or use a smaller model:

```bash
# ~2 minutes on one GPU: tiny model on tiny Shakespeare
python data/prepare.py shakespeare
python train.py --dataset=shakespeare --n_layer=4 --n_head=4 --n_embd=256 --block_size=256 \
    --batch_size=32 --gradient_accumulation_steps=2 --max_iters=1000 --warmup_iters=100 \
    --eval_interval=250 --eval_iters=20 --optimizer=muon --run_name=tiny_muon
```

### Multiple GPUs on one node

Use `torchrun` (PyTorch DistributedDataParallel):

```bash
torchrun --standalone --nproc_per_node=8 train.py config/gpt2_small_adamw.py
torchrun --standalone --nproc_per_node=4 train.py config/llama_medium_muon.py --dataset=fineweb
```

`gradient_accumulation_steps` must be divisible by the number of GPUs. With the shipped value
40, you can use 1, 2, 4, 5, 8, 10, 20 or 40 GPUs (80 for the large configs). Each GPU runs
`gradient_accumulation_steps / num_gpus` micro-batches per step. Gradients are averaged over
all GPUs once per step, so **1 GPU and 8 GPUs train the same model on the same data**.

To use only some of the GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ...`

To train both baselines back to back:

```bash
bash scripts/run_baselines.sh gpt2_small openwebtext 8      # <model> <dataset> <num_gpus> [overrides]
bash scripts/run_baselines.sh llama_small c4 1 --max_iters=20000
```

### Multiple nodes

Every node needs the code and the prepared data at the same path, for example on a shared
file system. Run the same command on each node, e.g. for 2 nodes × 8 GPUs:

```bash
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=run1 --rdzv_backend=c10d \
    --rdzv_endpoint=<ip-of-node-0>:29500 train.py config/gpt2_medium_adamw.py
```

With InfiniBand this needs no further setup. Without it, you may need to tell NCCL which
network interface to use (`NCCL_SOCKET_IFNAME=eth0`). Only global rank 0 writes logs and
checkpoints. Checkpoints go to `out_dir`, which should be on the shared file system.

### SLURM

```bash
sbatch scripts/slurm_multinode.sh config/gpt2_medium_adamw.py --dataset=fineweb
sbatch --nodes=4 scripts/slurm_multinode.sh config/llama_large_muon.py --dataset=c4
```

Edit the `#SBATCH` lines (partition, account, time, GPUs per node) for your cluster.

### Resuming, time limits and evaluation only

A checkpoint (`out_dir/ckpt.pt`: model, optimizer state, loss-scaler state, iteration, config)
is written at every evaluation (`eval_interval`, default 1000 steps). The write is atomic, so a
crash never leaves a half-written checkpoint. To continue a run:

```bash
python train.py config/gpt2_small_adamw.py --init_from=resume
```

The architecture always comes from the checkpoint. The rest of the configuration comes from
the command line as usual, so give the same config file. `log.jsonl` is truncated to the
checkpoint's iteration and then continued.

For clusters with a job time limit, `stop_iter` ends a run early **without** shortening the LR
schedule (that is set by `max_iters`). For example, a 100k-step run as five jobs:

```bash
python train.py config/gpt2_small_adamw.py --stop_iter=20000
python train.py config/gpt2_small_adamw.py --stop_iter=40000 --init_from=resume
...
python train.py config/gpt2_small_adamw.py --init_from=resume          # runs to max_iters
```

The result is identical to an uninterrupted run, because the data at each step does not
depend on when the run was resumed.

To evaluate a checkpoint without training it:

```bash
python train.py config/gpt2_small_adamw.py --init_from=resume --eval_only=True --eval_iters=1000
```

---

## Models and configs

| | GPT-2 (`model_type='gpt2'`) | LLaMA (`model_type='llama'`) |
|---|---|---|
| positions | learned absolute embeddings | rotary embeddings (RoPE, θ = 10,000) |
| normalization | pre-LayerNorm (no bias by default) | pre-RMSNorm |
| MLP | 4·d, GELU | SwiGLU, hidden 8/3·d rounded up to a multiple of 256 |
| biases | off by default (`bias=True` for original GPT-2) | none |
| embeddings | input and output tied | untied (`tie_weights=True` to tie) |
| init | N(0, 0.02); residual projections N(0, 0.02/√(2·n_layer)) | same |

Attention uses PyTorch's `scaled_dot_product_attention` (FlashAttention kernels on recent GPUs).

| config | layers | heads | d_model | parameters | AdamW peak LR |
|---|---|---|---|---|---|
| `gpt2_small_{adamw,muon}` | 12 | 12 | 768 | 124 M | 6e-4 |
| `gpt2_medium_{adamw,muon}` | 24 | 16 | 1024 | 355 M | 3e-4 |
| `gpt2_large_{adamw,muon}` | 36 | 20 | 1280 | 774 M | 2e-4 |
| `llama_small_{adamw,muon}` | 12 | 12 | 768 | 162 M (124 M tied) | 6e-4 |
| `llama_medium_{adamw,muon}` | 24 | 16 | 1024 | 411 M (360 M tied) | 3e-4 |
| `llama_large_{adamw,muon}` | 36 | 20 | 1280 | 860 M (796 M tied) | 2e-4 |

Each LLaMA model has the same depth and width as its GPT-2 counterpart. It has more
parameters because its output head is untied and SwiGLU has three matrices. Run with
`--tie_weights=True` to match the GPT-2 parameter count more closely.

The shared settings of every config are: context length 1024; 480 sequences per step
(`batch_size × gradient_accumulation_steps`); 100,000 steps; 2,000 linear warm-up steps;
cosine decay to 10% of the peak LR; gradient clipping at 1.0; β = (0.9, 0.95); dropout 0.

The AdamW learning rates are the ones of the Sophia and nanoGPT GPT-2 baselines. They are
reasonable starting points, **not tuned optima for every model/dataset pair**. For a paper,
sweep the learning rate of every optimizer (see [Running a fair comparison](#running-a-fair-comparison)).

---

## Baseline optimizers

All optimizers divide the parameters the same way (`optimizers/registry.py: split_params`):

| group | contains |
|---|---|
| `hidden_matrices` | ≥2-D weights inside the Transformer blocks (attention and MLP projections) |
| `embeddings` | token embedding (and GPT-2's position embedding; GPT-2's tied output head) |
| `head` | the output projection, when it is not tied (LLaMA) |
| `gains_biases` | all 1-D parameters: norm gains, biases |

### AdamW (`--optimizer=adamw`)

`torch.optim.AdamW` (the fused CUDA kernel), with decoupled weight decay `weight_decay` on
every ≥2-D parameter and no weight decay on gains and biases. Hyperparameters:
`learning_rate`, `beta1`, `beta2`, `eps`, `weight_decay`.

### Muon (`--optimizer=muon`)

[Muon](https://kellerjordan.github.io/posts/muon/) (Jordan et al., 2024), following the
reference implementation ([KellerJordan/Muon](https://github.com/KellerJordan/Muon)). For each
hidden weight matrix W with gradient G:

```
M ← μ·M + (1−μ)·G                         momentum (μ = muon_momentum = 0.95)
U ← (1−μ)·G + μ·M                         Nesterov (muon_nesterov = True)
O ← NewtonSchulz5(U)                      ≈ U's polar factor: singular values ≈ 1
W ← (1 − η·λ)·W − η·√max(1, fan_out/fan_in)·O
```

η is `muon_lr` (0.02) and λ is `muon_weight_decay` (0.01). The Newton–Schulz iteration runs
`muon_ns_steps` = 5 times, in bfloat16 on GPU.

Muon is only meant for hidden matrices. Embeddings, the output head, gains and biases are
optimized by an **auxiliary AdamW** inside the same optimizer object. It uses the same
hyperparameters as the AdamW baseline (`learning_rate`, `beta1`, `beta2`, `eps`,
`weight_decay` on ≥2-D parameters only). This auxiliary AdamW is tested to be numerically
identical to `torch.optim.AdamW`. The LR schedule scales the Muon and AdamW learning rates by
the same factor.

Notes:

* GPT-2 computes Q, K and V with one fused `c_attn` matrix (768 × 2304). Muon orthogonalizes
  it as one matrix. LLaMA has separate `wq`, `wk`, `wv` matrices.
* Under DDP every GPU holds the same averaged gradient, so every GPU runs the same Muon
  update. That is redundant work but exact. With the shipped configs Muon makes a step
  ≈ 2% slower than AdamW (measured on an A100: 2.47 vs 2.43 s for GPT-2 small, 2.65 vs
  2.61 s for LLaMA small). The overhead is a fixed cost per step, so it grows in relative
  terms when you use fewer micro-batches per step.
* The reference recommends weight decay 0 for Muon on short runs. Other work (e.g.
  Moonlight) uses decoupled weight decay for long runs. The default here is 0.01; it is a
  hyperparameter worth sweeping.

### SGD (`--optimizer=sgd`)

SGD with Nesterov momentum (`beta1`) and decoupled weight decay. It lives in
`optimizers/template.py` and serves mainly as a template for new optimizers. It is not a
competitive baseline for Transformers.

---

## Adding your own optimizer

1. **Copy the template:** `cp optimizers/template.py optimizers/myopt.py`.
2. **Implement the optimizer** as a `torch.optim.Optimizer` subclass (the `step` method).
   Keep all state in `self.state[p]` so that checkpoints save and restore it.
3. **Write a builder and register it under a name:**

   ```python
   from .registry import register, split_params

   @register('myopt')
   def build_myopt(model, cfg, device_type):
       groups = split_params(model)          # hidden_matrices / embeddings / head / gains_biases
       return MyOpt([
           dict(params=groups['hidden_matrices'] + groups['embeddings'] + groups['head'],
                weight_decay=cfg.weight_decay),
           dict(params=groups['gains_biases'], weight_decay=0.0),
       ], lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), rho=cfg.myopt_rho)
   ```

   Every param group must have an `lr`. train.py stores it as that group's **peak** LR and
   multiplies it by the warm-up/decay schedule before every step. Groups may use different
   peak LRs, as Muon does.
4. **Import the module** in `optimizers/__init__.py`: `from . import adamw, muon, template, myopt`.
5. **Add any new hyperparameters** to `defaults` in `train.py` (e.g. `myopt_rho=0.04`). Only
   keys listed there are accepted, and their type is checked.
6. **Add config files** next to the baselines, e.g. copy `config/gpt2_small_adamw.py` to
   `config/gpt2_small_myopt.py`, then set `optimizer = 'myopt'` and `run_name = 'gpt2_small_myopt'`.
   `tests/test_train.py` requires every config file's `run_name` to match its file name.
7. **Test it:** add its name to the `parametrize` lists in `tests/test_optimizers.py`.
   These tests check that every parameter is optimized exactly once, that gains and biases get
   no weight decay, that the loss decreases, and that a checkpoint round trip resumes exactly.
   Then do a smoke run:
   `python train.py --dataset=shakespeare --optimizer=myopt --n_layer=4 --n_head=4 --n_embd=256 --block_size=256 --max_iters=200 --compile=False`.

What train.py guarantees when it calls `optimizer.step()`:

* the gradients are averaged over all GPUs and over all micro-batches of the step;
* they have been clipped to global norm `grad_clip` (0 disables clipping), and unscaled
  when training in float16;
* the model is the plain `nn.Module`: not compiled, not DDP-wrapped. `id(p)` and the
  parameter names (`model.named_parameters()`) are the normal ones.

If your optimizer needs extra forward/backward passes (e.g. Sophia's Hessian estimates every
k steps, or SAM), add them to the training loop in `train.py`, next to the
`scaler.step(optimizer)` call. Remember that inside DDP each extra backward pass synchronizes
gradients unless it runs under `model.no_sync()`.

---

## Logging, evaluation and plotting

`out_dir` contains:

* `config.json`: the full resolved configuration of the run;
* `log.jsonl`: one JSON record per line:
  * every `log_interval` steps: `{"iter", "train_loss", "grad_norm", "lr_mult", "train_time"}`.
    `train_loss` is the mean over every micro-batch on every GPU since the previous record.
    `grad_norm` is the global gradient norm before clipping, at the logged step.
  * every `eval_interval` steps: `{"iter", "val_loss", "eval_train_loss", "train_time"}`.
    These are the mean loss over `eval_iters` batches of `batch_size` sequences. The batches
    are fixed, so they are identical at every evaluation and in every run with the same seed.
* `ckpt.pt`: the latest checkpoint.

`train_time` is cumulative training wall-clock time in seconds. Evaluation and checkpointing
are excluded; the one-off `torch.compile` time (typically under a minute) is included.

With `--wandb_log=True` the same records also go to Weights & Biases (project `wandb_project`,
run name `run_name`).

Plot runs against each other:

```bash
python plot.py out/openwebtext/gpt2_small_adamw out/openwebtext/gpt2_small_muon --out val.png
python plot.py out/openwebtext/gpt2_small_* --x tokens --out val_tokens.png
python plot.py out/openwebtext/gpt2_small_* --x time --metric train_loss --smooth 20 --out train.png
```

The validation loss is in nats per GPT-2 token, and perplexity = exp(loss). Unlike in
Sophia/nanoGPT, `eval_iters` × `batch_size` sequences are spread over all GPUs. The loss
therefore always averages the same 200 × 12 = 2,400 sequences (2.5 M tokens), whatever the GPU
count. For final numbers in a paper, re-evaluate the checkpoint on more data:
`--eval_only=True --init_from=resume --eval_iters=2000`.

---

## Running a fair comparison

Suggestions for the experiments of an optimizer paper:

1. **Tune every method.** Sweep the peak learning rate of each optimizer, including the
   baselines, on a short run (e.g. 10–20% of the steps, or the small model), and report the
   best. For Muon, sweep `muon_lr` (e.g. 0.005–0.05) and `muon_weight_decay`. A weaker-tuned
   baseline is the most common criticism of optimizer papers.
2. **Keep everything else fixed.** Model, data order, initialization (`seed`), batch size,
   schedule, warm-up and clipping should be the same for all methods. The configs differ
   only in the optimizer section.
3. **Compare at equal tokens and at equal wall-clock time.** Use `plot.py --x tokens` and
   `--x time`. Per-step cost differs between optimizers (Muon costs ≈ 2% extra per step
   with the shipped configs).
4. **Use several seeds.** Pass `--seed=1/2/3 --run_name=<name>_s1/...`. The seed changes the
   initialization and the training data order together. Report the mean and spread.
5. **Report the setup.** Give the token budget (`max_iters × 491,520`), the number of epochs
   over the dataset, the hardware, and the final validation loss from a larger evaluation (see
   above).
6. **Check the scaling.** Show that a gain holds at small, medium and large sizes and on at
   least two datasets.

---

## Tests

```bash
pip install pytest
python -m pytest tests -q        # ≈ 3 minutes on CPU; no GPU or data download needed
```

The tests cover:

* **Models:** output shapes; initial loss ≈ ln(vocab); causality; the GPT-2-small parameter
  count (124,439,808 with biases); weight tying; the LLaMA hidden size; RoPE against the
  complex-number reference formulation and its relative-position property.
* **Optimizers:**
  * every parameter is assigned exactly once, and gains and biases get no weight decay;
  * Newton–Schulz orthogonalizes its input;
  * Muon's first step and momentum recursion match the definition;
  * Muon's auxiliary AdamW equals `torch.optim.AdamW`, and the template SGD equals `torch.optim.SGD`;
  * all optimizers decrease the loss and resume exactly from a state dict.
* **Training end to end:**
  * train and resume (the log has no duplicates), and eval-only;
  * runs are reproducible with the same seed;
  * **2-process DDP gives the same losses as 1 process**;
  * **stopping and resuming gives the same losses as an uninterrupted run**;
  * the LR schedules;
  * all shipped config files are valid.
* **Configuration:** type checking, rejection of unknown keys and of invalid values.

Beyond the tests, the data pipeline was checked by re-tokenizing raw shards independently and
comparing them token by token with `train.bin`/`val.bin`. Every configuration was also trained
on an A100, in bfloat16 and in float16, single-process and via `torchrun`.

---

## Configuration reference

All keys and their defaults are in `defaults` at the top of `train.py`:

| key | default | meaning |
|---|---|---|
| `out_dir` | `''` | output directory; `''` means `out/<dataset>/<run_name>` |
| `run_name` | `gpt2_small_adamw` | name of the run (also the wandb name) |
| `init_from` | `scratch` | `scratch` or `resume` |
| `eval_interval` / `eval_iters` | 1000 / 200 | evaluate and checkpoint every N steps, on M batches |
| `log_interval` | 10 | steps between training-loss records |
| `eval_only` | False | evaluate once and exit |
| `always_save_checkpoint` | True | False: only save when the val loss improves |
| `wandb_log` / `wandb_project` | False / `optimizer` | Weights & Biases logging |
| `data_dir` / `dataset` | `data` / `openwebtext` | tokens are read from `<data_dir>/<dataset>/` |
| `batch_size` | 12 | sequences per micro-batch per GPU |
| `block_size` | 1024 | context length |
| `gradient_accumulation_steps` | 40 | micro-batches per step, **summed over all GPUs** |
| `model_type` | `gpt2` | `gpt2` or `llama` |
| `n_layer` / `n_head` / `n_embd` | 12 / 12 / 768 | model size |
| `dropout` | 0.0 | dropout probability |
| `bias` | False | GPT-2 only: biases in Linear/LayerNorm |
| `tie_weights` | False | LLaMA only: tie embedding and output head |
| `vocab_size` | None | None: tokenizer vocab from `meta.json`, rounded up to a multiple of 64 |
| `optimizer` | `adamw` | `adamw`, `muon`, `sgd` or your own |
| `max_iters` | 100000 | number of optimizer steps (and length of the LR schedule) |
| `stop_iter` | -1 | stop early at this step (for time-limited jobs) |
| `learning_rate` | 6e-4 | peak LR of AdamW (Muon: of the auxiliary AdamW) |
| `weight_decay` | 0.1 | decoupled weight decay (≥2-D parameters) |
| `beta1` / `beta2` / `eps` | 0.9 / 0.95 / 1e-8 | Adam hyperparameters |
| `grad_clip` | 1.0 | global gradient-norm clipping; 0 disables it |
| `muon_lr` | 0.02 | peak LR of Muon (hidden matrices) |
| `muon_momentum` / `muon_nesterov` | 0.95 / True | Muon momentum |
| `muon_ns_steps` | 5 | Newton–Schulz iterations |
| `muon_weight_decay` | 0.01 | Muon decoupled weight decay |
| `lr_schedule` | `cosine` | `cosine`, `linear` or `constant` (after warm-up) |
| `warmup_iters` | 2000 | linear warm-up steps |
| `min_lr_ratio` | 0.1 | final LR as a fraction of the peak |
| `device` | `cuda` | `cuda`, `cuda:1`, `cpu`, ... |
| `dtype` | `bfloat16` | `bfloat16`, `float16` (with loss scaling) or `float32` |
| `compile` | True | `torch.compile` the model |
| `seed` | 1337 | initialization and data order |

---

## Acknowledgements

* [nanoGPT](https://github.com/karpathy/nanoGPT) (A. Karpathy): GPT-2 model, data format and training loop.
* [Sophia](https://github.com/Liuhong99/Sophia) (Liu et al., 2023, *Sophia: A Scalable Stochastic
  Second-order Optimizer for Language Model Pre-training*): repository structure, GPT-2
  baselines and hyperparameters.
* [Muon](https://github.com/KellerJordan/Muon) (K. Jordan et al., 2024): the Muon optimizer.
* [LLaMA](https://github.com/meta-llama/llama) (Touvron et al., 2023): the LLaMA architecture.
* Datasets: [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext),
  [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (Penedo et al., 2024),
  [C4](https://huggingface.co/datasets/allenai/c4) (Raffel et al., 2020).
