"""
Plot loss curves of one or more runs from their <out_dir>/log.jsonl files.

    python plot.py out/openwebtext/gpt2_small_adamw out/openwebtext/gpt2_small_muon
    python plot.py out/openwebtext/* --x time --metric train_loss --out curves.pdf

--metric  val_loss (default), eval_train_loss, or train_loss (noisy single-batch loss,
          smoothed with a running mean over --smooth log records)
--x       iter (default), time (training wall-clock hours) or tokens (billions)
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_run(out_dir, metric):
    with open(os.path.join(out_dir, 'log.jsonl')) as f:
        records = [r for r in map(json.loads, f) if metric in r]
    with open(os.path.join(out_dir, 'config.json')) as f:
        config = json.load(f)
    return records, config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='run output directories')
    parser.add_argument('--metric', default='val_loss', choices=['val_loss', 'eval_train_loss', 'train_loss'])
    parser.add_argument('--x', default='iter', choices=['iter', 'time', 'tokens'])
    parser.add_argument('--smooth', type=int, default=50, help='running-mean window for train_loss')
    parser.add_argument('--ymax', type=float, default=None)
    parser.add_argument('--out', default='loss.png')
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for out_dir in args.runs:
        records, config = load_run(out_dir, args.metric)
        if not records:
            print(f"skipping {out_dir}: no '{args.metric}' records")
            continue
        iters = np.array([r['iter'] for r in records])
        values = np.array([r[args.metric] for r in records])
        if args.x == 'iter':
            x = iters
        elif args.x == 'time':
            x = np.array([r['train_time'] for r in records]) / 3600
        else:
            tokens_per_iter = config['gradient_accumulation_steps'] * config['batch_size'] * config['block_size']
            x = iters * tokens_per_iter / 1e9
        if args.metric == 'train_loss' and args.smooth > 1 and len(values) >= args.smooth:
            kernel = np.ones(args.smooth) / args.smooth
            values = np.convolve(values, kernel, mode='valid')
            x = x[args.smooth - 1:]
        ax.plot(x, values, label=os.path.basename(os.path.normpath(out_dir)))
        print(f"{out_dir}: final {args.metric} {values[-1]:.4f} at iter {iters[-1]}, min {values.min():.4f}")

    ax.set_xlabel({'iter': 'optimizer steps', 'time': 'training time (hours)', 'tokens': 'tokens (billions)'}[args.x])
    ax.set_ylabel(args.metric.replace('_', ' '))
    if args.ymax is not None:
        ax.set_ylim(top=args.ymax)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
