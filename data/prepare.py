"""
Download a pretraining corpus and tokenize it into the flat binary files that train.py
reads with np.memmap:

    <out_dir>/train.bin   token ids (uint16; uint32 for vocabularies larger than 65536)
    <out_dir>/val.bin
    <out_dir>/meta.json   tokenizer, vocabulary size, dtype and token counts

Usage (from the repository root; default out_dir is data/<dataset>/):

    python data/prepare.py openwebtext                  # ~9B tokens
    python data/prepare.py fineweb                      # FineWeb sample-10BT, ~10B tokens
    python data/prepare.py c4                           # first 64 of 1024 C4-en shards, ~11B tokens
    python data/prepare.py c4 --num_train_shards 0      # all of C4-en, ~175B tokens
    python data/prepare.py shakespeare                  # 1 MB of text, for smoke tests

Documents are tokenized with the GPT-2 BPE tokenizer (tiktoken) by default and
separated by the end-of-text token. The format is the one used by nanoGPT and Sophia.

The shard files of a corpus are downloaded from the Hugging Face Hub into the Hub cache
($HF_HOME/hub, default ~/.cache/huggingface/hub) and tokenized in parallel, one process
per shard. The result is deterministic: it does not depend on --num_proc.
See the README for disk-space requirements.
"""

import argparse
import gzip
import json
import multiprocessing as mp
import os
import re
import shutil
import urllib.request

import numpy as np
from tqdm import tqdm

# Every corpus is a set of shard files in a Hugging Face dataset repository.
#   directory:      folder in the repo that holds the shards (may use CLI arguments)
#   train / val:    regex selecting the training / validation shards in that folder;
#                   val=None: a random fraction of the training documents is held out
#   default_shards: number of training shards used by default (0 = all)
DATASETS = {
    'openwebtext': dict(
        repo='Skylion007/openwebtext', directory='plain_text',
        train=r'train-\d+-of-\d+\.parquet$', val=None, default_shards=0),
    'fineweb': dict(
        repo='HuggingFaceFW/fineweb', directory='sample/{fineweb_sample}',
        train=r'\.parquet$', val=None, default_shards=0),
    'c4': dict(
        repo='allenai/c4', directory='{c4_variant}',
        train=r'c4-train\.\d+-of-\d+\.json\.gz$', val=r'c4-validation\.\d+-of-\d+\.json\.gz$',
        default_shards=64),
}
SHAKESPEARE_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
SPLIT_SEED = 2357  # seed of the train/val document split
DOCS_PER_BATCH = 1024


class Tokenizer:
    """
    name: a tiktoken encoding (default 'gpt2'), or 'hf:<repo>' for a Hugging Face
    tokenizer, e.g. 'hf:meta-llama/Llama-2-7b-hf' (needs `pip install transformers`).
    """

    def __init__(self, name):
        self.name = name
        if name.startswith('hf:'):
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name[3:])
            self.vocab_size = len(tok)
            self.eot = tok.eos_token_id if tok.eos_token_id is not None else tok.bos_token_id
            if self.eot is None:
                raise ValueError(f"tokenizer {name} has neither an EOS nor a BOS token "
                                 f"to separate documents with")
            self._encode_batch = lambda texts: tok(texts, add_special_tokens=False)['input_ids']
        else:
            import tiktoken
            enc = tiktoken.get_encoding(name)
            self.vocab_size = enc.n_vocab
            self.eot = enc.eot_token
            # encode_ordinary ignores special tokens that appear literally in the text
            self._encode_batch = lambda texts: [enc.encode_ordinary(t) for t in texts]
        self.dtype = np.uint16 if self.vocab_size <= 2 ** 16 else np.uint32

    def encode_docs(self, texts):
        """Tokenize documents and return one array: doc1 <eot> doc2 <eot> ..."""
        arrays = []
        for ids in self._encode_batch(texts):
            ids.append(self.eot)
            arrays.append(np.asarray(ids, dtype=self.dtype))
        return np.concatenate(arrays) if arrays else np.zeros(0, dtype=self.dtype)


def iter_text_batches(path):
    """Yield lists of up to DOCS_PER_BATCH documents (the 'text' field) of one shard file."""
    if path.endswith('.parquet'):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=DOCS_PER_BATCH, columns=['text']):
            yield batch.column(0).to_pylist()
    elif path.endswith('.json.gz'):
        texts = []
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                texts.append(json.loads(line)['text'])
                if len(texts) == DOCS_PER_BATCH:
                    yield texts
                    texts = []
        if texts:
            yield texts
    else:
        raise ValueError(f"unsupported shard format: {path}")


# --- parallel tokenization: one task per shard file ---
_worker = {}


def _init_worker(tokenizer_name):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    _worker['tokenizer'] = Tokenizer(tokenizer_name)


def _tokenize_shard(task):
    """
    Tokenize one shard into <tmp_dir>/{train,val}_<index>.bin.

    role 'train' / 'val': all documents go to that split.
    role 'split': each document goes to val with probability val_fraction, decided by
    a random stream that only depends on (SPLIT_SEED, shard index, document position).
    """
    index, path, role, val_fraction, tmp_dir = task
    tokenizer = _worker['tokenizer']
    rng = np.random.default_rng([SPLIT_SEED, index])
    stats = {'train_docs': 0, 'train_tokens': 0, 'val_docs': 0, 'val_tokens': 0}
    files = {name: open(os.path.join(tmp_dir, f'{name}_{index:05d}.bin'), 'wb') for name in ('train', 'val')}
    try:
        for texts in iter_text_batches(path):
            if role == 'split':
                to_val = rng.random(len(texts)) < val_fraction
                parts = {'train': [t for t, v in zip(texts, to_val) if not v],
                         'val': [t for t, v in zip(texts, to_val) if v]}
            else:
                parts = {role: texts}
            for name, docs in parts.items():
                if docs:
                    tokens = tokenizer.encode_docs(docs)
                    tokens.tofile(files[name])
                    stats[f'{name}_docs'] += len(docs)
                    stats[f'{name}_tokens'] += len(tokens)
    finally:
        for f in files.values():
            f.close()
    return stats


def list_shards(repo, directory, pattern):
    """Sorted [(path, size_in_bytes)] of the files in `directory` matching `pattern`."""
    from huggingface_hub import HfApi
    entries = HfApi().list_repo_tree(repo, path_in_repo=directory, repo_type='dataset')
    return sorted((e.path, getattr(e, 'size', 0) or 0) for e in entries if re.search(pattern, e.path))


def take(shards, n, what):
    if n < 0 or n > len(shards):
        raise ValueError(f"asked for {n} {what} shards but there are {len(shards)}")
    shards = shards[:n] if n > 0 else shards
    print(f"{what}: {len(shards)} shard files, {sum(s for _, s in shards) / 1e9:.2f} GB to download")
    return [p for p, _ in shards]


def prepare_hub_dataset(args, tokenizer):
    from huggingface_hub import snapshot_download

    spec = DATASETS[args.dataset]
    directory = spec['directory'].format(fineweb_sample=args.fineweb_sample, c4_variant=args.c4_variant)
    train_shards = list_shards(spec['repo'], directory, spec['train'])
    if not train_shards:
        raise RuntimeError(f"no training shards found in {spec['repo']}/{directory}")
    n_train = spec['default_shards'] if args.num_train_shards is None else args.num_train_shards
    train_files = take(train_shards, n_train, 'train')
    if spec['val'] is not None:
        val_files = take(list_shards(spec['repo'], directory, spec['val']), args.num_val_shards, 'validation')
        shards = [(path, 'train') for path in train_files] + [(path, 'val') for path in val_files]
    else:
        shards = [(path, 'split') for path in train_files]

    # download (resumable; files that are already in the Hub cache are not fetched again)
    root = snapshot_download(spec['repo'], repo_type='dataset', allow_patterns=[p for p, _ in shards],
                             max_workers=min(args.num_proc, 16))
    missing = [p for p, _ in shards if not os.path.exists(os.path.join(root, p))]
    if missing:
        raise RuntimeError(f"download incomplete, missing {len(missing)} files, e.g. {missing[0]}")

    tmp_dir = os.path.join(args.out_dir, 'tmp_shards')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    tasks = [(i, os.path.join(root, path), role, args.val_fraction, tmp_dir)
             for i, (path, role) in enumerate(shards)]
    info = {'source': f"{spec['repo']}/{directory}", 'train_shards': len(train_files),
            'train_docs': 0, 'train_tokens': 0, 'val_docs': 0, 'val_tokens': 0}
    with mp.Pool(min(args.num_proc, len(tasks)), initializer=_init_worker,
                 initargs=(tokenizer.name,)) as pool:
        with tqdm(total=len(tasks), desc='tokenizing', unit='shard') as progress:
            for stats in pool.imap_unordered(_tokenize_shard, tasks):
                for key, value in stats.items():
                    info[key] += value
                progress.set_postfix(tokens=f"{(info['train_tokens'] + info['val_tokens']) / 1e9:.2f}B")
                progress.update()

    # concatenate the per-shard token files in shard order
    for name in ('train', 'val'):
        out_path = os.path.join(args.out_dir, f'{name}.bin')
        with open(out_path + '.tmp', 'wb') as out:
            for index in tqdm(range(len(tasks)), desc=f'writing {name}.bin', unit='shard'):
                part = os.path.join(tmp_dir, f'{name}_{index:05d}.bin')
                with open(part, 'rb') as f:
                    shutil.copyfileobj(f, out, length=64 * 1024 * 1024)
                os.remove(part)
        os.replace(out_path + '.tmp', out_path)
        n_tokens = os.path.getsize(out_path) // np.dtype(tokenizer.dtype).itemsize
        assert n_tokens == info[f'{name}_tokens'], (name, n_tokens, info[f'{name}_tokens'])
    shutil.rmtree(tmp_dir)
    if info['val_tokens'] == 0:
        raise RuntimeError("the validation split is empty; increase --val_fraction")
    return info


def prepare_shakespeare(args, tokenizer):
    with urllib.request.urlopen(SHAKESPEARE_URL) as response:
        text = response.read().decode('utf-8')
    n = int(0.9 * len(text))
    info = {'source': SHAKESPEARE_URL}
    for name, part in (('train', text[:n]), ('val', text[n:])):
        tokens = tokenizer.encode_docs([part])
        tokens.tofile(os.path.join(args.out_dir, f'{name}.bin'))
        info[f'{name}_docs'] = 1
        info[f'{name}_tokens'] = len(tokens)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=[*DATASETS, 'shakespeare'])
    parser.add_argument('--out_dir', default=None,
                        help='output directory (default: data/<dataset>/ next to this script)')
    parser.add_argument('--tokenizer', default='gpt2',
                        help="tiktoken encoding name, or hf:<repo> for a Hugging Face tokenizer")
    parser.add_argument('--num_proc', type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help='worker processes for tokenizing (one shard per process)')
    parser.add_argument('--num_train_shards', type=int, default=None,
                        help='use only the first N training shard files; 0 = all '
                             '(default: all, except 64 of 1024 for c4)')
    parser.add_argument('--num_val_shards', type=int, default=1,
                        help='c4 only: number of official validation shard files; 0 = all 8')
    parser.add_argument('--val_fraction', type=float, default=0.0005,
                        help='openwebtext/fineweb: fraction of documents held out for validation')
    parser.add_argument('--fineweb_sample', default='10BT', choices=['10BT', '100BT', '350BT'])
    parser.add_argument('--c4_variant', default='en',
                        choices=['en', 'en.noblocklist', 'en.noclean', 'realnewslike'])
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        parser.error('--val_fraction must be in (0, 1)')

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = Tokenizer(args.tokenizer)

    if args.dataset == 'shakespeare':
        info = prepare_shakespeare(args, tokenizer)
    else:
        info = prepare_hub_dataset(args, tokenizer)

    meta = {
        'dataset': args.dataset,
        'tokenizer': tokenizer.name,
        'vocab_size': tokenizer.vocab_size,
        'eot_token': tokenizer.eot,
        'dtype': np.dtype(tokenizer.dtype).name,
        **info,
    }
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"wrote train.bin, val.bin and meta.json to {args.out_dir}")


if __name__ == '__main__':
    main()
