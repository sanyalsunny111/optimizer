# GPT-2 large (774M parameters) trained with Muon (hidden weight matrices) + AdamW (everything else).
#
#   python train.py config/gpt2_large_muon.py
#   torchrun --standalone --nproc_per_node=8 train.py config/gpt2_large_muon.py
#
# Batch: 6 sequences x 1024 tokens x 80 accumulation steps = 491,520 tokens
# per optimizer step, independent of the number of GPUs.

run_name = 'gpt2_large_muon'

# model
model_type = 'gpt2'
n_layer = 36
n_head = 20
n_embd = 1280

# data
batch_size = 6
block_size = 1024
gradient_accumulation_steps = 80

# optimization
optimizer = 'muon'
max_iters = 100000
warmup_iters = 2000
muon_lr = 0.02  # hidden weight matrices
muon_momentum = 0.95
muon_weight_decay = 0.01
# auxiliary AdamW for embeddings, output head, gains and biases:
# same settings as config/gpt2_large_adamw.py
learning_rate = 2e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
