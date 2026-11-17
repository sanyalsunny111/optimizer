# LLaMA small (162M parameters; 124M with tie_weights=True) trained with Muon (hidden weight matrices) + AdamW (everything else).
#
#   python train.py config/llama_small_muon.py
#   torchrun --standalone --nproc_per_node=8 train.py config/llama_small_muon.py
#
# Batch: 12 sequences x 1024 tokens x 40 accumulation steps = 491,520 tokens
# per optimizer step, independent of the number of GPUs.

run_name = 'llama_small_muon'

# model
model_type = 'llama'
n_layer = 12
n_head = 12
n_embd = 768

# data
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 40

# optimization
optimizer = 'muon'
max_iters = 100000
warmup_iters = 2000
muon_lr = 0.02  # hidden weight matrices
muon_momentum = 0.95
muon_weight_decay = 0.01
# auxiliary AdamW for embeddings, output head, gains and biases:
# same settings as config/llama_small_adamw.py
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
