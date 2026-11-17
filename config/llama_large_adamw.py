# LLaMA large (860M parameters; 796M with tie_weights=True) trained with AdamW.
#
#   python train.py config/llama_large_adamw.py
#   torchrun --standalone --nproc_per_node=8 train.py config/llama_large_adamw.py
#
# Batch: 6 sequences x 1024 tokens x 80 accumulation steps = 491,520 tokens
# per optimizer step, independent of the number of GPUs.

run_name = 'llama_large_adamw'

# model
model_type = 'llama'
n_layer = 36
n_head = 20
n_embd = 1280

# data
batch_size = 6
block_size = 1024
gradient_accumulation_steps = 80

# optimization
optimizer = 'adamw'
max_iters = 100000
warmup_iters = 2000
learning_rate = 2e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
