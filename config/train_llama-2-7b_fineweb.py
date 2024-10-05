# Train a Llama-2-7B down to a loss of ~ on 1 GPU of 
# -----------------------------------------------------------------------------
# model
model_name = 'meta-llama/Llama-2-7b-hf'            # huggingface model name
init_from = 'scratch'                              # 'resume', 'scratch', or 'pretrained'
cache_dir = '/fs/nexus-scratch/psando/huggingface' # where to store huggingface weights

# wandb logging
wandb_log = True
wandb_project = 'cRAMming-fineweb'
wandb_run_name = 'llama-2-7b'

# data
data_dir = '/fs/nexus-scratch/psando/fineweb-10b/llama-fineweb-10b/'

# these make the total batch size be ~0.2M
# 1 batch size * 4096 block size * 40 gradaccum = 163,840
batch_size = 1
block_size = 4096
gradient_accumulation_steps = 1

# this makes total number of tokens be 1B if gradient accumulation is 40
max_iters = 6100
lr_decay_iters = 6100

# eval stuff
eval_interval = 1000  # number of train steps after which to log train/val loss to wandb
eval_iters = 200      # number of batches to estimate train/val loss
log_interval = 10     # number of train steps after which to log current train loss to console

# weight decay
weight_decay = 1e-1
# -----------------------------------------------------------------------------