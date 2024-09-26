# Train a Llama-2-7B down to a loss of ~ on 1 GPU of 
# -----------------------------------------------------------------------------
# model
model_name = 'meta-llama/Llama-2-7b-hf'            # huggingface model name
init_from = 'scratch'                              # 'resume', 'scratch', or 'pretrained'
cache_dir = '/fs/nexus-scratch/psando/huggingface' # where to store huggingface weights

# wandb logging
wandb_log = True
wandb_project = 'cRAMming'
wandb_run_name = f"{model_name.split('/')[-1]}-{init_from}" # 'run' + str(time.time())

# data
data_dir = '/fs/nexus-scratch/psando/owt/llama-owt/'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
# 12 batch size * 1024 block size * 5 gradaccum * 1 GPU = 61,440
batch_size = 1
block_size = 4096

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000  # number of train steps after which to log train/val loss to wandb
eval_iters = 200      # number of batches to estimate train/val loss
log_interval = 10     # number of train steps after which to log current train loss to console

# weight decay
weight_decay = 1e-1
# -----------------------------------------------------------------------------