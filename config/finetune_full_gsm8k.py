# Train a Llama-2-7B down to a loss of ~ on 1 GPU of 
# -----------------------------------------------------------------------------
# model
model_name = 'meta-llama/Llama-2-7b-hf'            # huggingface model name
init_from = 'pretrained'                           # 'resume', 'scratch', or 'pretrained'
cache_dir = '/fs/nexus-scratch/psando/huggingface' # where to store huggingface weights

# wandb logging
wandb_log = True
wandb_project = 'finetune-gsm8k'
wandb_run_name = 'full'

batch_size = 4
block_size = 1024
gradient_accumulation_steps = 1

# Num samples in train: 7473
# Num samples in test: 1319
# Train tokens per sample: Min: 81, Max: 585, Avg: 210.39288103840494
# Test tokens per sample: Min: 85, Max: 584, Avg: 214.25473843821078
learning_rate = 3e-5
min_lr = 1e-6
warmup_iters = 20
epochs = 2

# eval stuff
eval_interval = 10    # number of train steps after which to log train/val loss to wandb
eval_iters = 100      # number of batches to estimate train/val loss
log_interval = 10     # number of train steps after which to log current train loss to console

# weight decay
weight_decay = 1e-1
# -----------------------------------------------------------------------------