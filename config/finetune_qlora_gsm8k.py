# Train a Llama-2-7B down to a loss of ~ on 1 GPU of 
# -----------------------------------------------------------------------------
# model
model_name = 'meta-llama/Llama-2-7b-hf'            # huggingface model name
init_from = 'pretrained'                           # 'resume', 'scratch', or 'pretrained'
cache_dir = '/fs/nexus-scratch/psando/huggingface' # where to store huggingface weights
qlora = True

# wandb logging
wandb_log = True
wandb_project = 'finetune-gsm8k'
wandb_run_name = 'qlora'

batch_size = 8
block_size = 512
gradient_accumulation_steps = 1

# lora
lora_rank = 64
lora_alpha = 128
lora_dropout = 0.05

# hyperparameters
learning_rate = 1e-4
min_lr = 1e-8
warmup_iters = 20
epochs = 2

# eval stuff
eval_interval = 20    # number of train steps after which to log train/val loss to wandb
eval_iters = 100      # number of batches to estimate train/val loss
log_interval = 10     # number of train steps after which to log current train loss to console

# weight decay
weight_decay = 1e-1
# -----------------------------------------------------------------------------