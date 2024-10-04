"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import sys
import time
import math
import signal
import inspect
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel, MODEL_TYPE_TO_APPLY_LIGER_FN
from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

import functools
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        apply_activation_checkpointing,
        CheckpointImpl,
    )
non_reentrant_wrapper = functools.partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    if block_class_name.startswith("nn."):
        return getattr(torch.nn, block_class_name.split(".")[-1])
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
save_checkpoints = False # if True, will save a checkpoint after each eval
# model
model_name = 'meta-llama/Llama-2-7b-hf' # huggingface model name
init_from = 'scratch' # 'resume', 'scratch', or 'pretrained'
cache_dir = 'cache' # where to store huggingface weights
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'cRAMming'
wandb_run_name = f"{model_name.split('/')[-1]}-{init_from}" # 'run' + str(time.time())
# data
data_dir = 'data'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 1 
block_size = 4096
# adamw optimizer
learning_rate = 3e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
cramming_offload_optim_cpu = True
cramming_offload_gradients_cpu = True
cramming_activation_checkpointing = True
cramming_fuse_optim_backward = False
using_preemptible = False # whether running on a preemptible instance
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

def liger_kernel_for_causal_lm(init_from, model_name, cache_dir, **kwargs):
    print(f"Initializing model {model_name} from {init_from}")
    model_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)

    # Determine the model type and apply the Liger Kernel if applicable
    # Note: _apply_liger_kernel will only pass relevant kwargs to the apply_liger_kernel_to_* function
    model_type = model_config.model_type
    _apply_liger_kernel(model_type, **kwargs)

    # Filter out kwargs that were passed to the apply_liger_* function, which will cause
    # model initialization errors otherwise
    apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]
    apply_fn_signature = inspect.signature(apply_fn)
    applicable_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in apply_fn_signature.parameters
    }
        
    if init_from == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **applicable_kwargs)
    elif init_from == "scratch":
        model = AutoModelForCausalLM.from_config(model_config, **applicable_kwargs)
    else:
        raise ValueError(f"Invalid init_from value: {init_from}")
    return model

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    if cramming_offload_optim_cpu:
        optimizer = CPUOffloadOptimizer(optim_groups, torch.optim.AdamW, offload_gradients=cramming_offload_gradients_cpu, fused=use_fused, lr=learning_rate, betas=betas)
        print("Using CPU Offload Optimizer")
    else:
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        print("Using PyTorch AdamW Optimizer")
    return optimizer

def configure_fused_optimizers(model, weight_decay, learning_rate, betas, device_type):
    print("=> Using optimizer fused into backward pass")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    optimizer_dict = dict()
    for pn, p in model.named_parameters():
        if p.requires_grad:
            if p.dim() >= 2:
                # use weight decay
                if cramming_offload_optim_cpu:
                    optimizer_dict[p] = CPUOffloadOptimizer([p], torch.optim.AdamW, offload_gradients=cramming_offload_gradients_cpu, fused=use_fused, lr=learning_rate, betas=betas, weight_decay=weight_decay)
                else:
                    optimizer_dict[p] = torch.optim.AdamW([p], lr=learning_rate, betas=betas, weight_decay=weight_decay, fused=use_fused)
            else:
                # no weight decay
                if cramming_offload_optim_cpu:
                    optimizer_dict[p] = CPUOffloadOptimizer([p], torch.optim.AdamW, offload_gradients=cramming_offload_gradients_cpu, fused=use_fused, lr=learning_rate, betas=betas, weight_decay=0.0)
                else:
                    optimizer_dict[p] = torch.optim.AdamW([p], lr=learning_rate, betas=betas, weight_decay=0.0, fused=use_fused)
    def optimizer_hook(parameter):
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad()
    for p in model.parameters():
        p.register_post_accumulate_grad_hook(optimizer_hook)
    return optimizer_dict

def save_checkpoint_on_signal(signum, frame):
    # save a checkpoint on SIGTERM or SIGUSR1 (e.g. from slurm sbatch)
    signal_name = {signal.SIGTERM: 'SIGTERM', signal.SIGUSR1: 'SIGUSR1 (BEFORE TIME LIMIT)'}[signum]
    print(f"Received {signal_name}, saving checkpoint to {ckpt_file}")
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': dict(init_from=init_from, model_name=model_name, cache_dir=cache_dir),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    ckpt_file = os.path.join(out_dir, f'signal_ckpt_iter_{iter_num}.pt')
    torch.save(checkpoint, ckpt_file)
    sys.exit(0)

signal.signal(signal.SIGTERM, save_checkpoint_on_signal)
signal.signal(signal.SIGUSR1, save_checkpoint_on_signal)

# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"Tokens per iteration: {tokens_per_iter:,}")
print(f"Tokens per wandb step: {tokens_per_iter * eval_interval:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
torch.set_default_dtype(ptdtype)

# poor man's data loader
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
if init_from == 'resume':
    # resume training from latest checkpoint
    ckpt_file = max([f for f in os.listdir(out_dir) if f.startswith('signal_ckpt_iter_')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
    ckpt_path = os.path.join(out_dir, ckpt_file)
    checkpoint = torch.load(ckpt_path)
    checkpoint_model_args, iter_num, best_val_loss = checkpoint['model_args'], checkpoint['iter_num'], checkpoint['best_val_loss']
    print(f"Resuming training from checkpoint:\n\tPath:{ckpt_path}\n\tIter:{iter_num}\n\tBest val loss:{best_val_loss}\n\tModel args:{checkpoint_model_args}")
    # create the model on GPU, so there is CPU memory available for optimizer state init
    with torch.device(device):
        model = liger_kernel_for_causal_lm(**checkpoint_model_args)
    model.load_state_dict(checkpoint['model'])
    # free model state dict GPU memory
    del checkpoint['model']
elif init_from in ['pretrained', 'scratch']:
    model = liger_kernel_for_causal_lm(init_from, model_name, cache_dir)
    # activation checkpointing
    if cramming_activation_checkpointing:
        wrap_class_str = 'LlamaDecoderLayer'
        wrap_class = get_block_class_from_model(model, wrap_class_str)
        check_fn = lambda submodule: isinstance(submodule, wrap_class)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
        print(f"Applied activation checkpointing to {wrap_class_str}")
else:
    raise ValueError(f"Unknown init_from value: {init_from}")

model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
eval_prompt = 'Llama 2 is a language model that'
eval_prompt = tokenizer(eval_prompt, return_tensors='pt').input_ids
generation_max_len = 64

# optimizer
if cramming_fuse_optim_backward:
    optimizer_dict = configure_fused_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)
else:
    optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    # free optimizer state dict cpu memory
    del checkpoint['optimizer']
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss_and_generate():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X = get_batch(split)
            with ctx:
                loss = model(input_ids=X, labels=X, use_cache=False).loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    # generate some text
    outputs = model.generate(eval_prompt.to(device), max_length=generation_max_len, 
                             do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    out['text'] = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    # add tokens_per_wandb_step to config for easy reference in wandb
    config['tokens_per_wandb_step'] = tokens_per_iter * eval_interval
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    if cramming_fuse_optim_backward:
        for suboptimizer in optimizer_dict.values():
            for param_group in suboptimizer.param_groups:
                param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        eval_out = estimate_loss_and_generate()
        print(f"step {iter_num}: train loss {eval_out['train']:.4f}, val loss {eval_out['val']:.4f}")
        print(f"generated text: {eval_out['text']}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train_loss": eval_out['train'],
                "val_loss": eval_out['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "text_sample": eval_out['text'],
            })
        if eval_out['val'] < best_val_loss:
            best_val_loss = eval_out['val']
            if iter_num > 0 and save_checkpoints:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': dict(init_from=init_from, model_name=model_name, cache_dir=cache_dir),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                ckpt_file = os.path.join(out_dir, f'ckpt_iter_{iter_num}.pt')
                print(f"saving checkpoint to {ckpt_file}")
                torch.save(checkpoint, ckpt_file)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            loss = model(input_ids=X, labels=X, use_cache=False).loss
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        loss.backward()
        X = get_batch('train')

    # step the optimizer
    if not cramming_fuse_optim_backward: # not needed for optimizer fused into backward pass
        optimizer.step()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item()
        # TODO: mfu calculation
        running_mfu = 0.0
        # if local_iter_num >= 5: # let the training loop settle a bit
        #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        #     running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
