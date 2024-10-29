#!/bin/bash
#SBATCH --account=scavenger
#SBATCH --job-name=eval-gsm8k
#SBATCH --time=0:30:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=/dev/null

export TOKENIZERS_PARALLELISM=true

MODEL_PATH=$1
USE_PEFT=$2

# We always use the llama-2-7b base tokenizer
TOKENIZER_PATH="meta-llama/Llama-2-7b-hf"

# evaluate.py arguments: pretrained_model_name_or_path, tokenizer_name_or_path, tasks
# max batch sizes: A6000=32

# USE_PEFT will be either "True" or "False"
if [ "$USE_PEFT" = "True" ]; then
    echo "Submitting evaluate.py with PEFT ${MODEL_PATH}"
    python evaluate.py ${TOKENIZER_PATH} ${TOKENIZER_PATH} gsm8k --batch_size=32 --peft=${MODEL_PATH}
else
    echo "Submitting evaluate.py ${MODEL_PATH}"
    python evaluate.py ${MODEL_PATH} ${TOKENIZER_PATH} gsm8k --batch_size=32
fi
