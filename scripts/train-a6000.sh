#!/bin/bash
#SBATCH --account=nexus
#SBATCH --job-name=owt-llama-a6000
#SBATCH --time=2-0:00:00
#SBATCH --partition=tron
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=slurm-%j-%x.out
#SBATCH --signal=B:USR1@120
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu

#--SBATCH --array=0-1
#--SBATCH --dependency=afterok:
#--SBATCH --mail-type=end          
#--SBATCH --mail-type=fail         
#--SBATCH --mail-user=psando@umd.edu

#--SBATCH --output /dev/null
#--SBATCH --output=slurm-%j-%x.out

# For gradient accumulation, it is necessary to turn off 
# gradient offloading to CPU
export TOKENIZERS_PARALLELISM=true
exec python train.py config/train_llama-2-7b_owt.py \
                --out_dir="/fs/nexus-scratch/psando/nanoGPT-experiments/out-cRAMming-a6000" \
                --gradient_accumulation_steps=40 \
                --cramming_offload_gradients_cpu=False \
                --eval_interval=25 \
                --log_interval=25 