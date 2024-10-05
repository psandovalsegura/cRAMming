#!/bin/bash
#SBATCH --account=nexus
#SBATCH --job-name=owt-llama-a4000
#SBATCH --time=2-0:00:00
#SBATCH --partition=tron
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=slurm-%j-%x.out
#SBATCH --signal=B:USR1@120
#SBATCH --mail-type=begin,end,fail
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
                --out_dir="/fs/nexus-scratch/psando/nanoGPT-experiments/out-cRAMming-a4000" \
                --gradient_accumulation_steps=1 \
                --cramming_offload_gradients_cpu=True \
                --eval_interval=1000 \
                --log_interval=1000 