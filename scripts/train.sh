#!/bin/bash
#SBATCH --account=nexus
#SBATCH --job-name=owt-llama
#SBATCH --time=2-0:00:00
#SBATCH --partition=tron
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=slurm-%j-%x.out
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

python train.py config/train_llama-2-7b.py \
                --out_dir="/fs/nexus-scratch/psando/nanoGPT-experiments/out-cRAMming" \
                --log_interval=1000