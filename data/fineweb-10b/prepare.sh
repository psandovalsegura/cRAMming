#!/bin/bash
#SBATCH --account=nexus
#SBATCH --job-name=prepare-fineweb-10b-dataset
#SBATCH --time=10:00:00
#SBATCH --partition=tron
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --output=%j-%x.out
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


# Details:
# total number of tokens in train files: 11,684,959,109 across 117 shards
# total number of tokens in val files: 100,000,000 across 1 shards


# Use the Llama tokenizer on FineWeb-10b
export HF_HOME='/fs/nexus-scratch/psando/fineweb-10b'
python data/fineweb-10b/prepare.py
