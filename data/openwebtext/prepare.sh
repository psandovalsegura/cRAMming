#!/bin/bash
#SBATCH --account=nexus
#SBATCH --job-name=prepare-owt-dataset
#SBATCH --time=0-3:00:00
#SBATCH --partition=tron
#SBATCH --qos=medium
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
# total number of tokens in train: 10,252,654,099 across 1 shard
# total number of tokens in val: 5,040,479 across 1 shard
python data/openwebtext/prepare.py
