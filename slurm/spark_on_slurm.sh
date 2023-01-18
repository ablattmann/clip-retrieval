#!/bin/bash
#SBATCH --partition=g40423
#SBATCH --job-name=clip-retrieval-sbert
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=0 # 0 means use all available memory (in MB)
#SBATCH --output=%x_%j.out
#SBATCH --comment stablediffusion
#SBATCH --exclusive

#module load cuda/11.7

srun --comment stablediffusion bash /fsx/Andreas/projects/clip-retrieval/slurm/worker_spark_on_slurm.sh
