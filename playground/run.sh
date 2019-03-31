#!/bin/bash -l

#SBATCH -J hello_SLURM
#SBATCH -o output.txt
#SBATCH -e errors.txt
#SBATCH -t 01:20:00
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --mem=4096

module purge
module load python-env/intelpython3.6-2018.3 gcc/5.4.0 cuda/9.0 cudnn/7.1-cuda9

python3 word2vec.py
