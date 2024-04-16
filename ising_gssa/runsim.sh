#!/bin/bash

# SBATCH --job-name=k01
# SBATCH -N 1
# SBATCH -n 1

# SBATCH --mem-per-cpu=32G
# SBATCH --export=ALL

echo $SLURMD_NODENAME
#module load gcc/6.2.0
# Initialize Modules

source /etc/profile

# Load Anaconda Module
module load anaconda/2022a

python compile.py  build_ext --inplace
