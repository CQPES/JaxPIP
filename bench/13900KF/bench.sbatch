#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p gworkq
#SBATCH --mem=0
#SBATCH --exclusive
#SABTCH --gres=gpu:nvidia_geforce_rtx_4090:1

python3 -u ../src/bench_linear.py ../MOLPIPx-set

