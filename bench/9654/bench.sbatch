#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p vip
#SBATCH --mem=0
#SBATCH --exclusive

python3 -u ../src/bench_linear.py ../MOLPIPx-set

