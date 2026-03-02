#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH -J "JaxPIP GPU benchmark"
#SBATCH -p gworkq
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --exclusive

python3 -c "import jax; [print(dev.device_kind, dev.platform) for dev in jax.devices()]"

python3 -u ../src/bench_linear_batch.py ../huge-set
