#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH -J "JaxPIP ONNX benchmark"
#SBATCH -p vip
#SBATCH --exclusive

export LD_LIBRARY_PATH=/home/jhli/onnxruntime-linux-x64-gpu-1.24.2/lib:$LD_LIBRARY_PATH

rm bench_linear_single.x

g++ -O3 -march=native ../src/bench_linear_single.cpp -o bench_linear_single.x -I/home/jhli/onnxruntime-linux-x64-gpu-1.24.2/include -L/home/jhli/onnxruntime-linux-x64-gpu-1.24.2/lib -lonnxruntime

python3 -u ../src/bench_linear.py ../MOLPIPx-set
