#!/bin/bash

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs available: $num_gpus"

# GSM cot qwen 2.5 0.5B training
# torchrun --nnodes 1 --nproc_per_node 1 run_coconut.py configs/coconut/gsm_cot_qwen2.5-0.5b.yaml

# MATH cot qwen 2.5 0.5B training
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_coconut.py configs/coconut/math_cot_qwen2.5-0.5b.yaml

# MATH cot qwen 2.5 0.5B evaluation
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_coconut.py configs/coconut/math_cot_qwen2.5-0.5b_eval.yaml

# MATH coconut qwen 2.5 0.5B training
torchrun --nnodes 1 --nproc_per_node $num_gpus run_coconut.py configs/coconut/math_coconut_qwen2.5-0.5b.yaml