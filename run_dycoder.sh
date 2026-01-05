#!/bin/bash

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs available: $num_gpus"

# GSM GPT-2 training
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/gsm_dycoder_gpt2.yaml

# GSM GPT-2 evaluation
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/gsm_dycoder_gpt2_eval.yaml

# MATH qwen 2.5 0.5B training
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/math_dycoder_qwen2.5-0.5b.yaml

# MATH smollm 2 135M training
torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/math_dycoder_smollm2_135m.yaml

# debugging
# python -m debugpy --listen 5678 --wait-for-client run_dycoder.py configs/dycoder/gsm_dycoder.yaml