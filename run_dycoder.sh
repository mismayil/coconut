#!/bin/bash

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs available: $num_gpus"

# GSM GPT-2 training
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/gsm_dycoder_gpt2.yaml

# GSM GPT-2 evaluation
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/gsm_dycoder_gpt2_eval.yaml

# MATH qwen 2.5 0.5B training
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/math_dycoder_qwen2.5-0.5b_with_diff.yaml

# MATH qwen 2.5 0.5B evaluation
torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/math_dycoder_qwen2.5-0.5b_with_diff_eval.yaml

# MATH smollm 2 135M training
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/math_dycoder_smollm2_135m.yaml

# MATH smollm 2 135M evaluation
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/math_dycoder_smollm2_135m_eval.yaml

# MATH smollm 2 135M training with difficulty-based latent tokens
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/math_dycoder_smollm2_135m_with_diff.yaml

# MATH smollm 2 135M evaluation with difficulty-based latent tokens
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/math_dycoder_smollm2_135m_with_diff_eval.yaml

# MATH smollm 2 1.7B training with difficulty-based latent tokens
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/math_dycoder_smollm2_1.7b_with_diff.yaml

# ProsQA gpt2 training
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/prosqa_dycoder_gpt2.yaml

# ProsQA gpt2 evaluation
# torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/prosqa_dycoder_gpt2_eval.yaml

# debugging
# python -m debugpy --listen 5678 --wait-for-client run_dycoder.py configs/dycoder/gsm_dycoder.yaml