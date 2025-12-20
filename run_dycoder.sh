#!/bin/bash


torchrun --nnodes 1 --nproc_per_node 1 run_dycoder.py configs/dycoder/gsm_dycoder.yaml
# python -m debugpy --listen 5678 --wait-for-client run_dycoder.py configs/dycoder/gsm_dycoder.yaml