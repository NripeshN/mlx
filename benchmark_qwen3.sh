#!/bin/bash

quant=${1-bf16}

# Disable OpenMPI ROCm accelerator to prevent segfault on exit
export OMPI_MCA_accelerator=^rocm

source venv/bin/activate
mlx_lm.benchmark --model mlx-community/Qwen3-0.6B-$quant -p 4096 -g 128
