#!/bin/bash
# Disable OpenMPI ROCm accelerator to prevent segfault on exit
export OMPI_MCA_accelerator=^rocm

source venv/bin/activate
mlx_lm.benchmark --model mlx-community/Qwen3-0.6B-bf16 -p 4096 -g 128
