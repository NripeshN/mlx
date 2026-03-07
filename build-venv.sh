#!/bin/bash -e

uv venv --clear --seed --python 3.13 venv
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.2
uv pip install mlx-lm mlx-vlm
