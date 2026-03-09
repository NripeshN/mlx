# MLX Vulkan Barrier-First Execution Plan

This document tracks the work needed to move MLX Vulkan from a submit-on-hazard model toward a barrier-first model inspired by `tmp_/ggml-vulkan.cpp`.

## Goals

- Reduce tiny hazard-driven queue submits during LLM prefill and decode.
- Keep same-stream dependencies inside a command buffer whenever possible.
- Remove forced host synchronization from hot Vulkan inference paths.
- Introduce explicit scratch-buffer reuse rules instead of relying on generic hazard submits.

## Execution Steps

- [x] Step 1 - Write down the execution plan and use it as the source of truth for progress.
- [x] Step 2 - Add Vulkan submission/hazard accounting so prefill and decode can report barriers vs submits.
- [x] Step 3 - Rework Vulkan RoPE so it does not read back offsets/frequencies to the host or force stream synchronization in the hot path.
- [x] Step 4 - Remove explicit stream synchronization from the Vulkan flash-attention path and keep the follow-on work on the GPU timeline.
- [x] Step 5 - Add a barrier-first hazard mode in `mlx/backend/vulkan/device.cpp`, keeping submit-on-hazard as a fallback escape hatch.
- [x] Step 6 - Add explicit scratch-lane tracking for the highest-pressure temporary buffers used by matmul and attention paths.
- [x] Step 7 - Validate the new execution model with focused CPU/GPU tests and the Qwen3 Vulkan profiler.

## Validation Requirements

- Build with `./build-vulkan.sh` from the repo root using `venv`.
- Run both CPU and GPU coverage after each meaningful backend change.
- Use `mlx/backend/vulkan/profile_qwen3_vulkan.py` to compare host enqueue time, sync checkpoints, and fallback behavior before and after each step.

## Notes

- The reference control-plane design is `tmp_/ggml-vulkan.cpp`, especially its graph batching, barrier-based synchronization, and scratch-buffer reuse flags.
- The immediate target is correctness-preserving latency reduction, not a literal port of ggml's graph engine.
- The profiler now captures backend stderr in-process, attributes sync-trace activity to prefill/decode, and reports submit reasons plus hazard counts without requiring shell-side grepping.
- RoPE now feeds offsets and optional frequency buffers directly to the Vulkan shader path, removing the prior host readback + staging round-trip from inference.
- Flash attention now keeps its transpose/cast/copy follow-on work on the GPU timeline instead of forcing a stream-wide synchronize after the native dispatch.
- Barrier-first hazard handling is now the default, with `MLX_VULKAN_SUBMIT_ON_HAZARD=1` retained as an escape hatch for comparison and rollback testing.
- Matmul and flash-attention now reuse named per-stream scratch lanes for their hottest temporary buffers instead of repeatedly allocating fresh transient storage.
- Validation for this run used Vulkan rebuilds, focused CPU/GPU unit coverage, the Vulkan parity suite, C++ RoPE coverage, and repeated short Qwen3 profiler passes.
