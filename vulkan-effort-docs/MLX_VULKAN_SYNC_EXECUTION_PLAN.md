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
- [x] Step 8 - Make Vulkan `gpu::finalize()` submit deferred work without turning it into a full stream synchronize.
- [x] Step 9 - Route Vulkan event/fence waits through stream retirement so async finalize still signals completion callbacks correctly.
- [x] Step 10 - Revalidate Qwen3 and focused tests after reducing finalize-driven explicit synchronize submits.
- [x] Step 11 - Attribute the remaining explicit synchronize submits to concrete call sites in Vulkan/Qwen3 inference.
- [ ] Step 12 - Remove or narrow the hottest remaining synchronize callers without regressing correctness.
- [ ] Step 13 - Revalidate the reduced-sync path with the same focused suites and the Qwen3 profiler.

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
- The next bottleneck is explicit synchronize traffic driven by Vulkan `gpu::finalize()`, which still behaves like a blocking synchronize instead of a non-blocking commit.
- Vulkan `gpu::finalize()` now submits without waiting, and Vulkan event/fence waits retire the source stream on demand. That change is correct, but short Qwen3 profiling shows only a small shift from `explicit synchronize` submits to `finalize`, so more sync-site attribution is still needed.
- Sync attribution now identifies the dominant remaining reasons as BF16 copy fallbacks, concatenate fallback on axis 3, RoPE fallback, compiled fallback, and a smaller matmul fallback slice.
- BF16 copy support is now always on for stable paths: contiguous `bf16 -> f32`, contiguous `bf16 -> bf16`, and the default deferred-op threshold is reduced to `8`, which keeps Qwen3 stable on the current Radeon path while preserving the new BF16 copy coverage.
