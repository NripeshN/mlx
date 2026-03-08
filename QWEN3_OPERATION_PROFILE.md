# Qwen3-0.6B-BF16 Operation Profile

## Model Information

| Property | Value |
|----------|-------|
| Model | mlx-community/qwen3-0.6b-bf16 |
| Total Parameters | 596.05M |
| Hidden Size | 1024 |
| Intermediate Size | 3072 |
| Number of Layers | 28 |
| Query Attention Heads | 16 |
| Key/Value Attention Heads | 8 |
| Head Dimension | 64 |
| Vocabulary Size | 151,936 |
| Precision | bfloat16 |

## Device Configuration

| Property | Value |
|----------|-------|
| Default Device | GPU (Device(gpu, 0)) |
| GPU Available | Yes |
| CPU Available | Yes |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Model Load Time | ~1270 ms |
| Average Inference Time (per token) | ~1250 ms |
| Throughput | ~0.80 tokens/sec |
| Total GFLOPs (seq_len=10) | 10.17 GFLOPs |
| Effective Compute | ~8.00 GFLOPs/sec |

## Complete Operation List

### Phase 1: Input Embedding
| Op # | Operation | Input Shape | Output Shape | Device | Time |
|------|-----------|-------------|--------------|--------|------|
| 1 | gather/embed_tokens | (batch, seq_len) | (batch, seq_len, 1024) | GPU | ~1-2 ms |

### Phase 2: Transformer Layers (×28)

Each layer performs the following operations:

| Op # | Operation | Details | Device |
|------|-----------|---------|--------|
| 1 | rms_norm | input_layernorm | GPU |
| 2 | matmul | q_proj: (seq, 1024) @ (1024, 2048) | GPU |
| 3 | matmul | k_proj: (seq, 1024) @ (1024, 1024) | GPU |
| 4 | matmul | v_proj: (seq, 1024) @ (1024, 1024) | GPU |
| 5 | rms_norm | q_norm | GPU |
| 6 | rms_norm | k_norm | GPU |
| 7 | rope | apply_rotary_pos_emb | GPU |
| 8 | matmul | Q @ K.T: (16, seq, 64) @ (16, 64, seq) | GPU |
| 9 | divide | scale by 1/sqrt(64) | GPU |
| 10 | softmax | attention scores | GPU |
| 11 | matmul | attn @ V: (16, seq, seq) @ (16, seq, 64) | GPU |
| 12 | transpose | reshape attention output | GPU |
| 13 | matmul | o_proj: (seq, 1024) @ (1024, 1024) | GPU |
| 14 | add | residual connection | GPU |
| 15 | rms_norm | post_attention_layernorm | GPU |
| 16 | matmul | gate_proj: (seq, 1024) @ (1024, 3072) | GPU |
| 17 | matmul | up_proj: (seq, 1024) @ (1024, 3072) | GPU |
| 18 | sigmoid | SiLU activation (gate) | GPU |
| 19 | multiply | SiLU: gate * sigmoid(gate) | GPU |
| 20 | multiply | gate_output * up_output | GPU |
| 21 | matmul | down_proj: (seq, 3072) @ (3072, 1024) | GPU |
| 22 | add | residual connection | GPU |

**Per-layer timing estimate:** ~44-45 ms

### Phase 3: Output Layer

| Op # | Operation | Details | Device |
|------|-----------|---------|--------|
| 1 | rms_norm | final norm | GPU |
| 2 | matmul | lm_head: (seq, 1024) @ (1024, 151936) | GPU |
| 3 | take | extract last token logits | GPU |
| 4 | argmax | sample next token | GPU |

## Operation Statistics Summary

| Operation Type | Count per Token | Percentage |
|----------------|-----------------|------------|
| **MatMul** | 197 | 31.7% |
| **RMS Normalization** | 85 | 13.7% |
| **Element-wise (add, mul, div)** | 140 | 22.5% |
| **Softmax** | 28 | 4.5% |
| **Sigmoid** | 28 | 4.5% |
| **Transpose/Reshape** | 28 | 4.5% |
| **RoPE** | 28 | 4.5% |
| **Other (gather, take, argmax)** | 3 | 0.5% |
| **TOTAL** | **~621** | **100%** |

## Detailed Operation Timing Estimates

Based on the total inference time of ~1250 ms per token:

| Operation Category | Estimated Time | Notes |
|-------------------|----------------|-------|
| **All 28 Transformer Layers** | ~1230 ms | 98.4% of total time |
| **Embedding Lookup** | ~2-3 ms | Memory-bound |
| **LM Head + Sampling** | ~15-20 ms | Large matrix multiply |

### Per-Layer Breakdown (~44 ms per layer)

| Component | Operations | Estimated Time |
|-----------|-----------|----------------|
| **Self-Attention** | QKV proj, RoPE, Q@K.T, Softmax, Attn@V, Output proj | ~18-20 ms |
| **MLP** | Gate/Up proj, SiLU, Down proj | ~22-24 ms |
| **RMS Norms** | 3 norms per layer | ~2-3 ms |

## Key Observations

### 1. Memory-Bound Operations
- **Embedding lookup** (gather/indexing)
- **LM head projection** (1024 × 151936 matrix)
- All linear projections require loading weights from memory
- bfloat16 precision helps reduce memory bandwidth

### 2. Compute-Bound Operations
- **Q @ K.T attention scores** - quadratic in sequence length
- **Attention @ V** - matrix multiplication
- **FFN projections** (especially gate/up projections)

### 3. Optimization Features
- **Grouped Query Attention (GQA)**: 8 KV heads vs 16 Q heads
  - Reduces memory bandwidth for KV cache
  - Improves inference speed for long sequences
- **bfloat16 precision**: Memory-efficient with good numerical stability
- **RMS Normalization**: Simpler than LayerNorm (no mean subtraction)
- **SiLU activation**: Smooth, non-monotonic activation

### 4. All Operations Run on GPU
Every single operation in the model runs on:
- **Device**: GPU (Device(gpu, 0))
- **Backend**: MLX Metal (macOS) or CUDA/Vulkan (Linux)
- **Precision**: bfloat16

## CPU vs GPU Comparison

While the current configuration runs entirely on GPU, MLX supports CPU fallback:

| Device | Time per Token | Relative Speed |
|--------|----------------|----------------|
| **GPU** | ~1250 ms | 1.0x (baseline) |
| **CPU** | Not benchmarked (would be significantly slower) | ~10-100x slower estimated |

## FLOPs Analysis (Sequence Length = 10)

| Component | FLOPs per Token |
|-----------|----------------|
| Per Layer | 252,067,840 |
| All 28 Layers | 7,057,899,520 |
| LM Head | 3,111,649,280 |
| **Total** | **10,169,548,800** (10.17 GFLOPs) |

## Recommendations

1. **For faster inference**:
   - Use quantized versions (4-bit or 8-bit) to reduce memory bandwidth
   - Enable KV cache optimization for longer sequences
   - Consider using `mlx_lm` with `max_kv_size` for memory-constrained scenarios

2. **For profiling specific layers**:
   - The MLP component (gate/up/down projections) takes ~50% of layer time
   - Attention QKV projections and output projection are the next heaviest

3. **Memory optimization**:
   - Current model uses ~1.2 GB for weights (bfloat16)
   - KV cache grows with sequence length: 2 × num_layers × num_kv_heads × head_dim × seq_len × 2 bytes

## Vulkan Backend Findings (Code Review)

Based on a review of `mlx/backend/vulkan/`, the current Vulkan execution path has additional bottlenecks beyond model FLOPs.

### Confirmed Backend Bottlenecks

1. **Per-operation synchronization (major)**
   - `mlx/backend/vulkan/eval.cpp` currently synchronizes after each primitive.
   - This serializes command submission and prevents normal GPU pipeline overlap.

2. **CPU fallbacks on decode-critical ops**
   - In `mlx/backend/vulkan/primitives.cpp`, several ops still fallback to CPU paths (including common transformer primitives and fast variants).
   - For Qwen-style decode workloads, fallback on gather/norm/rope/attention-related paths can dominate latency.

3. **MatMul fast-path coverage gaps**
   - `mlx/backend/vulkan/matmul.cpp` has strict shape/layout constraints; non-matching cases fallback to CPU.
   - `AddMM` is CPU-only in Vulkan backend.
   - Extra transpose/contiguous conversions around matmul add overhead.

4. **Allocator overhead and memory placement issues**
   - `mlx/backend/vulkan/allocator.cpp` has no buffer cache/pool reuse.
   - Frequent `vkCreateBuffer`/`vkAllocateMemory`/`vkFreeMemory` in token loops is expensive.
   - On discrete GPUs, current memory-type preference can underutilize optimal device-local placement for compute.

5. **High dispatch/descriptor overhead for small ops**
   - Many elementwise/unary ops dispatch separately, each requiring descriptor updates and command recording work.
   - With hundreds of ops per token, backend overhead becomes significant.

### Concrete Optimization Plan (Priority Order)

1. **Remove per-op global sync first (highest priority)**
   - Replace conservative post-op sync with in-flight lifetime tracking and stream-level completion.
   - Keep only explicit sync points (`finalize`, user sync, dependency boundaries).

2. **Eliminate CPU fallback on decode hot path**
   - Implement/enable Vulkan paths for gather-family, RMSNorm, RoPE, and attention-critical primitives.
   - Add robust bf16 staged/fused paths where native bf16 kernels are missing.

3. **Broaden Vulkan matmul coverage**
   - Support common linear shapes used in transformer projections without CPU fallback.
   - Implement Vulkan `AddMM`.
   - Prepack/cache weight layouts to avoid repeated transpose/contiguous work.

4. **Introduce allocator pooling/suballocation**
   - Reuse buffers and device memory blocks instead of per-op allocate/free.
   - Prefer device-local memory for compute tensors; use staging only where required.

5. **Reduce kernel count via fusion and tuned kernels**
   - Fuse frequent chains (e.g., SiLU*mul, norm+scale patterns) where possible.
   - Reuse existing fused Vulkan kernels and tune matmul/softmax specializations for decode shapes.

### Expected Impact

- **Largest near-term win:** removing per-op synchronization + removing decode-path CPU fallbacks.
- **Next wins:** matmul coverage and allocator pooling.
- **Sustained gains:** kernel fusion and specialization tuning.

These backend findings explain why observed throughput can remain low even when raw FLOPs appear modest.

## Vulkan Backend Findings (Implementation Progress - Mar 2026)

### Completed Since Initial Review

1. **Deferred submission infrastructure exists, but default is currently conservative**
   - Vulkan deferred submission support remains available behind `MLX_VULKAN_DEFERRED_SUBMISSION=1`.
   - The default was moved back to immediate submission while alias- and donation-heavy graphs are hardened.
   - Command buffers are still bounded by `MLX_VULKAN_MAX_DEFERRED_OPS` when deferred mode is enabled.

2. **In-flight lifetime retention added**
   - Arrays used by recorded commands are retained until the submission fence completes.
   - Retention is wired through eval, kernel dispatch bindings, and raw copy paths.
   - Pending submissions now continue retaining arrays even after recording stops, which fixes shared-buffer lifetime bugs across multi-op graphs.

3. **Per-submission descriptor epochs added**
   - Deferred descriptor set frees are tagged by submission epoch and reclaimed only after that epoch completes.
   - This prevents descriptor reuse/free races across in-flight submissions.

4. **Hazard-driven boundaries implemented**
   - Primitive-level read/write hazard tracking was added to detect unsynchronized overlaps.
   - Donation-capable inputs are now treated as potential writes when evaluating hazards.
   - Hazard boundaries can trigger either memory barriers or submit boundaries (`MLX_VULKAN_SUBMIT_ON_HAZARD=1`).

5. **Debug instrumentation added**
   - Sync/submit/epoch tracing: `MLX_VULKAN_TRACE_SYNC=1`.
   - Descriptor lifecycle tracing: `MLX_VULKAN_TRACE_DESCRIPTORS=1`.
   - Submit failures now include a richer context summary (epoch/op counts/fence and submit retry state).

### Bugs Found During Rollout and Fixes

1. **Segfault in deferred hazard tracking**
   - Root cause: hazard range extraction dereferenced arrays before validating materialized data.
   - Fix: null-safe access via `data_shared_ptr()` and guarded buffer pointer checks.

2. **Incorrect outputs (degenerate `'!'` token loops / NaNs)**
   - Root cause: missing conservative inter-op dependency between deferred primitive dispatches.
   - Fix: restored conservative per-op memory barrier insertion for deferred command buffers.

### Current Deferred Runtime Knobs

- `MLX_VULKAN_DEFERRED_SUBMISSION` (`0`/unset default, `1` enables deferred mode)
- `MLX_VULKAN_MAX_DEFERRED_OPS` (flush threshold)
- `MLX_VULKAN_SUBMIT_ON_HAZARD` (`1` to force submit at hazard boundaries)
- `MLX_VULKAN_TRACE_SYNC` / `MLX_VULKAN_TRACE_DESCRIPTORS` (debug)

### Latest Validation Update (Mar 2026)

1. **`rsqrt_f32` Vulkan shader path is now wired and validated**
   - Added `rsqrt.comp` and registered `rsqrt_f32` in `kernels/vulkan-shaders-gen.cpp`.
   - Rebuilt and reinstalled editable MLX with Vulkan enabled; runtime now resolves `rsqrt_f32` correctly.
   - Smoke check: `mx.rsqrt([1,4,9]) -> [1,0.5,0.333333]` on Vulkan GPU.

2. **Qwen fallback profile after fix**
   - `Sqrt/Rsqrt` fallback logs no longer appear in `MLX_VULKAN_TRACE_FALLBACKS=1` runs.
   - Remaining dominant fallbacks in decode traces are currently `Softmax`, `Gather`, and `LogSumExp`.

3. **Build-system hardening for future shader additions**
   - Vulkan shader glob now uses `CONFIGURE_DEPENDS` in `mlx/backend/vulkan/CMakeLists.txt`.
   - This reduces risk of missing newly added `.comp` files when incremental builds are used.

### Latest Validation Update (Mar 2026, continued)

1. **Rank>4 Vulkan softmax fallback removed for grouped-attention shapes**
   - `try_eval_softmax_vulkan()` now flattens higher-rank inputs to a 2D row-wise view before dispatch and reshapes the output view back afterward.
   - This fixes decode-time softmax fallback for tensors shaped like `(1, 8, 2, 1, 5)`, which appeared in Qwen traces.

2. **Validation after softmax fix**
   - Added a targeted regression test in `python/tests/test_ops.py` covering rank-5 grouped-attention style softmax input.
   - `PYTHONPATH=python/tests python -m unittest python.tests.test_ops.TestOps.test_softmax_rank5_grouped_attention_shape` passes.
   - `MLX_VULKAN_TRACE_FALLBACKS=1` Qwen decode traces no longer report `Softmax` fallback entries.

3. **Current dominant decode-path fallbacks after the softmax fix**
   - `RMSNorm`: 226 fallback hits in a 1-token Qwen trace.
   - `RoPE`: 112 fallback hits.
   - `ScaledDotProductAttention` / `ScaledDotProductAttentionVJP`: 56 hits each.
   - This shifts the next highest-value work to fast primitive coverage rather than generic softmax.

### Latest Validation Update (Mar 2026, RMSNorm)

1. **Fast Vulkan RMSNorm is now wired for inference dtypes**
   - `fast::RMSNorm` now uses the existing Vulkan `rms_norm_f32` kernel instead of unconditional fallback.
   - `float16` and `bfloat16` inputs are staged through `float32` kernel buffers and converted back to the requested output dtype.
   - Higher-rank inputs are flattened to a row-wise 2D view when needed so the kernel can still execute on Vulkan.

2. **Gradient path remains functional**
   - `fast::RMSNormVJP` still uses the primitive fallback graph, but it now executes safely from the Vulkan backend instead of throwing due to the missing custom kernel.

3. **Validation after RMSNorm fix**
   - `PYTHONPATH=python/tests python -m unittest python.tests.test_fast.TestFast.test_rms_norm` passes.
   - `PYTHONPATH=python/tests python -m unittest python.tests.test_fast.TestFast.test_rms_norm_grad` passes.
   - `MLX_VULKAN_TRACE_FALLBACKS=1` Qwen decode traces no longer report `RMSNorm` fallback entries.

4. **Current dominant decode-path fallbacks after the RMSNorm fix**
   - `RoPE`: 112 fallback hits in a 1-token Qwen trace.
   - `ScaledDotProductAttention` / `ScaledDotProductAttentionVJP`: 56 hits each.
   - The next best decode-path target is now fast Vulkan `RoPE`.

### Latest Validation Update (Mar 2026, RoPE + eval/lifetime fixes)

1. **Fast Vulkan RoPE is now wired to the existing GGML-style rope shaders**
   - `mlx/backend/vulkan/rope.cpp` now dispatches `rope_norm_*` / `rope_neox_*` directly for supported layouts.
   - The Vulkan path is validated for `float32` and `float16` inputs.
   - `bfloat16` currently stays on the fallback graph path, but that path now executes safely from the Vulkan backend instead of throwing `RuntimeError: NYI`.

2. **A broader Vulkan lifetime bug was found while validating RoPE**
   - Root cause: arrays referenced by in-flight Vulkan work could be dropped too early once command recording stopped, especially around sibling outputs, donation, and shared-buffer reuse.
   - Fixes landed in `mlx/backend/vulkan/device.cpp` and `mlx/backend/vulkan/eval.cpp` to retain pending-work references correctly and to keep sibling outputs alive across evaluation.

3. **CPU fallback reads from Vulkan outputs were also racing**
   - Root cause: `concatenate_gpu()` uses a CPU fallback implementation, but it could read inputs before prior Vulkan work on the same stream completed.
   - Fix: `mlx/backend/vulkan/copy.cpp` now synchronizes the Vulkan stream before the CPU concatenate fallback reads GPU-produced arrays.

4. **Validation after the RoPE and lifetime fixes**
   - `DEVICE=gpu python -m unittest python.tests.test_fast.TestFast.test_rope python.tests.test_fast.TestFast.test_rope_batch python.tests.test_fast.TestFast.test_rope_with_freqs python.tests.test_fast.TestFast.test_rope_grad python.tests.test_fast.TestFast.test_rope_with_large_offset python.tests.test_nn.TestLayers.test_rope` passes.
   - `DEVICE=cpu python -m unittest python.tests.test_fast.TestFast.test_rope python.tests.test_fast.TestFast.test_rope_batch python.tests.test_fast.TestFast.test_rope_with_freqs python.tests.test_fast.TestFast.test_rope_grad python.tests.test_fast.TestFast.test_rope_with_large_offset python.tests.test_nn.TestLayers.test_rope` passes.

5. **RoPE status after this round**
   - Targeted RoPE correctness issues on Vulkan are resolved for the tested paths.
   - The remaining high-value decode work shifts back to native fast attention coverage and broader fallback elimination rather than RoPE correctness triage.

### Remaining High-Impact Work

1. **Eliminate decode-path CPU fallbacks** (fast ScaledDotProductAttention, remaining logsumexp/gather cases, and native bf16 RoPE coverage)
2. **Broaden Vulkan matmul coverage** (plus Vulkan AddMM)
3. **Allocator pooling/suballocation** to reduce alloc/free overhead per token
4. **Selective fusion tuning** to cut dispatch count without sacrificing correctness
