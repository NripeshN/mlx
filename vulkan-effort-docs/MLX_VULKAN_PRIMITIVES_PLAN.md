# MLX Vulkan Primitives Implementation Plan

This plan describes how to replace the current CPU fallback path in `mlx/backend/vulkan/primitives.cpp` with real Vulkan primitive execution using the already-compiled shader kernels.

## Progress Update (current branch)

- Dispatch infrastructure in `mlx/backend/vulkan/kernels.cpp` is implemented (`dispatch_binary_op`, `dispatch_unary_op`).
- Descriptor set lifetime is handled across async submission via deferred free + reclaim hooks.
- Host push constants were updated to match binary/unary/generic shader ABI for the current elementwise slice.
- ggml-style 512/262144 tiling is wired for elementwise dispatch.
- Primitive wiring landed:
  - Binary: `Add`, `Subtract`, `Multiply`, `Divide` (with strict fallback guards).
  - Unary (strided ABI path): `Log` (natural log path), `Square`, `Sqrt` (`recip=false`) with guarded Vulkan path.
  - Unary (generic contiguous path): `Abs`, `Negative`, `Exp`, `Floor`, `Ceil`, `Round` (ties-to-even), `Sigmoid`, `Tanh` with strict contiguous/offset guards.
  - Constructor op: `Arange` (`float32`) with Vulkan dispatch on contiguous outputs and CPU fallback otherwise.
- `Sin` / `Cos` Vulkan path was attempted but left on CPU fallback for now due precision-sensitive test behavior; to be revisited.

It follows the wiring model in `vulkan-effort-docs/GGML_VULKAN_OPS.md`:

```text
Shader (.comp) -> SPIR-V (build) -> Pipeline (init/cache) -> Op dispatch -> Primitive eval_gpu
```

## Current Gaps

- `mlx/backend/vulkan/primitives.cpp` routes most primitives through CPU fallback macros.
- `mlx/backend/vulkan/kernels.cpp` has unimplemented dispatch stubs (`dispatch_binary_op`, `dispatch_unary_op`).
- Push constant structs in `mlx/backend/vulkan/kernels.h` do not fully match the active shader layouts used by `generic_binary_head.glsl`, `generic_unary_head.glsl`, and op-specific shaders.
- Workgroup/grid dispatch logic is still generic and does not match ggml kernel indexing assumptions for several shaders.

## Implementation Strategy

### 1) Build dispatch infrastructure first (hard dependency)

1. Implement a common Vulkan compute dispatch path in `mlx/backend/vulkan/kernels.cpp`:
   - pipeline lookup/creation by shader variant name
   - descriptor set allocation and storage-buffer binding
   - push constant upload
   - `vkCmdBindPipeline`, `vkCmdBindDescriptorSets`, `vkCmdDispatch`
2. Ensure descriptor lifetime is safe across async stream submission.
3. Keep command recording integrated with existing stream helpers:
   - `begin_command_recording()` / `end_command_recording()` in `mlx/backend/vulkan/device.cpp`

### 2) Align host push constants with shader ABI

1. Add/adjust host-side push constant structs to match shader layouts exactly:
   - binary kernels (`generic_binary_head.glsl`)
   - unary kernels (`generic_unary_head.glsl`)
   - generic kernels (`generic_head.glsl`)
   - specialized kernels (e.g. `soft_max.comp`, `sum_rows.comp`, `argmax.comp`)
2. Encode offset fields exactly as expected by shaders (`misalign_offsets` packing).
3. Port helper math for fast division parameters used by unary/reduction indexing.

### 3) Port ggml-style dispatch geometry

1. Add op-specific grid calculation helpers in host code.
2. Follow shader indexing assumptions (`gl_GlobalInvocationID` patterns with 512/262144 tiling where required).
3. Avoid one-size-fits-all dispatch dimensions for kernels that assume row-wise/group-wise execution.

### 4) Primitive wiring in phases

#### Phase A: Binary elementwise primitives (highest ROI) [In Progress]

Implement Vulkan `eval_gpu` for:

- `Add`
- `Subtract`
- `Multiply`
- `Divide`

Notes:

- Use compiled shader variants selected by dtype (f32/f16 combos where available).
- Preserve CPU fallback for unsupported dtypes/layouts.
- For `add.comp`, handle binding expectations (including partial buffer slot) consistently.

#### Phase B: Unary elementwise primitives [In Progress]

Implement Vulkan `eval_gpu` for first-wave unary ops that map cleanly to existing kernels:

- `Abs`, `Negative`
- `Sin`, `Cos`, `Tanh`, `Sigmoid`
- `Exp`, `Log`
- `Square`, `Sqrt`
- `Floor`, `Ceil`, `Round`
- `Arange`

Notes:

- Some kernels assume contiguous access; explicitly gate and fallback when stride/layout is unsupported.
- Preserve primitive semantics (e.g. reciprocal sqrt behavior if using `Sqrt(recip=true)`).

#### Phase C: Structured ops (targeted subset)

Implement initial scoped support for:

- `Reduce` (start with Sum on last axis via `sum_rows` path)
- `ArgReduce` (start with ArgMax on last axis)
- `Softmax` (common float path)
- `Scan` (start with cumsum)

Notes:

- These require specialized push constants and dispatch layouts.
- Keep partial feature support with explicit fallback guards.

### 5) Explicit fallback policy per primitive

For each wired primitive, add clear capability checks before dispatch:

- dtype support
- rank/axis constraints
- stride/contiguity requirements
- optional feature constraints (if any)

If unsupported, route through the existing CPU fallback helpers in `mlx/backend/vulkan/primitives.cpp`.

### 6) Testing and rollout

1. Add focused correctness tests per newly wired primitive:
   - compare Vulkan vs CPU outputs
   - include broadcasting and non-contiguous cases
   - verify fallback path correctness when unsupported
2. Run Vulkan build and relevant test targets after each phase.
3. Land changes incrementally (infra + Phase A, then Phase B, then Phase C).

## Recommended Execution Order

1. Dispatch infrastructure + ABI-safe push constants
2. Binary elementwise primitives
3. Unary elementwise primitives
4. Reduce/ArgReduce/Softmax/Scan subset
5. Expand coverage in additional slices

This order minimizes risk while delivering early end-to-end GPU wins on common ops.

## Primitive Tracking Table

Status legend:

- `Planned`: mapped and in scope, not wired yet
- `Partial`: initial constrained Vulkan path planned (fallback for unsupported cases)
- `Deferred`: intentionally postponed to later slices

| Primitive | Shader(s) | Push constants/layout | Fallback guards (initial) | Status |
|---|---|---|---|---|
| `Add` | `add_*` variants from `add.comp` | `generic_binary_head.glsl` (`misalign_offsets`, strides, shape) | currently constrained to same-shape, row-contiguous, rank<=4, offset==0; fallback otherwise | Partial |
| `Subtract` | `sub_*` variants from `sub.comp` | `generic_binary_head.glsl` | currently constrained to same-shape, row-contiguous, rank<=4, offset==0; fallback otherwise | Partial |
| `Multiply` | `mul_*` variants from `mul.comp` | `generic_binary_head.glsl` | currently constrained to same-shape, row-contiguous, rank<=4, offset==0; fallback otherwise | Partial |
| `Divide` | `div_*` variants from `div.comp` | `generic_binary_head.glsl` | same constraints as above; additionally f16 divide currently forced to fallback | Partial |
| `Abs` | `abs_f16`, `abs_f32` | `generic_head.glsl` (`KX`, scalar params unused) | currently constrained to same-dtype float16/float32 + row-contiguous + offset==0; fallback otherwise | Partial |
| `Negative` | `neg_f16`, `neg_f32` | `generic_head.glsl` | currently constrained to same-dtype float16/float32 + row-contiguous + offset==0; fallback otherwise | Partial |
| `Sin` | `sin_f32` (and optional `sin` unary path variants) | `generic_unary_head.glsl` or `generic_head.glsl` (variant-dependent) | kept on CPU fallback for now; revisit precision behavior | Planned |
| `Cos` | `cos_f32` | `generic_unary_head.glsl` or `generic_head.glsl` | kept on CPU fallback for now; revisit precision behavior | Planned |
| `Exp` | `exp_f16[_rte]`, `exp_f32[_rte]` | `generic_head.glsl` | currently constrained to same-dtype float16/float32 + row-contiguous + offset==0 (using `exp_f16_rte` for f16 path); fallback otherwise | Partial |
| `Log` | `log_f16[_rte]`, `log_f32[_rte]` | `generic_unary_head.glsl` | current Vulkan wiring only for natural log with supported layout; fallback otherwise | Partial |
| `Square` | `sqr_f32` (from `square.comp`) | `generic_unary_head.glsl` | current Vulkan wiring limited to f32 + supported layout; fallback otherwise | Partial |
| `Sqrt` | `sqrt_f32` | `generic_unary_head.glsl` | `recip=true` (`Rsqrt`) until dedicated mapping added; unsupported dtype/layout | Partial |
| `Floor` | `floor_f16`, `floor_f32` | `generic_head.glsl` | currently constrained to same-dtype float16/float32 + row-contiguous + offset==0; fallback otherwise | Partial |
| `Ceil` | `ceil_f16`, `ceil_f32` | `generic_head.glsl` | currently constrained to same-dtype float16/float32 + row-contiguous + offset==0; fallback otherwise | Partial |
| `Round` | `round_f16`, `round_f32` | `generic_head.glsl` | currently constrained to same-dtype float16/float32 + row-contiguous + offset==0; fallback otherwise | Partial |
| `Sigmoid` | `sigmoid_f16`, `sigmoid_f32` | `generic_head.glsl` | currently constrained to same-dtype float16/float32 + row-contiguous + offset==0; fallback otherwise | Partial |
| `Tanh` | `tanh_f16`, `tanh_f32` | `generic_head.glsl` | currently constrained to same-dtype float16/float32 + row-contiguous + offset==0; fallback otherwise | Partial |
| `Arange` | `arange_f32` | `generic_head.glsl` (`param1=start`, `param2=step`) | `float32` + row-contiguous + offset==0 only in first slice; fallback otherwise | Partial |
| `Reduce` | `sum_rows_f32` (first slice) | `sum_rows.glsl` struct + fastdiv fields | only `ReduceType::Sum`, last-axis or supported axis pattern, f32 first | Partial |
| `ArgReduce` | `argmax_f32` (first slice) | `generic_head.glsl` (`KX`,`KY`) + local-size specialization | only `ArgMax`, last-axis, f32 first | Partial |
| `Softmax` | `soft_max_f32`, `soft_max_large*`, optional `_f16` variants | `soft_max.comp` custom struct | common contiguous float path only in first slice | Partial |
| `Scan` | `cumsum_f32`, `cumsum_multipass*` | scan-specific custom params | `ReduceType::Sum` only, constrained axis/layout | Partial |
| `Equal` | (no direct kernel selected in this plan) | n/a | keep CPU fallback in initial slices | Deferred |
| `Convolution` | (`conv2d_*` kernels exist) | op-specific | defer until elementwise/reduce path is stable | Deferred |
| `Sort` / `ArgSort` | `argsort_f32`, `argsort_large_f32` | op-specific | defer until core dispatch + reductions are stable | Deferred |

Notes:

- Shader naming above refers to generated variant families emitted by `kernels/vulkan-shaders-gen.cpp`.
- The first implementation pass should prefer strict capability checks and fallback rather than broad-but-risky coverage.
