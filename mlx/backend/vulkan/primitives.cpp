// Copyright © 2024 Apple Inc.

#include "mlx/distributed/primitives.h"
#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

#define CPU_FALLBACK(func)                                            \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    eval_cpu_fallback_on_stream<func>(inputs, out, stream());         \
  }

#define CPU_FALLBACK_STATE(func)                                      \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    eval_cpu_fallback_with_state_on_stream<func>(                     \
        inputs, out, stream(), state());                              \
  }

#define CPU_FALLBACK_MULTI(func)                                        \
  void func::eval_gpu(                                                  \
      const std::vector<array>& inputs, std::vector<array>& outputs) {  \
    eval_cpu_fallback_multi_on_stream<func>(inputs, outputs, stream()); \
  }

#define CPU_FALLBACK_MULTI_STATE(func)                                 \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    eval_cpu_fallback_multi_with_state_on_stream<func>(                \
        inputs, outputs, stream(), state());                           \
  }

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no Vulkan implementation.");  \
  }

#define NO_GPU_MULTI_STATE(func)                                       \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no Vulkan implementation.");  \
  }

#define NO_GPU_STATE(func)                                            \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

#define NO_GPU_USE_FALLBACK(func)                             \
  bool func::use_fallback(Stream s) {                         \
    trace_use_fallback(#func, s, "no Vulkan implementation"); \
    return true;                                              \
  }                                                           \
  NO_GPU_MULTI(func)

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

// Primitives with state that have CPU fallbacks
CPU_FALLBACK_STATE(Equal)

// Primitives implemented in other files:
// - binary.cpp: Add, Minimum, Maximum, Divide, Subtract, Multiply
// - unary.cpp: Abs, Ceil, Cos, Exp, Erf, ErfInv, Floor, Log, Sin, etc.
// - reduce.cpp: Reduce, ArgReduce
// - softmax.cpp: Softmax, LogSumExp
// - gather.cpp: Gather, GatherAxis
// - scan.cpp: Scan
// - arange.cpp: Arange
// - rope.cpp: RoPE (no-op fallback)
// - fast.cpp: LayerNorm, RMSNorm, Quantize, ConvertFP8, CustomKernel, SDPA
// - random.cpp: RandomBits

// Load primitive - throw NYI like Metal backend
void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[Load::eval_gpu] Not implemented.");
}

// CPU fallbacks for primitives not implemented on Vulkan
CPU_FALLBACK(ArcCos)
CPU_FALLBACK(ArcCosh)
CPU_FALLBACK(ArcSin)
CPU_FALLBACK(ArcSinh)
CPU_FALLBACK(ArcTan)
CPU_FALLBACK(ArcTan2)
CPU_FALLBACK(ArcTanh)
CPU_FALLBACK_STATE(ArgPartition)
CPU_FALLBACK_STATE(ArgSort)
CPU_FALLBACK_STATE(BitwiseBinary)
CPU_FALLBACK(BitwiseInvert)
CPU_FALLBACK(Conjugate)
CPU_FALLBACK_STATE(Convolution)
CPU_FALLBACK(Cosh)
CPU_FALLBACK_MULTI(DivMod)
CPU_FALLBACK(Remainder)
CPU_FALLBACK(Expm1)
CPU_FALLBACK_STATE(FFT)
CPU_FALLBACK_STATE(GatherMM)
CPU_FALLBACK_STATE(GatherQMM)
CPU_FALLBACK(Greater)
CPU_FALLBACK_STATE(Hadamard)
CPU_FALLBACK(Imag)
CPU_FALLBACK(Less)
CPU_FALLBACK(LessEqual)
CPU_FALLBACK(Log1p)
CPU_FALLBACK(LogicalNot)
CPU_FALLBACK(LogicalAnd)
CPU_FALLBACK(LogicalOr)
CPU_FALLBACK(LogAddExp)
// Linear algebra operations - throw NYI like Metal backend
NO_GPU_MULTI(LUF)
NO_GPU_MULTI(QRF)
NO_GPU_STATE(Inverse)
NO_GPU_STATE(Cholesky)
NO_GPU_MULTI_STATE(Eigh)
NO_GPU_MULTI_STATE(Eig)
NO_GPU_MULTI_STATE(SVD)

CPU_FALLBACK(NotEqual)
CPU_FALLBACK_STATE(Partition)
CPU_FALLBACK(Power)
CPU_FALLBACK_STATE(QuantizedMatmul)
CPU_FALLBACK_STATE(QQMatmul)
CPU_FALLBACK(Real)
CPU_FALLBACK(Sign)
CPU_FALLBACK(Sinh)
CPU_FALLBACK_STATE(Sort)
CPU_FALLBACK(Tan)
CPU_FALLBACK(MaskedScatter)
CPU_FALLBACK_STATE(Scatter)
CPU_FALLBACK_STATE(ScatterAxis)
CPU_FALLBACK(Select)
CPU_FALLBACK(SegmentedMM)

namespace distributed {
NO_GPU_MULTI(AllReduce)
NO_GPU_MULTI(AllGather)
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
NO_GPU_MULTI(ReduceScatter)
} // namespace distributed

} // namespace mlx::core
