// Copyright © 2024 Apple Inc.

#include "mlx/distributed/primitives.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace mlx::core {

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no Vulkan implementation.");  \
  }

#define NO_GPU_USE_FALLBACK(func)     \
  bool func::use_fallback(Stream s) { \
    return true;                      \
  }                                   \
  NO_GPU_MULTI(func)

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

void Add::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto cpu_stream = default_stream(Device::cpu);
  Add cpu_add(cpu_stream);
  cpu_add.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto cpu_stream = default_stream(Device::cpu);
  Equal cpu_equal(cpu_stream, state());
  cpu_equal.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axes] = state();
  auto cpu_stream = default_stream(Device::cpu);
  Reduce cpu_reduce(cpu_stream, reduce_type, axes);
  cpu_reduce.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

void Divide::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto cpu_stream = default_stream(Device::cpu);
  Divide cpu_divide(cpu_stream);
  cpu_divide.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

void Subtract::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto cpu_stream = default_stream(Device::cpu);
  Subtract cpu_subtract(cpu_stream);
  cpu_subtract.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

void Minimum::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto cpu_stream = default_stream(Device::cpu);
  Minimum cpu_minimum(cpu_stream);
  cpu_minimum.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

void Maximum::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto cpu_stream = default_stream(Device::cpu);
  Maximum cpu_maximum(cpu_stream);
  cpu_maximum.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

void Multiply::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto cpu_stream = default_stream(Device::cpu);
  Multiply cpu_multiply(cpu_stream);
  cpu_multiply.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [shape, width] = state();
  auto cpu_stream = default_stream(Device::cpu);
  RandomBits cpu_random_bits(cpu_stream, shape, width);
  cpu_random_bits.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

bool fast::ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  return true;
}

bool fast::ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

bool fast::ScaledDotProductAttentionVJP::use_fallback(
    const array& q,
    Stream s) {
  return true;
}

NO_GPU(Abs)
// Add has CPU fallback above.
// AddMM implemented in matmul.cpp
NO_GPU(Arange)
NO_GPU(ArcCos)
NO_GPU(ArcCosh)
NO_GPU(ArcSin)
NO_GPU(ArcSinh)
NO_GPU(ArcTan)
NO_GPU(ArcTan2)
NO_GPU(ArcTanh)
NO_GPU(ArgPartition)
NO_GPU(ArgReduce)
NO_GPU(ArgSort)
NO_GPU(BitwiseBinary)
NO_GPU(BitwiseInvert)
// BlockMaskedMM implemented in matmul.cpp
NO_GPU(Ceil)
NO_GPU_MULTI(Compiled)
NO_GPU(Conjugate)
NO_GPU(Convolution)
NO_GPU(Cos)
NO_GPU(Cosh)
// Divide has CPU fallback above.
NO_GPU_MULTI(DivMod)
NO_GPU(Remainder)
// Equal has CPU fallback above.
NO_GPU(Erf)
NO_GPU(ErfInv)
NO_GPU(Exp)
NO_GPU(Expm1)
NO_GPU(FFT)
NO_GPU(Floor)
NO_GPU(Gather)
NO_GPU(GatherAxis)
NO_GPU(GatherMM)
NO_GPU(GatherQMM)
NO_GPU(Greater)
NO_GPU(GreaterEqual)
NO_GPU(Hadamard)
NO_GPU(Imag)
NO_GPU(Less)
NO_GPU(LessEqual)
NO_GPU(Load)
NO_GPU(Log)
NO_GPU(Log1p)
NO_GPU(LogicalNot)
NO_GPU(LogicalAnd)
NO_GPU(LogicalOr)
NO_GPU(LogAddExp)
NO_GPU(LogSumExp)
NO_GPU_MULTI(LUF)
// Matmul implemented in matmul.cpp
// Maximum has CPU fallback above.
// Minimum has CPU fallback above.
// Multiply has CPU fallback above.
NO_GPU(Negative)
NO_GPU(NotEqual)
NO_GPU(Partition)
NO_GPU(Power)
NO_GPU_MULTI(QRF)
NO_GPU(QuantizedMatmul)
NO_GPU(QQMatmul)
// RandomBits has CPU fallback above.
NO_GPU(Real)
// Reduce has CPU fallback above.
NO_GPU(Round)
NO_GPU(Scan)
NO_GPU(Scatter)
NO_GPU(ScatterAxis)
NO_GPU(Select)
NO_GPU(SegmentedMM)
NO_GPU(Sigmoid)
NO_GPU(Sign)
NO_GPU(Sin)
NO_GPU(Sinh)
NO_GPU(Softmax)
NO_GPU(Sort)
NO_GPU(Square)
NO_GPU(Sqrt)
// Subtract has CPU fallback above.
NO_GPU_MULTI(SVD)
NO_GPU(Tan)
NO_GPU(Tanh)
NO_GPU(Inverse)
NO_GPU(Cholesky)
NO_GPU_MULTI(Eigh)
NO_GPU_MULTI(Eig)
NO_GPU(MaskedScatter)

namespace fast {
NO_GPU_USE_FALLBACK(LayerNorm)
NO_GPU_MULTI(LayerNormVJP)
NO_GPU_USE_FALLBACK(RMSNorm)
NO_GPU_MULTI(RMSNormVJP)
NO_GPU_USE_FALLBACK(RoPE)
NO_GPU_MULTI(ScaledDotProductAttention)
NO_GPU_MULTI(ScaledDotProductAttentionVJP)
NO_GPU_MULTI(ConvertFP8)
NO_GPU_MULTI(Quantize)
NO_GPU_MULTI(CustomKernel)
} // namespace fast

namespace distributed {
NO_GPU_MULTI(AllReduce)
NO_GPU_MULTI(AllGather)
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
NO_GPU_MULTI(ReduceScatter)
} // namespace distributed

} // namespace mlx::core
