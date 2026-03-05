// Copyright © 2024 Apple Inc.

#include "mlx/distributed/primitives.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

#include <tuple>
#include <type_traits>
#include <utility>

namespace mlx::core {

namespace {

template <typename Primitive, typename... Args>
void eval_cpu_fallback(
    const std::vector<array>& inputs,
    array& out,
    Args&&... args) {
  auto cpu_stream = default_stream(Device::cpu);
  Primitive cpu_primitive(cpu_stream, std::forward<Args>(args)...);
  cpu_primitive.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

template <typename Primitive, typename... Args>
void eval_cpu_fallback_multi(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    Args&&... args) {
  auto cpu_stream = default_stream(Device::cpu);
  Primitive cpu_primitive(cpu_stream, std::forward<Args>(args)...);
  cpu_primitive.eval_cpu(inputs, outputs);
  synchronize(cpu_stream);
}

template <typename T, typename = void>
struct is_tuple_like : std::false_type {};

template <typename T>
struct is_tuple_like<
    T,
    std::void_t<decltype(std::tuple_size<std::decay_t<T>>::value)>>
    : std::true_type {};

template <typename Primitive, typename State>
void eval_cpu_fallback_with_state(
    const std::vector<array>& inputs,
    array& out,
    State&& state) {
  if constexpr (is_tuple_like<State>::value) {
    std::apply(
        [&](auto&&... state_args) {
          eval_cpu_fallback<Primitive>(
              inputs, out, std::forward<decltype(state_args)>(state_args)...);
        },
        std::forward<State>(state));
  } else {
    eval_cpu_fallback<Primitive>(inputs, out, std::forward<State>(state));
  }
}

template <typename Primitive, typename State>
void eval_cpu_fallback_multi_with_state(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    State&& state) {
  if constexpr (is_tuple_like<State>::value) {
    std::apply(
        [&](auto&&... state_args) {
          eval_cpu_fallback_multi<Primitive>(
              inputs,
              outputs,
              std::forward<decltype(state_args)>(state_args)...);
        },
        std::forward<State>(state));
  } else {
    eval_cpu_fallback_multi<Primitive>(
        inputs, outputs, std::forward<State>(state));
  }
}

} // namespace

#define CPU_FALLBACK(func)                                            \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    eval_cpu_fallback<func>(inputs, out);                             \
  }

#define CPU_FALLBACK_STATE(func)                                      \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    eval_cpu_fallback_with_state<func>(inputs, out, state());         \
  }

#define CPU_FALLBACK_MULTI(func)                                       \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    eval_cpu_fallback_multi<func>(inputs, outputs);                    \
  }

#define CPU_FALLBACK_MULTI_STATE(func)                                  \
  void func::eval_gpu(                                                  \
      const std::vector<array>& inputs, std::vector<array>& outputs) {  \
    eval_cpu_fallback_multi_with_state<func>(inputs, outputs, state()); \
  }

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

CPU_FALLBACK(Add)
CPU_FALLBACK_STATE(Equal)
CPU_FALLBACK_STATE(Reduce)
CPU_FALLBACK(Divide)
CPU_FALLBACK(Subtract)
CPU_FALLBACK(Minimum)
CPU_FALLBACK(Maximum)
CPU_FALLBACK(Multiply)
CPU_FALLBACK_STATE(RandomBits)

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto cpu_stream = default_stream(Device::cpu);
  Compiled cpu_compiled(cpu_stream, inputs_, outputs_, tape_, constant_ids_);
  cpu_compiled.eval_cpu(inputs, outputs);
  synchronize(cpu_stream);
}

void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto cpu_stream = default_stream(Device::cpu);
  Load cpu_load(cpu_stream, reader_, offset_, swap_endianness_);
  cpu_load.eval_cpu(inputs, out);
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

CPU_FALLBACK(Abs)
// Add has CPU fallback above.
// AddMM implemented in matmul.cpp
CPU_FALLBACK_STATE(Arange)
CPU_FALLBACK(ArcCos)
CPU_FALLBACK(ArcCosh)
CPU_FALLBACK(ArcSin)
CPU_FALLBACK(ArcSinh)
CPU_FALLBACK(ArcTan)
CPU_FALLBACK(ArcTan2)
CPU_FALLBACK(ArcTanh)
CPU_FALLBACK_STATE(ArgPartition)
CPU_FALLBACK_STATE(ArgReduce)
CPU_FALLBACK_STATE(ArgSort)
CPU_FALLBACK_STATE(BitwiseBinary)
CPU_FALLBACK(BitwiseInvert)
// BlockMaskedMM implemented in matmul.cpp
CPU_FALLBACK(Ceil)
// Compiled has CPU fallback above.
CPU_FALLBACK(Conjugate)
CPU_FALLBACK_STATE(Convolution)
CPU_FALLBACK(Cos)
CPU_FALLBACK(Cosh)
// Divide has CPU fallback above.
CPU_FALLBACK_MULTI(DivMod)
CPU_FALLBACK(Remainder)
// Equal has CPU fallback above.
CPU_FALLBACK(Erf)
CPU_FALLBACK(ErfInv)
CPU_FALLBACK(Exp)
CPU_FALLBACK(Expm1)
CPU_FALLBACK_STATE(FFT)
CPU_FALLBACK(Floor)
CPU_FALLBACK_STATE(Gather)
CPU_FALLBACK_STATE(GatherAxis)
CPU_FALLBACK_STATE(GatherMM)
CPU_FALLBACK_STATE(GatherQMM)
CPU_FALLBACK(Greater)
CPU_FALLBACK(GreaterEqual)
CPU_FALLBACK_STATE(Hadamard)
CPU_FALLBACK(Imag)
CPU_FALLBACK(Less)
CPU_FALLBACK(LessEqual)
// Load has CPU fallback above.
CPU_FALLBACK_STATE(Log)
CPU_FALLBACK(Log1p)
CPU_FALLBACK(LogicalNot)
CPU_FALLBACK(LogicalAnd)
CPU_FALLBACK(LogicalOr)
CPU_FALLBACK(LogAddExp)
CPU_FALLBACK(LogSumExp)
CPU_FALLBACK_MULTI(LUF)
// Matmul implemented in matmul.cpp
// Maximum has CPU fallback above.
// Minimum has CPU fallback above.
// Multiply has CPU fallback above.
CPU_FALLBACK(Negative)
CPU_FALLBACK(NotEqual)
CPU_FALLBACK_STATE(Partition)
CPU_FALLBACK(Power)
CPU_FALLBACK_MULTI(QRF)
CPU_FALLBACK_STATE(QuantizedMatmul)
CPU_FALLBACK_STATE(QQMatmul)
// RandomBits has CPU fallback above.
CPU_FALLBACK(Real)
// Reduce has CPU fallback above.
CPU_FALLBACK(Round)
CPU_FALLBACK_STATE(Scan)
CPU_FALLBACK_STATE(Scatter)
CPU_FALLBACK_STATE(ScatterAxis)
CPU_FALLBACK(Select)
CPU_FALLBACK(SegmentedMM)
CPU_FALLBACK(Sigmoid)
CPU_FALLBACK(Sign)
CPU_FALLBACK(Sin)
CPU_FALLBACK(Sinh)
CPU_FALLBACK_STATE(Softmax)
CPU_FALLBACK_STATE(Sort)
CPU_FALLBACK(Square)
CPU_FALLBACK_STATE(Sqrt)
// Subtract has CPU fallback above.
CPU_FALLBACK_MULTI_STATE(SVD)
CPU_FALLBACK(Tan)
CPU_FALLBACK(Tanh)
CPU_FALLBACK_STATE(Inverse)
CPU_FALLBACK_STATE(Cholesky)
CPU_FALLBACK_MULTI_STATE(Eigh)
CPU_FALLBACK_MULTI_STATE(Eig)
CPU_FALLBACK(MaskedScatter)

void fast::ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto cpu_stream = default_stream(Device::cpu);
  fast::ConvertFP8 cpu_convert(cpu_stream, state());
  cpu_convert.eval_cpu(inputs, outputs);
  synchronize(cpu_stream);
}

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto fallback_outputs = fallback_(inputs);
  if (fallback_outputs.size() != outputs.size()) {
    throw std::runtime_error(
        "[vulkan::Quantize::eval_gpu] Fallback output count mismatch.");
  }
  eval(fallback_outputs);
  for (int i = 0; i < outputs.size(); ++i) {
    outputs[i].copy_shared_buffer(fallback_outputs[i]);
  }
}

namespace fast {
NO_GPU_USE_FALLBACK(LayerNorm)
NO_GPU_MULTI(LayerNormVJP)
NO_GPU_USE_FALLBACK(RMSNorm)
NO_GPU_MULTI(RMSNormVJP)
NO_GPU_USE_FALLBACK(RoPE)
NO_GPU_MULTI(ScaledDotProductAttention)
NO_GPU_MULTI(ScaledDotProductAttentionVJP)
// ConvertFP8 and Quantize have CPU fallbacks above.
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
