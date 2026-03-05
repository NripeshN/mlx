// Copyright © 2024 Apple Inc.

#include "mlx/distributed/primitives.h"
#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

#include <limits>
#include <string_view>
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

bool is_vulkan_float_dtype(Dtype dtype) {
  return dtype == float16 || dtype == float32;
}

std::string dtype_suffix(Dtype dtype) {
  switch (dtype) {
    case float16:
      return "f16";
    case float32:
      return "f32";
    default:
      return {};
  }
}

bool is_supported_elementwise_layout(const array& arr) {
  if (arr.ndim() > 4 || !arr.flags().row_contiguous || arr.offset() != 0) {
    return false;
  }
  if (arr.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  for (auto dim : arr.shape()) {
    if (dim < 0 || dim > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  for (auto stride : arr.strides()) {
    if (stride < 0 || stride > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool is_supported_unary_layout(const array& arr) {
  if (arr.ndim() > 4 || arr.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  if (arr.offset() < 0 || arr.offset() > 0xFFFF) {
    return false;
  }
  for (auto dim : arr.shape()) {
    if (dim < 0 || dim > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  for (auto stride : arr.strides()) {
    if (stride < 0 || stride > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool is_supported_generic_unary_layout(const array& arr) {
  return arr.flags().row_contiguous && arr.offset() == 0 &&
      arr.size() <= std::numeric_limits<uint32_t>::max();
}

template <typename Primitive>
bool try_eval_binary_op_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const char* op_name,
    Stream s) {
  if (inputs.size() != 2) {
    return false;
  }

  const auto& a = inputs[0];
  const auto& b = inputs[1];
  if (!is_vulkan_float_dtype(a.dtype()) || !is_vulkan_float_dtype(b.dtype()) ||
      !is_vulkan_float_dtype(out.dtype())) {
    return false;
  }

  if (std::string_view(op_name) == "div" &&
      (a.dtype() == float16 || b.dtype() == float16 ||
       out.dtype() == float16)) {
    return false;
  }

  if (a.shape() != out.shape() || b.shape() != out.shape()) {
    return false;
  }

  if (!is_supported_elementwise_layout(a) ||
      !is_supported_elementwise_layout(b)) {
    return false;
  }

  auto suffix_a = dtype_suffix(a.dtype());
  auto suffix_b = dtype_suffix(b.dtype());
  auto suffix_out = dtype_suffix(out.dtype());
  if (suffix_a.empty() || suffix_b.empty() || suffix_out.empty()) {
    return false;
  }

  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt);
  if (!is_supported_elementwise_layout(out)) {
    return false;
  }

  if (out.size() == 0) {
    return true;
  }

  std::string shader_name =
      std::string(op_name) + "_" + suffix_a + "_" + suffix_b + "_" + suffix_out;
  if (out.dtype() == float16) {
    shader_name += "_rte";
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_binary_op(a, b, out, shader_name, command_buffer, s);
    vulkan::end_command_recording(s.index);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

template <typename Primitive>
void eval_binary_vulkan_or_cpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op_name,
    Stream s) {
  if (try_eval_binary_op_vulkan<Primitive>(inputs, out, op_name, s)) {
    return;
  }
  eval_cpu_fallback<Primitive>(inputs, out);
}

template <typename Primitive>
bool try_eval_unary_op_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::string& shader_name,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f) {
  if (inputs.size() != 1) {
    return false;
  }

  const auto& in = inputs[0];
  set_unary_output_data(in, out);
  if (!is_supported_unary_layout(in) || !is_supported_unary_layout(out)) {
    return false;
  }

  if (out.size() == 0) {
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_unary_op(
        in, out, shader_name, command_buffer, s, param1, param2);
    vulkan::end_command_recording(s.index);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

template <typename Primitive>
void eval_unary_vulkan_or_cpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string& shader_name,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f) {
  if (try_eval_unary_op_vulkan<Primitive>(
          inputs, out, shader_name, s, param1, param2)) {
    return;
  }
  eval_cpu_fallback<Primitive>(inputs, out);
}

template <typename Primitive>
bool try_eval_generic_unary_op_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::string& shader_name,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f,
    float param3 = 0.0f,
    float param4 = 0.0f) {
  if (inputs.size() != 1) {
    return false;
  }

  const auto& in = inputs[0];
  if (!is_vulkan_float_dtype(in.dtype()) || in.dtype() != out.dtype()) {
    return false;
  }

  set_unary_output_data(in, out);
  if (!is_supported_generic_unary_layout(in) ||
      !is_supported_generic_unary_layout(out)) {
    return false;
  }

  if (out.size() == 0) {
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_generic_unary_op(
        in,
        out,
        shader_name,
        command_buffer,
        s,
        param1,
        param2,
        param3,
        param4);
    vulkan::end_command_recording(s.index);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

template <typename Primitive>
void eval_generic_unary_vulkan_or_cpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string& shader_name,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f,
    float param3 = 0.0f,
    float param4 = 0.0f) {
  if (try_eval_generic_unary_op_vulkan<Primitive>(
          inputs, out, shader_name, s, param1, param2, param3, param4)) {
    return;
  }
  eval_cpu_fallback<Primitive>(inputs, out);
}

bool try_eval_arange_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s,
    double start,
    double step) {
  if (!inputs.empty() || out.dtype() != float32 ||
      !is_supported_generic_unary_layout(out)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_arange_op(
        out,
        "arange_f32",
        command_buffer,
        s,
        static_cast<float>(start),
        static_cast<float>(step));
    vulkan::end_command_recording(s.index);
    return true;
  } catch (const std::runtime_error&) {
    return false;
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

CPU_FALLBACK_STATE(Equal)
CPU_FALLBACK_STATE(Reduce)
CPU_FALLBACK(Minimum)
CPU_FALLBACK(Maximum)
CPU_FALLBACK_STATE(RandomBits)

void Add::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval_binary_vulkan_or_cpu<Add>(inputs, out, "add", stream());
}

void Divide::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval_binary_vulkan_or_cpu<Divide>(inputs, out, "div", stream());
}

void Subtract::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval_binary_vulkan_or_cpu<Subtract>(inputs, out, "sub", stream());
}

void Multiply::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval_binary_vulkan_or_cpu<Multiply>(inputs, out, "mul", stream());
}

void Abs::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto suffix = dtype_suffix(out.dtype());
    if (!suffix.empty()) {
      eval_generic_unary_vulkan_or_cpu<Abs>(
          inputs, out, "abs_" + suffix, stream());
      return;
    }
  }
  eval_cpu_fallback<Abs>(inputs, out);
}

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [start, stop, step] = state();
  if (try_eval_arange_vulkan(inputs, out, stream(), start, step)) {
    return;
  }
  eval_cpu_fallback<Arange>(inputs, out, start, stop, step);
}

void Ceil::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto suffix = dtype_suffix(out.dtype());
    if (!suffix.empty()) {
      eval_generic_unary_vulkan_or_cpu<Ceil>(
          inputs, out, "ceil_" + suffix, stream());
      return;
    }
  }
  eval_cpu_fallback<Ceil>(inputs, out);
}

void Cos::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval_cpu_fallback<Cos>(inputs, out);
}

void Exp::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    if (out.dtype() == float32) {
      if (try_eval_generic_unary_op_vulkan<Exp>(
              inputs, out, "exp_f32", stream())) {
        return;
      }
    } else if (out.dtype() == float16) {
      if (try_eval_generic_unary_op_vulkan<Exp>(
              inputs, out, "exp_f16_rte", stream())) {
        return;
      }
    }
  }
  eval_cpu_fallback<Exp>(inputs, out);
}

void Floor::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto suffix = dtype_suffix(out.dtype());
    if (!suffix.empty()) {
      eval_generic_unary_vulkan_or_cpu<Floor>(
          inputs, out, "floor_" + suffix, stream());
      return;
    }
  }
  eval_cpu_fallback<Floor>(inputs, out);
}

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    if (state() == Log::e && out.dtype() == float32) {
      if (try_eval_unary_op_vulkan<Log>(inputs, out, "log_f32", stream())) {
        return;
      }
    }
    if (state() == Log::e && out.dtype() == float16) {
      if (try_eval_unary_op_vulkan<Log>(inputs, out, "log_f16_rte", stream())) {
        return;
      }
    }
  }
  eval_cpu_fallback<Log>(inputs, out, state());
}

void Sin::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval_cpu_fallback<Sin>(inputs, out);
}

void Negative::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto suffix = dtype_suffix(out.dtype());
    if (!suffix.empty()) {
      eval_generic_unary_vulkan_or_cpu<Negative>(
          inputs, out, "neg_" + suffix, stream());
      return;
    }
  }
  eval_cpu_fallback<Negative>(inputs, out);
}

void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto suffix = dtype_suffix(out.dtype());
    if (!suffix.empty()) {
      eval_generic_unary_vulkan_or_cpu<Round>(
          inputs, out, "round_" + suffix, stream());
      return;
    }
  }
  eval_cpu_fallback<Round>(inputs, out);
}

void Sigmoid::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto suffix = dtype_suffix(out.dtype());
    if (!suffix.empty()) {
      eval_generic_unary_vulkan_or_cpu<Sigmoid>(
          inputs, out, "sigmoid_" + suffix, stream());
      return;
    }
  }
  eval_cpu_fallback<Sigmoid>(inputs, out);
}

void Square::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    eval_unary_vulkan_or_cpu<Square>(inputs, out, "sqr_f32", stream());
    return;
  }
  eval_cpu_fallback<Square>(inputs, out);
}

void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!state() && inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    eval_unary_vulkan_or_cpu<Sqrt>(inputs, out, "sqrt_f32", stream());
    return;
  }
  eval_cpu_fallback<Sqrt>(inputs, out, state());
}

void Tanh::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto suffix = dtype_suffix(out.dtype());
    if (!suffix.empty()) {
      eval_generic_unary_vulkan_or_cpu<Tanh>(
          inputs, out, "tanh_" + suffix, stream());
      return;
    }
  }
  eval_cpu_fallback<Tanh>(inputs, out);
}

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

// Abs implemented above.
// Add implemented above.
// AddMM implemented in matmul.cpp
// Arange implemented above.
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
// Ceil implemented above.
// Compiled has CPU fallback above.
CPU_FALLBACK(Conjugate)
CPU_FALLBACK_STATE(Convolution)
// Cos implemented above.
CPU_FALLBACK(Cosh)
// Divide implemented above.
CPU_FALLBACK_MULTI(DivMod)
CPU_FALLBACK(Remainder)
// Equal has CPU fallback above.
CPU_FALLBACK(Erf)
CPU_FALLBACK(ErfInv)
// Exp implemented above.
CPU_FALLBACK(Expm1)
CPU_FALLBACK_STATE(FFT)
// Floor implemented above.
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
// Log implemented above.
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
// Multiply implemented above.
// Negative implemented above.
CPU_FALLBACK(NotEqual)
CPU_FALLBACK_STATE(Partition)
CPU_FALLBACK(Power)
CPU_FALLBACK_MULTI(QRF)
CPU_FALLBACK_STATE(QuantizedMatmul)
CPU_FALLBACK_STATE(QQMatmul)
// RandomBits has CPU fallback above.
CPU_FALLBACK(Real)
// Reduce has CPU fallback above.
// Round implemented above.
CPU_FALLBACK_STATE(Scan)
CPU_FALLBACK_STATE(Scatter)
CPU_FALLBACK_STATE(ScatterAxis)
CPU_FALLBACK(Select)
CPU_FALLBACK(SegmentedMM)
// Sigmoid implemented above.
CPU_FALLBACK(Sign)
// Sin implemented above.
CPU_FALLBACK(Sinh)
CPU_FALLBACK_STATE(Softmax)
CPU_FALLBACK_STATE(Sort)
// Square implemented above.
// Sqrt implemented above.
// Subtract implemented above.
CPU_FALLBACK_MULTI_STATE(SVD)
CPU_FALLBACK(Tan)
// Tanh implemented above.
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
