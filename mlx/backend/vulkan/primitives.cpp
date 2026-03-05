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
constexpr vulkan::BinaryDispatchVariant binary_dispatch_variant() {
  if constexpr (std::is_same_v<Primitive, Add>) {
    return vulkan::BinaryDispatchVariant::AddWithPartials;
  } else {
    return vulkan::BinaryDispatchVariant::Standard;
  }
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

  array a = inputs[0];
  array b = inputs[1];
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

  if (!is_supported_elementwise_layout(a)) {
    a = contiguous_copy_gpu(a, s);
  }
  if (!is_supported_elementwise_layout(b)) {
    b = contiguous_copy_gpu(b, s);
  }

  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work =
      staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;

  auto suffix_a = dtype_suffix(a.dtype());
  auto suffix_b = dtype_suffix(b.dtype());
  auto suffix_out = dtype_suffix(out.dtype());
  if (suffix_a.empty() || suffix_b.empty() || suffix_out.empty()) {
    return false;
  }

  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out_work, bopt);
  if (!is_supported_elementwise_layout(out_work)) {
    return false;
  }

  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  std::string shader_name =
      std::string(op_name) + "_" + suffix_a + "_" + suffix_b + "_" + suffix_out;
  if (out_work.dtype() == float16) {
    shader_name += "_rte";
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_binary_op(
        a,
        b,
        out_work,
        shader_name,
        command_buffer,
        s,
        binary_dispatch_variant<Primitive>());
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
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

  array in = inputs[0];
  if (!is_vulkan_float_dtype(in.dtype()) || in.dtype() != out.dtype()) {
    return false;
  }

  if (!is_supported_generic_unary_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }

  const bool staged_output = !is_supported_generic_unary_layout(out);
  array out_work =
      staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;

  set_unary_output_data(in, out_work);
  if (!is_supported_generic_unary_layout(in) ||
      !is_supported_generic_unary_layout(out_work)) {
    return false;
  }

  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_generic_unary_op(
        in,
        out_work,
        shader_name,
        command_buffer,
        s,
        param1,
        param2,
        param3,
        param4);
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
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

template <typename Primitive>
void eval_generic_unary_suffix_vulkan_or_cpu(
    const std::vector<array>& inputs,
    array& out,
    std::string_view op_name,
    Stream s,
    bool f16_with_rte = false) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto suffix = dtype_suffix(out.dtype());
    if (!suffix.empty()) {
      std::string shader_name = std::string(op_name) + "_" + suffix;
      if (f16_with_rte && out.dtype() == float16) {
        shader_name += "_rte";
      }
      eval_generic_unary_vulkan_or_cpu<Primitive>(inputs, out, shader_name, s);
      return;
    }
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

int normalize_axis(int axis, int ndim) {
  if (axis < 0) {
    axis += ndim;
  }
  return axis;
}

bool has_keepdims_axis_shape(const array& in, const array& out, int axis) {
  if (in.ndim() != out.ndim()) {
    return false;
  }

  for (int i = 0; i < in.ndim(); ++i) {
    const int64_t expected = (i == axis) ? 1 : in.shape(i);
    if (out.shape(i) != expected) {
      return false;
    }
  }
  return true;
}

bool has_squeezed_axis_shape(const array& in, const array& out, int axis) {
  if (in.ndim() - 1 != out.ndim()) {
    return false;
  }

  int out_dim = 0;
  for (int in_dim = 0; in_dim < in.ndim(); ++in_dim) {
    if (in_dim == axis) {
      continue;
    }
    if (out.shape(out_dim) != in.shape(in_dim)) {
      return false;
    }
    out_dim++;
  }
  return true;
}

Shape keepdims_shape_for_axis(const array& in, int axis) {
  auto shape = in.shape();
  shape[axis] = 1;
  return shape;
}

bool try_eval_reduce_sum_rows_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    Stream s) {
  if (inputs.size() != 1 || reduce_type != Reduce::Sum || axes.size() != 1) {
    return false;
  }

  array in = inputs[0];
  if (in.dtype() != float32 || out.dtype() != float32) {
    return false;
  }

  const int axis = normalize_axis(axes[0], in.ndim());

  if (in.ndim() == 0 || in.ndim() > 4 || out.ndim() > 4) {
    return false;
  }

  const bool out_is_keepdims = has_keepdims_axis_shape(in, out, axis);
  const bool out_is_squeezed = has_squeezed_axis_shape(in, out, axis);
  if (!out_is_keepdims && !out_is_squeezed) {
    return false;
  }

  array in_kernel = in;
  if (axis != in.ndim() - 1) {
    in_kernel = swapaxes_in_eval(in, axis, in.ndim() - 1);
  }

  if (!in_kernel.flags().row_contiguous ||
      !is_supported_unary_layout(in_kernel)) {
    in_kernel = contiguous_copy_gpu(in_kernel, s);
  }

  array kernel_out(
      keepdims_shape_for_axis(in_kernel, in_kernel.ndim() - 1),
      out.dtype(),
      nullptr,
      {});

  const bool staged_output = !kernel_out.flags().row_contiguous ||
      !is_supported_unary_layout(kernel_out);
  array out_work = staged_output
      ? array(kernel_out.shape(), kernel_out.dtype(), nullptr, {})
      : kernel_out;

  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
    }
    if (out_is_squeezed) {
      auto squeezed = reshape_in_eval(kernel_out, out.shape(), s);
      copy_gpu(squeezed, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_sum_rows_op(
        in_kernel, out_work, "sum_rows_f32", command_buffer, s, 1.0f);
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
    }

    array restored_keepdims = kernel_out;
    if (axis != in.ndim() - 1) {
      restored_keepdims = swapaxes_in_eval(kernel_out, axis, in.ndim() - 1);
    }

    if (out_is_squeezed) {
      auto squeezed = reshape_in_eval(restored_keepdims, out.shape(), s);
      copy_gpu(squeezed, out, CopyType::GeneralGeneral, s);
    } else {
      copy_gpu(restored_keepdims, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

bool try_eval_arg_reduce_vulkan(
    const std::vector<array>& inputs,
    array& out,
    ArgReduce::ReduceType reduce_type,
    int axis,
    Stream s) {
  if (inputs.size() != 1 || reduce_type != ArgReduce::ArgMax) {
    return false;
  }

  array in = inputs[0];
  if (in.ndim() == 0 || in.dtype() != float32 || out.dtype() != uint32) {
    return false;
  }

  axis = normalize_axis(axis, in.ndim());

  const bool out_is_keepdims = has_keepdims_axis_shape(in, out, axis);
  const bool out_is_squeezed = has_squeezed_axis_shape(in, out, axis);
  if (!out_is_keepdims && !out_is_squeezed) {
    return false;
  }

  array in_kernel = in;
  if (axis != in.ndim() - 1) {
    in_kernel = swapaxes_in_eval(in, axis, in.ndim() - 1);
  }

  if (in_kernel.size() > std::numeric_limits<uint32_t>::max() ||
      out.size() > std::numeric_limits<uint32_t>::max() ||
      in_kernel.shape(in_kernel.ndim() - 1) >
          std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  if (!in_kernel.flags().row_contiguous || in_kernel.offset() != 0 ||
      !is_supported_unary_layout(in_kernel)) {
    in_kernel = contiguous_copy_gpu(in_kernel, s);
  }

  array kernel_out(
      keepdims_shape_for_axis(in_kernel, in_kernel.ndim() - 1),
      out.dtype(),
      nullptr,
      {});

  const bool staged_output = !kernel_out.flags().row_contiguous ||
      kernel_out.offset() != 0 || !is_supported_unary_layout(kernel_out);
  array out_work = staged_output
      ? array(kernel_out.shape(), kernel_out.dtype(), nullptr, {})
      : kernel_out;

  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
    }
    if (out_is_squeezed) {
      auto squeezed = reshape_in_eval(kernel_out, out.shape(), s);
      copy_gpu(squeezed, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_argmax_op(
        in_kernel, out_work, "argmax_f32", command_buffer, s);
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
    }

    array restored_keepdims = kernel_out;
    if (axis != in.ndim() - 1) {
      restored_keepdims = swapaxes_in_eval(kernel_out, axis, in.ndim() - 1);
    }

    if (out_is_squeezed) {
      auto squeezed = reshape_in_eval(restored_keepdims, out.shape(), s);
      copy_gpu(squeezed, out, CopyType::GeneralGeneral, s);
    } else {
      copy_gpu(restored_keepdims, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

bool try_eval_softmax_vulkan(
    const std::vector<array>& inputs,
    array& out,
    bool /*precise*/,
    Stream s) {
  if (inputs.size() != 1) {
    return false;
  }

  array in = inputs[0];
  const bool f32_io = in.dtype() == float32 && out.dtype() == float32;
  const bool f16_io = in.dtype() == float16 && out.dtype() == float16;
  if (in.ndim() == 0 || (!f32_io && !f16_io)) {
    return false;
  }

  const bool use_f16_variant = f16_io;
  if (use_f16_variant) {
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }

  array softmax_out_target =
      use_f16_variant ? array(out.shape(), float32, nullptr, {}) : out;

  if (!in.flags().contiguous || in.offset() != 0 || in.strides().back() != 1 ||
      !is_supported_unary_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }

  const bool staged_output = !softmax_out_target.flags().contiguous ||
      softmax_out_target.offset() != 0 ||
      softmax_out_target.strides().back() != 1 ||
      !is_supported_unary_layout(softmax_out_target);
  array out_work = staged_output
      ? array(
            softmax_out_target.shape(), softmax_out_target.dtype(), nullptr, {})
      : softmax_out_target;

  set_unary_output_data(in, out_work);
  if (in.shape() != out_work.shape()) {
    return false;
  }

  if (in.size() > std::numeric_limits<uint32_t>::max() ||
      out_work.size() > std::numeric_limits<uint32_t>::max() ||
      in.shape(in.ndim() - 1) > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const uint32_t row_width = static_cast<uint32_t>(in.shape(in.ndim() - 1));
  const bool use_large_softmax = row_width > 16384u;

  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, softmax_out_target, CopyType::GeneralGeneral, s);
    }
    if (use_f16_variant) {
      copy_gpu(softmax_out_target, out, CopyType::General, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    if (use_large_softmax) {
      vulkan::dispatch_softmax_large_op(
          in,
          out_work,
          use_f16_variant ? "soft_max_large1_f32_f16" : "soft_max_large1_f32",
          use_f16_variant ? "soft_max_large2_f32_f16" : "soft_max_large2_f32",
          use_f16_variant ? "soft_max_large3_f32_f16" : "soft_max_large3_f32",
          command_buffer,
          s);
    } else {
      vulkan::dispatch_softmax_op(
          in,
          out_work,
          use_f16_variant ? "soft_max_f32_f16" : "soft_max_f32",
          command_buffer,
          s);
    }
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(out_work, softmax_out_target, CopyType::GeneralGeneral, s);
    }
    if (use_f16_variant) {
      copy_gpu(softmax_out_target, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

bool try_eval_scan_cumsum_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Scan::ReduceType reduce_type,
    int axis,
    bool reverse,
    bool inclusive,
    Stream s) {
  if (inputs.size() != 1 || reduce_type != Scan::Sum) {
    return false;
  }

  array in = inputs[0];
  if (in.ndim() == 0 || in.dtype() != float32 || out.dtype() != float32) {
    return false;
  }

  int normalized_axis = normalize_axis(axis, in.ndim());
  if (normalized_axis < 0 || normalized_axis >= in.ndim()) {
    return false;
  }

  array in_kernel = in;
  if (normalized_axis != in.ndim() - 1) {
    in_kernel = swapaxes_in_eval(in, normalized_axis, in.ndim() - 1);
  }

  auto reverse_last_axis_contiguous = [&](const array& arr) {
    Shape starts(arr.ndim(), 0);
    Shape strides(arr.ndim(), 1);
    starts[arr.ndim() - 1] = arr.shape(arr.ndim() - 1) - 1;
    strides[arr.ndim() - 1] = -1;
    array reversed_view(arr.shape(), arr.dtype(), nullptr, {});
    slice_gpu(arr, reversed_view, starts, strides, s);
    return contiguous_copy_gpu(reversed_view, s);
  };

  array scan_input =
      reverse ? reverse_last_axis_contiguous(in_kernel) : in_kernel;

  if (!scan_input.flags().contiguous || scan_input.offset() != 0 ||
      scan_input.strides().back() != 1 ||
      !is_supported_unary_layout(scan_input)) {
    scan_input = contiguous_copy_gpu(scan_input, s);
  }

  if (scan_input.shape() != out.shape()) {
    return false;
  }

  if (scan_input.size() > std::numeric_limits<uint32_t>::max() ||
      scan_input.shape(scan_input.ndim() - 1) >
          std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  array inclusive_out(scan_input.shape(), scan_input.dtype(), nullptr, {});
  inclusive_out.set_data(allocator::malloc(inclusive_out.nbytes()));
  if (inclusive_out.size() == 0) {
    copy_gpu(inclusive_out, out, CopyType::GeneralGeneral, s);
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);

    vulkan::dispatch_cumsum_op(
        scan_input, inclusive_out, "cumsum_f32", command_buffer, s);

    array scan_result = inclusive_out;
    if (!inclusive) {
      array exclusive_out(scan_input.shape(), scan_input.dtype(), nullptr, {});
      exclusive_out.set_data(allocator::malloc(exclusive_out.nbytes()));
      vulkan::dispatch_binary_op(
          inclusive_out,
          scan_input,
          exclusive_out,
          "sub_f32_f32_f32",
          command_buffer,
          s,
          vulkan::BinaryDispatchVariant::Standard);
      scan_result = exclusive_out;
    }

    vulkan::end_command_recording(s.index);

    array restored =
        reverse ? reverse_last_axis_contiguous(scan_result) : scan_result;
    if (normalized_axis != in.ndim() - 1) {
      restored = swapaxes_in_eval(restored, normalized_axis, in.ndim() - 1);
    }

    copy_gpu(restored, out, CopyType::GeneralGeneral, s);
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

#define VULKAN_BINARY_GPU(func, op_name)                              \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    eval_binary_vulkan_or_cpu<func>(inputs, out, op_name, stream());  \
  }

#define VULKAN_GENERIC_UNARY_GPU(func, op_name)                       \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    eval_generic_unary_suffix_vulkan_or_cpu<func>(                    \
        inputs, out, op_name, stream());                              \
  }

#define VULKAN_GENERIC_UNARY_RTE_GPU(func, op_name)                   \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    eval_generic_unary_suffix_vulkan_or_cpu<func>(                    \
        inputs, out, op_name, stream(), true);                        \
  }

CPU_FALLBACK_STATE(Equal)
CPU_FALLBACK(Minimum)
CPU_FALLBACK(Maximum)
CPU_FALLBACK_STATE(RandomBits)

VULKAN_BINARY_GPU(Add, "add")
VULKAN_BINARY_GPU(Divide, "div")
VULKAN_BINARY_GPU(Subtract, "sub")
VULKAN_BINARY_GPU(Multiply, "mul")

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis] = state();
  if (try_eval_arg_reduce_vulkan(inputs, out, reduce_type, axis, stream())) {
    return;
  }
  eval_cpu_fallback<ArgReduce>(inputs, out, reduce_type, axis);
}

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axes] = state();
  if (try_eval_reduce_sum_rows_vulkan(
          inputs, out, reduce_type, axes, stream())) {
    return;
  }
  eval_cpu_fallback<Reduce>(inputs, out, reduce_type, axes);
}

void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (try_eval_softmax_vulkan(inputs, out, state(), stream())) {
    return;
  }
  eval_cpu_fallback<Softmax>(inputs, out, state());
}

VULKAN_GENERIC_UNARY_GPU(Abs, "abs")

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [start, stop, step] = state();
  if (try_eval_arange_vulkan(inputs, out, stream(), start, step)) {
    return;
  }
  eval_cpu_fallback<Arange>(inputs, out, start, stop, step);
}

VULKAN_GENERIC_UNARY_GPU(Ceil, "ceil")

void Cos::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    if (try_eval_unary_op_vulkan<Cos>(inputs, out, "cos_f32", stream())) {
      return;
    }
  }
  eval_cpu_fallback<Cos>(inputs, out);
}

VULKAN_GENERIC_UNARY_RTE_GPU(Exp, "exp")

VULKAN_GENERIC_UNARY_GPU(Floor, "floor")

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
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    if (try_eval_unary_op_vulkan<Sin>(inputs, out, "sin_f32", stream())) {
      return;
    }
  }
  eval_cpu_fallback<Sin>(inputs, out);
}

VULKAN_GENERIC_UNARY_GPU(Negative, "neg")

VULKAN_GENERIC_UNARY_GPU(Round, "round")

VULKAN_GENERIC_UNARY_GPU(Sigmoid, "sigmoid")

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

VULKAN_GENERIC_UNARY_GPU(Tanh, "tanh")

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis, reverse, inclusive] = state();
  if (try_eval_scan_cumsum_vulkan(
          inputs, out, reduce_type, axis, reverse, inclusive, stream())) {
    return;
  }
  eval_cpu_fallback<Scan>(inputs, out, reduce_type, axis, reverse, inclusive);
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
// Reduce implemented above.
// Round implemented above.
// Scan implemented above.
CPU_FALLBACK_STATE(Scatter)
CPU_FALLBACK_STATE(ScatterAxis)
CPU_FALLBACK(Select)
CPU_FALLBACK(SegmentedMM)
// Sigmoid implemented above.
CPU_FALLBACK(Sign)
// Sin implemented above.
CPU_FALLBACK(Sinh)
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
