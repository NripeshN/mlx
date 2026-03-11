// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/primitives_utils.h"

#include <algorithm>

namespace mlx::core {

namespace {

bool is_vulkan_compare_dtype(Dtype dtype) {
  switch (dtype) {
    case float16:
    case float32:
    case bfloat16:
    case int32:
    case int64:
    case uint32:
    case uint64:
      return true;
    default:
      return false;
  }
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

  const bool low_precision_div = std::string_view(op_name) == "div" &&
      (a.dtype() == float16 || b.dtype() == float16 || out.dtype() == float16 ||
       a.dtype() == bfloat16 || b.dtype() == bfloat16 ||
       out.dtype() == bfloat16);
  const bool use_f32_staging_io = low_precision_div || a.dtype() == bfloat16 ||
      b.dtype() == bfloat16 || out.dtype() == bfloat16;
  if (use_f32_staging_io) {
    array a_f32(a.shape(), float32, nullptr, {});
    array b_f32(b.shape(), float32, nullptr, {});
    copy_gpu(a, a_f32, CopyType::General, s);
    copy_gpu(b, b_f32, CopyType::General, s);
    a = a_f32;
    b = b_f32;
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

  const bool staged_output =
      use_f32_staging_io || !is_supported_elementwise_layout(out);
  array out_work = staged_output
      ? array(
            out.shape(),
            use_f32_staging_io ? float32 : out.dtype(),
            nullptr,
            {})
      : out;

  auto suffix_a = dtype_suffix(a.dtype());
  auto suffix_b = dtype_suffix(b.dtype());
  auto suffix_out = dtype_suffix(out_work.dtype());
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
    auto dispatch_variant = binary_dispatch_variant<Primitive>();
    if constexpr (std::is_same_v<Primitive, Add>) {
      if (use_f32_staging_io) {
        dispatch_variant = vulkan::BinaryDispatchVariant::Standard;
      }
    }
    vulkan::dispatch_binary_op(
        a, b, out_work, shader_name, command_buffer, s, dispatch_variant);
    vulkan::end_command_recording(s.index);
    if (staged_output || use_f32_staging_io) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "binary_dispatch_failed op=" << op_name << " reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

template <typename Primitive>
void eval_binary_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const char* op_name,
    Stream s) {
  if (!try_eval_binary_op_vulkan<Primitive>(inputs, out, op_name, s)) {
    throw std::runtime_error(
        std::string("Binary operation ") + op_name +
        " failed on Vulkan (unsupported dtype or layout).");
  }
}

bool try_eval_greater_equal_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 2 || out.dtype() != bool_) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  if (!is_vulkan_compare_dtype(a.dtype()) ||
      !is_vulkan_compare_dtype(b.dtype())) {
    return false;
  }
  if (a.dtype() != b.dtype() &&
      !(is_vulkan_float_dtype(a.dtype()) && is_vulkan_float_dtype(b.dtype()))) {
    return false;
  }
  if (a.shape() != out.shape() || b.shape() != out.shape()) {
    return false;
  }

  const bool use_f32_staging_io =
      a.dtype() == bfloat16 || b.dtype() == bfloat16;
  if (use_f32_staging_io) {
    array a_f32(a.shape(), float32, nullptr, {});
    array b_f32(b.shape(), float32, nullptr, {});
    copy_gpu(a, a_f32, CopyType::General, s);
    copy_gpu(b, b_f32, CopyType::General, s);
    a = a_f32;
    b = b_f32;
  }

  if (!is_supported_elementwise_layout(a)) {
    a = contiguous_copy_gpu(a, s);
  }
  if (!is_supported_elementwise_layout(b)) {
    b = contiguous_copy_gpu(b, s);
  }

  auto suffix_a = dtype_suffix(a.dtype());
  auto suffix_b = dtype_suffix(b.dtype());
  if (suffix_a.empty() || suffix_b.empty()) {
    return false;
  }

  array out_u8(out.shape(), uint8, nullptr, {});
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out_u8, bopt, [&](size_t n) {
    return vulkan::allocator().malloc(n);
  });
  if (!is_supported_elementwise_layout(out_u8)) {
    return false;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_binary_op(
        a,
        b,
        out_u8,
        std::string("greater_equal_") + suffix_a + "_" + suffix_b + "_u8",
        command_buffer,
        s,
        vulkan::BinaryDispatchVariant::Standard);
    vulkan::end_command_recording(s.index);
    out.copy_shared_buffer(
        out_u8,
        out_u8.strides(),
        out_u8.flags(),
        out_u8.data_size(),
        out_u8.offset());
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "binary_dispatch_failed op=greater_equal reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

#define VULKAN_BINARY_GPU(func, op_name)                              \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    eval_binary_vulkan<func>(inputs, out, op_name, stream());         \
  }

VULKAN_BINARY_GPU(Add, "add")
VULKAN_BINARY_GPU(Minimum, "minimum")
VULKAN_BINARY_GPU(Maximum, "maximum")
VULKAN_BINARY_GPU(Divide, "div")
VULKAN_BINARY_GPU(Subtract, "sub")
VULKAN_BINARY_GPU(Multiply, "mul")

void GreaterEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_greater_equal_vulkan(inputs, out, stream())) {
    throw std::runtime_error(
        "GreaterEqual operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
