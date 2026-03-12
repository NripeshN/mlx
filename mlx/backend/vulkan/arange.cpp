// Copyright © 2024 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/vulkan.h"

#include <vector>

namespace mlx::core {

namespace {

template <typename T>
void fill_arange_like_cpu(array& out, Stream s, double start, double step) {
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  T value = static_cast<T>(start);
  const T next = static_cast<T>(start + step);
  const T step_size = next - value;

  if (vulkan::VulkanContext::get().is_unified_memory()) {
    auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());
    auto* dst = static_cast<T*>(out_buf->mapped_ptr);
    for (size_t i = 0; i < out.size(); ++i) {
      dst[i] = value;
      value += step_size;
    }
    return;
  }

  std::vector<T> host_values(out.size());
  for (size_t i = 0; i < out.size(); ++i) {
    host_values[i] = value;
    value += step_size;
  }

  auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());
  vulkan::enqueue_owned_staging_upload(
      s,
      host_values.data(),
      host_values.size() * sizeof(T),
      out_buf->buffer,
      out.offset());
  vulkan::retain_array_for_stream(s, out);
}

bool try_eval_arange_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s,
    double start,
    double step) {
  if (!inputs.empty() || !is_supported_generic_unary_layout(out)) {
    return false;
  }

  switch (out.dtype()) {
    case bool_:
      return false;
    case uint8:
      fill_arange_like_cpu<uint8_t>(out, s, start, step);
      return true;
    case uint16:
      fill_arange_like_cpu<uint16_t>(out, s, start, step);
      return true;
    case uint32:
      fill_arange_like_cpu<uint32_t>(out, s, start, step);
      return true;
    case uint64:
      fill_arange_like_cpu<uint64_t>(out, s, start, step);
      return true;
    case int8:
      fill_arange_like_cpu<int8_t>(out, s, start, step);
      return true;
    case int16:
      fill_arange_like_cpu<int16_t>(out, s, start, step);
      return true;
    case int32:
      fill_arange_like_cpu<int32_t>(out, s, start, step);
      return true;
    case int64:
      fill_arange_like_cpu<int64_t>(out, s, start, step);
      return true;
    case float16:
      fill_arange_like_cpu<float16_t>(out, s, start, step);
      return true;
    case float64:
      fill_arange_like_cpu<double>(out, s, start, step);
      return true;
    case bfloat16:
      fill_arange_like_cpu<bfloat16_t>(out, s, start, step);
      return true;
    case complex64:
      fill_arange_like_cpu<complex64_t>(out, s, start, step);
      return true;
    case float32:
      break;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_arange_op(
        out,
        vulkan::StaticShaderId::arange_f32,
        command_buffer,
        s,
        static_cast<float>(start),
        static_cast<float>(step));
    vulkan::end_command_recording(s.index);

    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "arange_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [start, stop, step] = state();
  if (!try_eval_arange_vulkan(inputs, out, stream(), start, step)) {
    throw std::runtime_error(
        "Arange operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
