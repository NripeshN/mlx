// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

namespace {

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
