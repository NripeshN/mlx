// Copyright © 2024 Apple Inc.

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/primitives.h"

namespace mlx::core::gpu {

void eval(array& arr) {
  auto outputs = arr.outputs();
  auto s = arr.primitive().stream();

  vulkan::record_primitive_for_stream(s, arr.primitive().name());
  vulkan::begin_primitive_tracking(s, arr.inputs(), outputs);
  {
    // Keep tracer inputs alive so they are not donated.
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }

  vulkan::end_primitive_tracking(s, arr.inputs(), outputs);

  auto donated_data = arr.data_shared_ptr();
  for (const auto& in : arr.inputs()) {
    if (in.data_shared_ptr() != donated_data) {
      vulkan::retain_array_for_stream(s, in);
    }
  }
  for (const auto& sibling : arr.siblings()) {
    vulkan::retain_array_for_stream(s, sibling);
  }
  for (const auto& out : outputs) {
    vulkan::retain_array_for_stream(s, out);
  }
}

void finalize(Stream s) {
  vulkan::finalize_stream(s);
}

} // namespace mlx::core::gpu
