// Copyright © 2024 Apple Inc.

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/primitives.h"

namespace mlx::core::gpu {

void eval(array& arr) {
  auto outputs = arr.outputs();
  auto s = arr.primitive().stream();

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

  for (const auto& in : arr.inputs()) {
    vulkan::retain_array_for_stream(s, in);
  }
  for (const auto& out : outputs) {
    vulkan::retain_array_for_stream(s, out);
  }
}

void finalize(Stream s) {
  ::mlx::core::gpu::synchronize(s);
}

} // namespace mlx::core::gpu
