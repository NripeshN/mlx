// Copyright © 2024 Apple Inc.

#include "mlx/backend/gpu/eval.h"
#include "mlx/primitives.h"

namespace mlx::core::gpu {

void eval(array& arr) {
  auto outputs = arr.outputs();
  {
    // Keep tracer inputs alive so they are not donated.
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }

  // Temporary conservative behavior: synchronize after each op.
  // This avoids lifetime hazards while Vulkan command scheduling matures.
  ::mlx::core::gpu::synchronize(arr.primitive().stream());
}

void finalize(Stream s) {
  ::mlx::core::gpu::synchronize(s);
}

} // namespace mlx::core::gpu
