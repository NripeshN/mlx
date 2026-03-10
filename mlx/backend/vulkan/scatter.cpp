// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("Scatter has no Vulkan implementation.");
}

} // namespace mlx::core
