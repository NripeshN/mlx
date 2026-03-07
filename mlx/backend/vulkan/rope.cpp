// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core {

namespace fast {

bool RoPE::use_fallback(Stream s) {
  trace_use_fallback("RoPE", s, "no Vulkan implementation");
  return true;
}

void RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("RoPE has no Vulkan implementation.");
}

} // namespace fast

} // namespace mlx::core
