// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/primitives.h"

namespace mlx::core {

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  // For now, fall back to CPU implementation
  // TODO: Implement Vulkan matmul using mul_mm.comp shaders
  eval_cpu(inputs, out);
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  // For now, fall back to CPU implementation
  // TODO: Implement Vulkan AddMM using mul_mm.comp shaders
  eval_cpu(inputs, out);
}

void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  // For now, fall back to CPU implementation
  eval_cpu(inputs, out);
}

} // namespace mlx::core
