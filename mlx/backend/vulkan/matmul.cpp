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
  auto cpu_stream = default_stream(Device::cpu);
  Matmul cpu_matmul(cpu_stream);
  cpu_matmul.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [alpha, beta] = state();
  auto cpu_stream = default_stream(Device::cpu);
  AddMM cpu_addmm(cpu_stream, alpha, beta);
  cpu_addmm.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto block_size = state();
  auto cpu_stream = default_stream(Device::cpu);
  BlockMaskedMM cpu_block_masked_mm(cpu_stream, block_size);
  cpu_block_masked_mm.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

} // namespace mlx::core
