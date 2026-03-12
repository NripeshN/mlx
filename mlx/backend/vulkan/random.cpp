// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  array keys = inputs[0];
  size_t num_keys = keys.size() / 2;

  if (num_keys == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }

  size_t elems_per_key = out.size() / num_keys;
  size_t bytes_per_key = out.itemsize() * elems_per_key;
  out.set_data(allocator::malloc(out.nbytes()));

  // For now, support only uint32 keys/output on Vulkan.
  if (keys.dtype() != uint32 || out.dtype() != uint32) {
    throw std::runtime_error(
        "RandomBits failed on Vulkan (only uint32 keys/output supported).");
  }

  // Shader expects packed key layout.
  if (!keys.flags().contiguous || keys.offset() != 0 ||
      keys.strides().back() != 1) {
    keys = contiguous_copy_gpu(keys, stream());
  }

  // Calculate dispatch grid
  uint32_t out_skip = (static_cast<uint32_t>(bytes_per_key) + 4 - 1) / 4;
  uint32_t half_size = out_skip / 2;
  bool odd = (out_skip % 2) != 0;
  constexpr uint32_t kRandomBitsLocalSizeX = 256;

  try {
    auto cmd_buffer = vulkan::begin_command_recording(stream().index);

    // Set up push constants
    vulkan::RandomBitsPushConstants push_constants;
    push_constants.num_keys = static_cast<uint32_t>(num_keys);
    push_constants.bytes_per_key = static_cast<uint32_t>(bytes_per_key);
    push_constants.odd = odd ? 1u : 0u;
    push_constants.out_skip = out_skip;

    // Dispatch the kernel
    std::array<uint32_t, 3> grid = {
        (static_cast<uint32_t>(num_keys) + kRandomBitsLocalSizeX - 1) /
            kRandomBitsLocalSizeX,
        half_size + (odd ? 1u : 0u),
        1};
    vulkan::dispatch_random_bits_op(
        keys,
        out,
        vulkan::StaticShaderId::random_bits_f32,
        cmd_buffer,
        stream(),
        push_constants,
        grid);

    vulkan::end_command_recording(stream().index);
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(
        std::string("RandomBits failed on Vulkan: ") + e.what());
  }
}

} // namespace mlx::core
