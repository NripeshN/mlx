// Copyright © 2024 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/stream.h"

#include <cstring>

namespace mlx::core {

void copy_gpu(const array& src, array& out, CopyType ctype, const Stream& s) {
  bool donated = set_copy_output_data(src, out, ctype);
  if (donated && src.dtype() == out.dtype()) {
    // If the output has the same type as the input then there is nothing to
    // copy, just use the buffer.
    return;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_gpu_inplace(src, out, ctype, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    const Stream& s,
    std::optional<array> dynamic_i_offset,
    std::optional<array> dynamic_o_offset) {
  if (out.size() == 0) {
    return;
  }

  const bool same_dtype = in.dtype() == out.dtype();
  const bool raw_buffer_copy = same_dtype && ctype == CopyType::Vector;

  if (!raw_buffer_copy) {
    auto cpu_stream = default_stream(Device::cpu);
    copy_cpu_inplace(
        in,
        out,
        data_shape,
        i_strides,
        o_strides,
        i_offset,
        o_offset,
        ctype,
        cpu_stream,
        dynamic_i_offset,
        dynamic_o_offset);
    synchronize(cpu_stream);
    return;
  }

  // Get Vulkan command buffer
  VkCommandBuffer cmd_buffer = vulkan::begin_command_recording(s.index);

  // Get buffer handles
  auto* in_buf = static_cast<vulkan::VulkanBuffer*>(
      const_cast<void*>(static_cast<const void*>(in.buffer().ptr())));
  auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());

  if (ctype == CopyType::Vector) {
    // Simple contiguous memory copy using Vulkan command buffer
    VkBufferCopy copy_region{};
    copy_region.srcOffset =
        static_cast<VkDeviceSize>(i_offset * size_of(in.dtype()));
    copy_region.dstOffset =
        static_cast<VkDeviceSize>(o_offset * size_of(out.dtype()));
    copy_region.size = static_cast<VkDeviceSize>(out.nbytes());

    vkCmdCopyBuffer(
        cmd_buffer, in_buf->buffer, out_buf->buffer, 1, &copy_region);

  } else {
    throw std::runtime_error("Unsupported Vulkan copy type.");
  }

  vulkan::end_command_recording(s.index);
}

// Note: The simpler overload copy_gpu_inplace(in, out, ctype, s) is defined in
// mlx/backend/gpu/copy.cpp and calls the complex version implemented above.

void fill_gpu(const array& val, array& out, const Stream& s) {
  if (out.size() == 0) {
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  // For unified memory, we can directly fill on CPU
  auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());

  if (vulkan::VulkanContext::get().is_unified_memory()) {
    // Direct CPU fill for unified memory
    char* dst_ptr = static_cast<char*>(out_buf->mapped_ptr);
    const char* val_ptr = static_cast<const char*>(val.data<void>());
    size_t val_size = size_of(val.dtype());
    size_t out_size = out.nbytes();

    for (size_t i = 0; i < out_size; i += val_size) {
      std::memcpy(dst_ptr + i, val_ptr, val_size);
    }
  } else {
    // For discrete GPUs, we need to use a compute shader or staging buffer
    // TODO: Implement compute shader fill
    throw std::runtime_error("fill_gpu not yet implemented for discrete GPUs");
  }
}

void reshape_gpu(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu_inplace(
        in,
        out,
        in.shape(),
        in.strides(),
        make_contiguous_strides(in.shape()),
        0,
        0,
        CopyType::General,
        s);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s) {
  throw std::runtime_error("concatenate_gpu has no Vulkan implementation.");
}

array compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes,
    const Stream& s) {
  throw std::runtime_error(
      "compute_dynamic_offset has no Vulkan implementation.");
  return array({});
}

} // namespace mlx::core
