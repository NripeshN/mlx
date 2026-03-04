// Copyright © 2024 Apple Inc.

#pragma once

#include <vulkan/vulkan.h>
#include <mutex>
#include "mlx/allocator.h"

namespace mlx::core::vulkan {

struct VulkanBuffer {
  void* mapped_ptr{nullptr};
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory memory{VK_NULL_HANDLE};
  size_t size{0};
  bool is_unified{false};
};

class VulkanAllocator : public allocator::Allocator {
 public:
  VulkanAllocator() = default;
  ~VulkanAllocator() = default;

  allocator::Buffer malloc(size_t size) override;
  void free(allocator::Buffer buffer) override;
  size_t size(allocator::Buffer buffer) const override;

 private:
  std::mutex mutex_;
};

VulkanAllocator& allocator();

} // namespace mlx::core::vulkan