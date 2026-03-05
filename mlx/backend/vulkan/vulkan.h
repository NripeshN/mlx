// Copyright © 2024 Apple Inc.

#pragma once

#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "mlx/api.h"

namespace mlx::core::vulkan {

MLX_API bool is_available();
MLX_API bool is_unified_memory();
MLX_API int device_count();

class VulkanContext {
 public:
  static VulkanContext& get();

  VkInstance instance() const {
    return instance_;
  }
  VkPhysicalDevice physical_device() const {
    return physical_device_;
  }
  VkDevice device() const {
    return device_;
  }
  VkQueue compute_queue() const {
    return compute_queue_;
  }
  uint32_t compute_queue_family_index() const {
    return compute_queue_family_index_;
  }

  // Memory properties
  bool is_unified_memory() const {
    return is_unified_memory_;
  }
  VkPhysicalDeviceMemoryProperties memory_properties() const {
    return mem_properties_;
  }

  // Find memory type that supports the given properties
  uint32_t find_memory_type(
      uint32_t typeFilter,
      VkMemoryPropertyFlags properties) const;

 private:
  VulkanContext();
  ~VulkanContext();

  VulkanContext(const VulkanContext&) = delete;
  VulkanContext& operator=(const VulkanContext&) = delete;

  void init();
  void cleanup();

  VkInstance instance_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
  VkDevice device_{VK_NULL_HANDLE};
  VkQueue compute_queue_{VK_NULL_HANDLE};
  uint32_t compute_queue_family_index_{0};

  bool initialized_{false};
  bool is_unified_memory_{false};
  VkPhysicalDeviceMemoryProperties mem_properties_{};
};

} // namespace mlx::core::vulkan
