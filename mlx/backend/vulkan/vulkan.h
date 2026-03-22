// Copyright © 2024 Apple Inc.

#pragma once

#include <atomic>
#include <memory>
#include <vector>

// Use C++ Vulkan API instead of C API
#include <vulkan/vulkan.hpp>

#include "mlx/api.h"

namespace mlx::core::vulkan {

MLX_API bool is_available();
MLX_API bool is_unified_memory();
MLX_API int device_count();

class VulkanContext {
 public:
  static VulkanContext& get();

  // C++ Vulkan API accessors
  const vk::Instance& instance() const {
    return instance_;
  }
  const vk::PhysicalDevice& physical_device() const {
    return physical_device_;
  }
  const vk::Device& device() const {
    return device_;
  }
  const vk::Queue& compute_queue() const {
    return compute_queue_;
  }
  const vk::Queue& transfer_queue() const {
    return transfer_queue_;
  }
  uint32_t compute_queue_family_index() const {
    return compute_queue_family_index_;
  }
  uint32_t transfer_queue_family_index() const {
    return transfer_queue_family_index_;
  }
  bool has_separate_transfer_queue() const {
    return has_separate_transfer_queue_;
  }

  // Timeline semaphore accessors
  vk::Semaphore timeline_semaphore() const {
    return timeline_semaphore_;
  }
  uint64_t current_timeline_value() const {
    return timeline_value_.load();
  }
  uint64_t increment_timeline();

  // Memory properties
  bool is_unified_memory() const {
    return is_unified_memory_;
  }
  const vk::PhysicalDeviceMemoryProperties& memory_properties() const {
    return mem_properties_;
  }
  bool shader_float16_supported() const {
    return shader_float16_supported_;
  }
  bool subgroup_size_control_supported() const {
    return subgroup_size_control_supported_;
  }
  bool subgroup_require_full_support() const {
    return subgroup_require_full_support_;
  }
  uint32_t subgroup_min_size() const {
    return subgroup_min_size_;
  }
  uint32_t subgroup_max_size() const {
    return subgroup_max_size_;
  }
  bool pipeline_robustness_supported() const {
    return pipeline_robustness_supported_;
  }
  bool coopmat_flash_attention_f32acc_supported() const {
    return coopmat_flash_attention_f32acc_supported_;
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

  // C++ Vulkan API objects (RAII)
  vk::Instance instance_;
  vk::PhysicalDevice physical_device_;
  vk::Device device_;
  vk::Queue compute_queue_;
  vk::Queue transfer_queue_;
  uint32_t compute_queue_family_index_{0};
  uint32_t transfer_queue_family_index_{0};
  bool has_separate_transfer_queue_{false};

  // Timeline semaphore for cross-queue synchronization
  vk::Semaphore timeline_semaphore_;
  std::atomic<uint64_t> timeline_value_{0};

  bool initialized_{false};
  bool is_unified_memory_{false};
  bool shader_float16_supported_{false};
  bool subgroup_size_control_supported_{false};
  bool subgroup_require_full_support_{false};
  uint32_t subgroup_min_size_{0};
  uint32_t subgroup_max_size_{0};
  bool pipeline_robustness_supported_{false};
  bool coopmat_flash_attention_f32acc_supported_{false};
  vk::PhysicalDeviceMemoryProperties mem_properties_{};
};

// Helper function to check Vulkan result and throw on error
inline void throw_if_vk_error(VkResult result, const std::string& context) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        context + " (VkResult=" + std::to_string(result) + ").");
  }
}

} // namespace mlx::core::vulkan
