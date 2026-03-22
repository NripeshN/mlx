// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <atomic>
#include <mutex>
#include <unordered_set>

// Use C++ Vulkan API instead of C API
#include <vulkan/vulkan.hpp>

#include "mlx/allocator.h"

namespace mlx::core::vulkan {

using allocator::Buffer;

struct VulkanBuffer {
  enum class QueueAffinity : uint32_t {
    None = 0,
    Compute = 1,
    Transfer = 2,
  };

  void* mapped_ptr{nullptr};
  // Use C++ Vulkan API types
  vk::Buffer buffer;
  vk::DeviceMemory memory;
  size_t size{0};
  size_t allocation_size{0};
  vk::MemoryPropertyFlags memory_flags{};
  std::mutex queue_affinity_mutex;
  vk::Semaphore last_semaphore;
  uint64_t last_timeline_value{0};
  QueueAffinity queue_affinity{QueueAffinity::None};

  // Keep vk::Buffer and vk::DeviceMemory accessible
  operator vk::Buffer() const {
    return buffer;
  }
  operator vk::DeviceMemory() const {
    return memory;
  }
};

class VulkanAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;
  Buffer make_buffer(void* ptr, size_t size) override;
  void release(Buffer buffer) override;

  size_t get_active_memory() const {
    return active_memory_;
  }
  size_t get_peak_memory() const {
    return peak_memory_;
  }
  void reset_peak_memory() {
    std::unique_lock lk(mutex_);
    peak_memory_ = 0;
  }
  size_t get_cache_memory() const {
    return 0;
  }
  size_t set_cache_limit(size_t limit);
  size_t set_memory_limit(size_t limit);
  size_t get_memory_limit() const;
  size_t set_wired_limit(size_t limit);
  void clear_cache();
  bool owns(Buffer buffer) const;

 private:
  VulkanAllocator();
  ~VulkanAllocator() = default;
  friend VulkanAllocator& allocator();

  size_t block_limit_{0};
  size_t gc_limit_{0};
  size_t active_memory_{0};
  size_t peak_memory_{0};
  size_t max_pool_size_{0};
  size_t wired_limit_{0};
  size_t num_resources_{0};
  size_t resource_limit_{0};
  std::unordered_set<VulkanBuffer*> live_buffers_;

  mutable std::mutex mutex_;
};

VulkanAllocator& allocator();
bool is_vulkan_buffer(Buffer buffer);

} // namespace mlx::core::vulkan
