// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/allocator.h"
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/memory.h"

namespace mlx::core {

namespace allocator {

// Simple CPU allocator for Vulkan backend
// Uses std::malloc/free with size tracking at the beginning
class VulkanCompatibleAllocator : public Allocator {
 public:
  virtual Buffer malloc(size_t size) override;
  virtual void free(Buffer buffer) override;
  virtual size_t size(Buffer buffer) const override;

 private:
  std::mutex mutex_;
};

Buffer VulkanCompatibleAllocator::malloc(size_t size) {
  void* ptr = std::malloc(size + sizeof(size_t));
  if (ptr != nullptr) {
    *static_cast<size_t*>(ptr) = size;
  }
  return Buffer{ptr};
}

void VulkanCompatibleAllocator::free(Buffer buffer) {
  std::free(buffer.ptr());
}

size_t VulkanCompatibleAllocator::size(Buffer buffer) const {
  if (buffer.ptr() == nullptr) {
    return 0;
  }
  return *static_cast<size_t*>(buffer.ptr());
}

Allocator& allocator() {
  static VulkanCompatibleAllocator allocator_;
  return allocator_;
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  // Skip the size field at the beginning
  return static_cast<size_t*>(ptr_) + 1;
}

} // namespace allocator

} // namespace mlx::core

namespace mlx::core::vulkan {

// Vulkan-specific allocator for GPU buffers
// This is used explicitly when creating GPU buffers
allocator::Buffer VulkanAllocator::malloc(size_t size) {
  std::lock_guard lock(mutex_);

  if (size == 0)
    return allocator::Buffer{nullptr};

  const VulkanContext& ctx = VulkanContext::get();
  VkDevice device = ctx.device();
  const bool use_unified = ctx.is_unified_memory();

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer vk_buffer;
  if (vkCreateBuffer(device, &bufferInfo, nullptr, &vk_buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, vk_buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;

  // Choose memory properties based on unified memory support
  VkMemoryPropertyFlags memory_flags;
  if (use_unified) {
    // For unified memory (integrated GPUs), use device-local + host-visible
    memory_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  } else {
    // For discrete GPUs, use host-visible memory
    memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }

  allocInfo.memoryTypeIndex =
      ctx.find_memory_type(memRequirements.memoryTypeBits, memory_flags);

  VkDeviceMemory vk_memory;
  if (vkAllocateMemory(device, &allocInfo, nullptr, &vk_memory) != VK_SUCCESS) {
    vkDestroyBuffer(device, vk_buffer, nullptr);
    throw std::runtime_error("failed to allocate buffer memory");
  }

  vkBindBufferMemory(device, vk_buffer, vk_memory, 0);

  void* mapped_ptr;
  vkMapMemory(device, vk_memory, 0, size, 0, &mapped_ptr);

  auto* buf =
      new VulkanBuffer{mapped_ptr, vk_buffer, vk_memory, size, use_unified};
  return allocator::Buffer{buf};
}

void VulkanAllocator::free(allocator::Buffer buffer) {
  std::lock_guard lock(mutex_);

  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  if (!buf)
    return;

  VkDevice device = VulkanContext::get().device();

  vkUnmapMemory(device, buf->memory);
  vkDestroyBuffer(device, buf->buffer, nullptr);
  vkFreeMemory(device, buf->memory, nullptr);

  delete buf;
}

size_t VulkanAllocator::size(allocator::Buffer buffer) const {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  return buf ? buf->size : 0;
}

VulkanAllocator& allocator() {
  static VulkanAllocator allocator_;
  return allocator_;
}

} // namespace mlx::core::vulkan

namespace mlx::core {
size_t get_active_memory() {
  return 0;
}
size_t get_peak_memory() {
  return 0;
}
void reset_peak_memory() {}
size_t set_memory_limit(size_t limit) {
  return limit;
}
size_t get_memory_limit() {
  return 0;
}
size_t get_cache_memory() {
  return 0;
}
size_t set_cache_limit(size_t limit) {
  return limit;
}
void clear_cache() {}
size_t set_wired_limit(size_t limit) {
  return limit;
}
} // namespace mlx::core
