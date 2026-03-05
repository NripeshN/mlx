// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/vulkan/allocator.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/memory.h"

namespace mlx::core {

namespace allocator {

Allocator& allocator() {
  return vulkan::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<vulkan::VulkanBuffer*>(ptr_)->mapped_ptr;
}

} // namespace allocator

namespace vulkan {

namespace {

uint32_t find_memory_type_index(
    const VulkanContext& ctx,
    uint32_t type_filter,
    const std::vector<VkMemoryPropertyFlags>& preferred_flags) {
  for (auto flags : preferred_flags) {
    try {
      return ctx.find_memory_type(type_filter, flags);
    } catch (const std::runtime_error&) {
    }
  }
  throw std::runtime_error("[vulkan::malloc] No suitable memory type found.");
}

} // namespace

VulkanAllocator::VulkanAllocator() {
  const auto& info = gpu::device_info(0);
  auto memory_size = std::get<size_t>(info.at("memory_size"));
  auto max_rec_size =
      std::get<size_t>(info.at("max_recommended_working_set_size"));

  if (memory_size == 0) {
    memory_size = 1UL << 33;
  }
  if (max_rec_size == 0) {
    max_rec_size = memory_size;
  }

  resource_limit_ = std::get<size_t>(info.at("resource_limit"));
  if (resource_limit_ == 0) {
    resource_limit_ = std::numeric_limits<size_t>::max();
  }

  block_limit_ = std::min(
      static_cast<size_t>(1.5 * static_cast<double>(max_rec_size)),
      static_cast<size_t>(0.95 * static_cast<double>(memory_size)));
  gc_limit_ = block_limit_;
  max_pool_size_ = block_limit_;
}

size_t VulkanAllocator::set_cache_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, max_pool_size_);
  return limit;
}

size_t VulkanAllocator::set_memory_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, block_limit_);
  gc_limit_ = std::min(gc_limit_, block_limit_);
  max_pool_size_ = std::min(max_pool_size_, block_limit_);
  return limit;
}

size_t VulkanAllocator::get_memory_limit() const {
  return block_limit_;
}

size_t VulkanAllocator::set_wired_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, wired_limit_);
  return limit;
}

void VulkanAllocator::clear_cache() {
  // No cache in Vulkan allocator yet.
}

Buffer VulkanAllocator::malloc(size_t size) {
  if (size == 0) {
    return Buffer{nullptr};
  }

  {
    std::unique_lock lk(mutex_);
    if (num_resources_ >= resource_limit_) {
      std::ostringstream msg;
      msg << "[vulkan::malloc] Resource limit (" << resource_limit_
          << ") exceeded.";
      throw std::runtime_error(msg.str());
    }
  }

  auto& ctx = VulkanContext::get();
  auto device = ctx.device();

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer vk_buffer = VK_NULL_HANDLE;
  if (vkCreateBuffer(device, &buffer_info, nullptr, &vk_buffer) != VK_SUCCESS) {
    throw std::runtime_error("[vulkan::malloc] Failed to create buffer.");
  }

  VkMemoryRequirements mem_requirements{};
  vkGetBufferMemoryRequirements(device, vk_buffer, &mem_requirements);
  const size_t allocation_size = static_cast<size_t>(mem_requirements.size);

  {
    std::unique_lock lk(mutex_);
    if (num_resources_ >= resource_limit_) {
      vkDestroyBuffer(device, vk_buffer, nullptr);
      std::ostringstream msg;
      msg << "[vulkan::malloc] Resource limit (" << resource_limit_
          << ") exceeded.";
      throw std::runtime_error(msg.str());
    }

    const bool exceeds_limit = active_memory_ > block_limit_ ||
        allocation_size > (block_limit_ - active_memory_);
    if (exceeds_limit) {
      vkDestroyBuffer(device, vk_buffer, nullptr);
      std::ostringstream msg;
      msg << "[vulkan::malloc] Memory limit (" << block_limit_
          << " bytes) exceeded while allocating " << size << " bytes ("
          << allocation_size << " bytes with alignment).";
      throw std::runtime_error(msg.str());
    }
  }

  std::vector<VkMemoryPropertyFlags> preferred_memory_types;
  if (ctx.is_unified_memory()) {
    preferred_memory_types = {
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
  } else {
    preferred_memory_types = {
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT};
  }

  uint32_t memory_type_index = 0;
  try {
    memory_type_index = find_memory_type_index(
        ctx, mem_requirements.memoryTypeBits, preferred_memory_types);
  } catch (...) {
    vkDestroyBuffer(device, vk_buffer, nullptr);
    throw;
  }

  auto mem_props = ctx.memory_properties();
  const auto memory_flags =
      mem_props.memoryTypes[memory_type_index].propertyFlags;

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex = memory_type_index;

  VkDeviceMemory vk_memory = VK_NULL_HANDLE;
  if (vkAllocateMemory(device, &alloc_info, nullptr, &vk_memory) !=
      VK_SUCCESS) {
    vkDestroyBuffer(device, vk_buffer, nullptr);
    throw std::runtime_error(
        "[vulkan::malloc] Failed to allocate device memory.");
  }

  if (vkBindBufferMemory(device, vk_buffer, vk_memory, 0) != VK_SUCCESS) {
    vkFreeMemory(device, vk_memory, nullptr);
    vkDestroyBuffer(device, vk_buffer, nullptr);
    throw std::runtime_error("[vulkan::malloc] Failed to bind buffer memory.");
  }

  void* mapped_ptr = nullptr;
  if (memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    if (vkMapMemory(device, vk_memory, 0, VK_WHOLE_SIZE, 0, &mapped_ptr) !=
        VK_SUCCESS) {
      vkFreeMemory(device, vk_memory, nullptr);
      vkDestroyBuffer(device, vk_buffer, nullptr);
      throw std::runtime_error("[vulkan::malloc] Failed to map memory.");
    }
  }

  auto* buf = new VulkanBuffer{
      mapped_ptr, vk_buffer, vk_memory, size, allocation_size, memory_flags};

  {
    std::unique_lock lk(mutex_);
    active_memory_ += buf->allocation_size;
    peak_memory_ = std::max(peak_memory_, active_memory_);
    num_resources_++;
    live_buffers_.insert(buf);
  }

  return Buffer{static_cast<void*>(buf)};
}

void VulkanAllocator::free(Buffer buffer) {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }

  {
    std::unique_lock lk(mutex_);
    if (!live_buffers_.contains(buf)) {
      return;
    }
    live_buffers_.erase(buf);
    active_memory_ -= std::min(active_memory_, buf->allocation_size);
    num_resources_ -= std::min<size_t>(1, num_resources_);
  }

  auto device = VulkanContext::get().device();

  // A buffer must be destroyed before freeing its bound memory.
  vkDestroyBuffer(device, buf->buffer, nullptr);
  vkFreeMemory(device, buf->memory, nullptr);

  delete buf;
}

size_t VulkanAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  return buf ? buf->size : 0;
}

Buffer VulkanAllocator::make_buffer(void*, size_t) {
  // Vulkan no-copy host-pointer import requires optional extensions and
  // additional device setup that is not enabled yet.
  return Buffer{nullptr};
}

void VulkanAllocator::release(Buffer) {
  // No-op because make_buffer currently returns nullptr.
}

VulkanAllocator& allocator() {
  static auto* allocator_ = new VulkanAllocator();
  return *allocator_;
}

} // namespace vulkan

size_t set_cache_limit(size_t limit) {
  return vulkan::allocator().set_cache_limit(limit);
}

size_t set_memory_limit(size_t limit) {
  return vulkan::allocator().set_memory_limit(limit);
}

size_t get_memory_limit() {
  return vulkan::allocator().get_memory_limit();
}

size_t set_wired_limit(size_t limit) {
  return vulkan::allocator().set_wired_limit(limit);
}

size_t get_active_memory() {
  return vulkan::allocator().get_active_memory();
}

size_t get_peak_memory() {
  return vulkan::allocator().get_peak_memory();
}

void reset_peak_memory() {
  vulkan::allocator().reset_peak_memory();
}

size_t get_cache_memory() {
  return vulkan::allocator().get_cache_memory();
}

void clear_cache() {
  vulkan::allocator().clear_cache();
}

} // namespace mlx::core
