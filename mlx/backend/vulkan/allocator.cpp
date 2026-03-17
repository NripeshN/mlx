// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/vulkan/allocator.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/memory.h"

namespace mlx::core {

namespace allocator {

namespace {

using namespace vk;

bool trace_cpu_access_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_TRACE_CPU_ACCESS");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

} // namespace

Allocator& allocator() {
  return vulkan::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  auto* buf = static_cast<vulkan::VulkanBuffer*>(ptr_);
  if (trace_cpu_access_enabled()) {
    // Convert memory flags to uint32_t for printing
    uint32_t flags = static_cast<uint32_t>(buf->memory_flags);
    std::cerr << "[vulkan-cpu-access] raw_ptr buffer=" << buf->buffer
              << " mapped=" << buf->mapped_ptr << " size=" << buf->size
              << " alloc=" << buf->allocation_size << " flags=0x" << std::hex
              << flags << std::dec << "\n";
  }
  vulkan::synchronize_all();
  return buf->mapped_ptr;
}

} // namespace allocator

namespace vulkan {

namespace {

uint32_t find_memory_type_index(
    const VulkanContext& ctx,
    uint32_t type_filter,
    const std::vector<vk::MemoryPropertyFlags>& preferred_flags) {
  for (auto flags : preferred_flags) {
    try {
      // Convert vk::MemoryPropertyFlags to VkMemoryPropertyFlags
      // This is safe since they're the same underlying type
      VkMemoryPropertyFlags vk_flags = static_cast<VkMemoryPropertyFlags>(flags);
      return ctx.find_memory_type(type_filter, vk_flags);
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
  auto vk_device = ctx.device();

  // Use C++ Vulkan API for buffer creation
  vk::BufferCreateInfo buffer_info(vk::BufferCreateFlags(), size);
  buffer_info.usage = vk::BufferUsageFlagBits::eStorageBuffer | 
                      vk::BufferUsageFlagBits::eTransferSrc | 
                      vk::BufferUsageFlagBits::eTransferDst;
  buffer_info.sharingMode = vk::SharingMode::eExclusive;

  vk::Buffer vk_buffer = vk_device.createBuffer(buffer_info);

  vk::MemoryRequirements mem_requirements = vk_device.getBufferMemoryRequirements(vk_buffer);
  const size_t allocation_size = static_cast<size_t>(mem_requirements.size);

  {
    std::unique_lock lk(mutex_);
    if (num_resources_ >= resource_limit_) {
      vk_device.destroyBuffer(vk_buffer);
      std::ostringstream msg;
      msg << "[vulkan::malloc] Resource limit (" << resource_limit_
          << ") exceeded.";
      throw std::runtime_error(msg.str());
    }
  }

  std::vector<vk::MemoryPropertyFlags> preferred_memory_types;
  if (ctx.is_unified_memory()) {
    preferred_memory_types = {
        vk::MemoryPropertyFlagBits::eDeviceLocal |
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent};
  } else {
    preferred_memory_types = {
        vk::MemoryPropertyFlagBits::eDeviceLocal |
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent |
            vk::MemoryPropertyFlagBits::eHostCached};
  }

  uint32_t memory_type_index = 0;
  try {
    memory_type_index = find_memory_type_index(
        ctx, mem_requirements.memoryTypeBits, preferred_memory_types);
  } catch (...) {
    vk_device.destroyBuffer(vk_buffer);
    throw;
  }

  auto mem_props = ctx.memory_properties();
  const auto memory_flags =
      mem_props.memoryTypes[memory_type_index].propertyFlags;

  // Use C++ Vulkan API for memory allocation
  vk::MemoryAllocateInfo alloc_info(mem_requirements.size, memory_type_index);
  vk::DeviceMemory vk_memory = vk_device.allocateMemory(alloc_info);

  // Bind memory - C++ API doesn't have this method, so use C function
  if (vkBindBufferMemory(vk_device, vk_buffer, vk_memory, 0) != VK_SUCCESS) {
    vk_device.freeMemory(vk_memory);
    vk_device.destroyBuffer(vk_buffer);
    throw std::runtime_error("[vulkan::malloc] Failed to bind buffer memory.");
  }

  void* mapped_ptr = nullptr;
  if (memory_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
    // Use C++ API to map memory
    mapped_ptr = vk_device.mapMemory(vk_memory, 0, VK_WHOLE_SIZE);
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
  // Use C-style function to get raw device handle
  auto vk_device = VulkanContext::get().device();
  
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }

  {
    std::unique_lock lk(mutex_);
    // Convert to unordered_set::find or use contains if C++17
    if (live_buffers_.find(buf) == live_buffers_.end()) {
      return;
    }
    live_buffers_.erase(buf);
    active_memory_ -= std::min(active_memory_, buf->allocation_size);
    num_resources_ -= std::min<size_t>(1, num_resources_);
  }

  // Use C++ Vulkan API for cleanup
  vk_device.destroyBuffer(buf->buffer);
  vk_device.freeMemory(buf->memory);

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
