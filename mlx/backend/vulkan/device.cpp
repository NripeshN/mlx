// Copyright © 2024 Apple Inc.

#include "mlx/device.h"

#include <vulkan/vulkan.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/stream.h"

namespace mlx::core::vulkan {

namespace {

void throw_if_vk_error(VkResult result, const std::string& context) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        context + " (VkResult=" + std::to_string(result) + ").");
  }
}

} // namespace

// Stream data structure for Vulkan
struct StreamData {
  VkCommandPool command_pool{VK_NULL_HANDLE};
  VkCommandBuffer command_buffer{VK_NULL_HANDLE};
  VkFence fence{VK_NULL_HANDLE};
  bool recording{false};
  bool has_pending_work{false};
  int stream_index{0};
};

class VulkanDevice {
 public:
  static VulkanDevice& get() {
    static auto* device = new VulkanDevice();
    return *device;
  }

  void ensure_stream(int index) {
    (void)get_stream(index);
  }

  StreamData* get_stream(int index) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = streams_.find(index);
    if (it == streams_.end()) {
      auto stream = create_stream(index);
      auto [inserted, _] = streams_.emplace(index, std::move(stream));
      it = inserted;
    }
    return it->second.get();
  }

  void synchronize(Stream s) {
    auto* stream = get_stream(s.index);
    if (stream->recording) {
      submit_commands(stream);
    }

    if (!stream->has_pending_work) {
      return;
    }

    VkDevice device = VulkanContext::get().device();
    throw_if_vk_error(
        vkWaitForFences(device, 1, &stream->fence, VK_TRUE, UINT64_MAX),
        "[vulkan::synchronize] Failed waiting for stream fence");
    throw_if_vk_error(
        vkResetFences(device, 1, &stream->fence),
        "[vulkan::synchronize] Failed resetting stream fence");
    stream->has_pending_work = false;
  }

  void synchronize() {
    throw_if_vk_error(
        vkQueueWaitIdle(VulkanContext::get().compute_queue()),
        "[vulkan::synchronize] Failed waiting for compute queue idle");
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [_, stream] : streams_) {
      stream->recording = false;
      stream->has_pending_work = false;
    }
  }

  VkCommandBuffer begin_recording(int stream_index) {
    auto* stream = get_stream(stream_index);

    if (!stream->recording) {
      VkDevice device = VulkanContext::get().device();

      if (stream->has_pending_work) {
        throw_if_vk_error(
            vkWaitForFences(device, 1, &stream->fence, VK_TRUE, UINT64_MAX),
            "[vulkan::begin_recording] Failed waiting for stream fence");
        throw_if_vk_error(
            vkResetFences(device, 1, &stream->fence),
            "[vulkan::begin_recording] Failed resetting stream fence");
        stream->has_pending_work = false;
      }

      // Reset command pool to allow reuse
      throw_if_vk_error(
          vkResetCommandPool(device, stream->command_pool, 0),
          "[vulkan::begin_recording] Failed resetting command pool");

      // Begin recording
      VkCommandBufferBeginInfo beginInfo{};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

      throw_if_vk_error(
          vkBeginCommandBuffer(stream->command_buffer, &beginInfo),
          "[vulkan::begin_recording] Failed beginning command buffer");
      stream->recording = true;
    }

    return stream->command_buffer;
  }

  void end_recording(int stream_index) {
    auto* stream = get_stream(stream_index);
    if (stream->recording) {
      submit_commands(stream);
    }
  }

 private:
  VulkanDevice() = default;

  ~VulkanDevice() {
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    try {
      auto& ctx = VulkanContext::get();
      device = ctx.device();
      queue = ctx.compute_queue();
    } catch (...) {
      return;
    }

    if (queue != VK_NULL_HANDLE) {
      vkQueueWaitIdle(queue);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [_, stream] : streams_) {
      if (stream->command_buffer != VK_NULL_HANDLE &&
          stream->command_pool != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(
            device, stream->command_pool, 1, &stream->command_buffer);
        stream->command_buffer = VK_NULL_HANDLE;
      }
      if (stream->fence != VK_NULL_HANDLE) {
        vkDestroyFence(device, stream->fence, nullptr);
        stream->fence = VK_NULL_HANDLE;
      }
      if (stream->command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, stream->command_pool, nullptr);
        stream->command_pool = VK_NULL_HANDLE;
      }
    }
  }

  std::unique_ptr<StreamData> create_stream(int index) {
    VkDevice device = VulkanContext::get().device();
    uint32_t queue_family = VulkanContext::get().compute_queue_family_index();

    auto stream = std::make_unique<StreamData>();
    stream->stream_index = index;

    // Create command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queue_family;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(
            device, &poolInfo, nullptr, &stream->command_pool) != VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool");
    }

    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = stream->command_pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &allocInfo, &stream->command_buffer) !=
        VK_SUCCESS) {
      vkDestroyCommandPool(device, stream->command_pool, nullptr);
      throw std::runtime_error("failed to allocate command buffer");
    }

    // Create fence for synchronization
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if (vkCreateFence(device, &fenceInfo, nullptr, &stream->fence) !=
        VK_SUCCESS) {
      vkFreeCommandBuffers(
          device, stream->command_pool, 1, &stream->command_buffer);
      vkDestroyCommandPool(device, stream->command_pool, nullptr);
      throw std::runtime_error("failed to create fence");
    }

    return stream;
  }

  void submit_commands(StreamData* stream) {
    if (!stream->recording) {
      return;
    }

    VkDevice device = VulkanContext::get().device();
    VkQueue queue = VulkanContext::get().compute_queue();

    throw_if_vk_error(
        vkEndCommandBuffer(stream->command_buffer),
        "[vulkan::submit_commands] Failed ending command buffer");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &stream->command_buffer;

    throw_if_vk_error(
        vkResetFences(device, 1, &stream->fence),
        "[vulkan::submit_commands] Failed resetting stream fence");
    throw_if_vk_error(
        vkQueueSubmit(queue, 1, &submitInfo, stream->fence),
        "[vulkan::submit_commands] Failed submitting command buffer");
    stream->recording = false;
    stream->has_pending_work = true;
  }

  std::mutex mutex_;
  std::unordered_map<int, std::unique_ptr<StreamData>> streams_;
};

} // namespace mlx::core::vulkan

namespace mlx::core::gpu {

void new_stream(Stream s) {
  if (s.device == mlx::core::Device::gpu) {
    mlx::core::vulkan::VulkanDevice::get().ensure_stream(s.index);
  }
}

void synchronize(Stream s) {
  mlx::core::vulkan::VulkanDevice::get().synchronize(s);
}

} // namespace mlx::core::gpu

namespace mlx::core::vulkan {

// Expose VulkanDevice methods to other files
VkCommandBuffer begin_command_recording(int stream_index) {
  return VulkanDevice::get().begin_recording(stream_index);
}

void end_command_recording(int stream_index) {
  VulkanDevice::get().end_recording(stream_index);
}

void synchronize_stream(Stream s) {
  VulkanDevice::get().synchronize(s);
}

void synchronize_all() {
  VulkanDevice::get().synchronize();
}

} // namespace mlx::core::vulkan
