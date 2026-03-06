// Copyright © 2024 Apple Inc.

#include "mlx/device.h"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/stream.h"

namespace mlx::core::vulkan {

namespace {

bool deferred_submission_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_DEFERRED_SUBMISSION");
        env != nullptr) {
      return std::string(env) != "0";
    }

    // Deferred submission is enabled by default. Set
    // MLX_VULKAN_DEFERRED_SUBMISSION=0 to disable.
    return true;
  }();
  return enabled;
}

uint32_t max_deferred_ops() {
  static const uint32_t value = []() {
    if (const char* env = std::getenv("MLX_VULKAN_MAX_DEFERRED_OPS");
        env != nullptr) {
      try {
        const int parsed = std::stoi(env);
        return parsed > 0 ? static_cast<uint32_t>(parsed) : 1u;
      } catch (...) {
        return 32u;
      }
    }
    return 32u;
  }();
  return value;
}

bool submit_on_hazard_boundary() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_SUBMIT_ON_HAZARD");
        env != nullptr) {
      return std::string(env) != "0";
    }
    return false;
  }();
  return enabled;
}

bool trace_sync_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_TRACE_SYNC");
        env != nullptr) {
      return std::string(env) != "0";
    }
    return false;
  }();
  return enabled;
}

void trace_sync(const std::string& msg) {
  if (!trace_sync_enabled()) {
    return;
  }

  using Clock = std::chrono::steady_clock;
  static const auto start = Clock::now();
  static std::mutex trace_mutex;

  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      Clock::now() - start)
                      .count();

  std::lock_guard<std::mutex> lock(trace_mutex);
  std::cerr << "[vulkan-trace +" << ms
            << "ms tid=" << std::this_thread::get_id() << "] " << msg
            << std::endl;
}

const char* vk_result_name(VkResult result) {
  switch (result) {
    case VK_SUCCESS:
      return "VK_SUCCESS";
    case VK_NOT_READY:
      return "VK_NOT_READY";
    case VK_TIMEOUT:
      return "VK_TIMEOUT";
    case VK_EVENT_SET:
      return "VK_EVENT_SET";
    case VK_EVENT_RESET:
      return "VK_EVENT_RESET";
    case VK_INCOMPLETE:
      return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
      return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
      return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
      return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:
      return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL:
      return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_UNKNOWN:
      return "VK_ERROR_UNKNOWN";
    case VK_ERROR_OUT_OF_POOL_MEMORY:
      return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE:
      return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_FRAGMENTATION:
      return "VK_ERROR_FRAGMENTATION";
    case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS:
      return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
    case VK_ERROR_SURFACE_LOST_KHR:
      return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
      return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR:
      return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR:
      return "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
      return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
    case VK_ERROR_VALIDATION_FAILED_EXT:
      return "VK_ERROR_VALIDATION_FAILED_EXT";
    case VK_ERROR_INVALID_SHADER_NV:
      return "VK_ERROR_INVALID_SHADER_NV";
    case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT:
      return "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
    case VK_ERROR_NOT_PERMITTED_KHR:
      return "VK_ERROR_NOT_PERMITTED_KHR";
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
      return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
    case VK_THREAD_IDLE_KHR:
      return "VK_THREAD_IDLE_KHR";
    case VK_THREAD_DONE_KHR:
      return "VK_THREAD_DONE_KHR";
    case VK_OPERATION_DEFERRED_KHR:
      return "VK_OPERATION_DEFERRED_KHR";
    case VK_OPERATION_NOT_DEFERRED_KHR:
      return "VK_OPERATION_NOT_DEFERRED_KHR";
    case VK_PIPELINE_COMPILE_REQUIRED:
      return "VK_PIPELINE_COMPILE_REQUIRED";
    default:
      return "VK_RESULT_UNKNOWN";
  }
}

std::string format_vk_result(VkResult result) {
  return std::string(vk_result_name(result)) + " (" +
      std::to_string(static_cast<int>(result)) + ")";
}

void throw_if_vk_error(VkResult result, const std::string& context) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        context + " (VkResult=" + format_vk_result(result) + ").");
  }
}

} // namespace

// Stream data structure for Vulkan
struct BufferAccessRange {
  const void* buffer{nullptr};
  uint64_t begin{0};
  uint64_t end{0};
};

struct StreamData {
  VkCommandPool command_pool{VK_NULL_HANDLE};
  VkCommandBuffer command_buffer{VK_NULL_HANDLE};
  VkFence fence{VK_NULL_HANDLE};
  bool recording{false};
  bool has_pending_work{false};
  int stream_index{0};
  uint64_t recording_epoch{0};
  uint64_t pending_epoch{0};
  uint64_t next_epoch{1};
  uint32_t recorded_ops{0};
  std::vector<std::shared_ptr<array::Data>> recording_refs;
  std::unordered_set<const array::Data*> recording_ref_ids;
  std::vector<std::shared_ptr<array::Data>> in_flight_refs;
  std::vector<BufferAccessRange> unsynced_reads;
  std::vector<BufferAccessRange> unsynced_writes;
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
    if (trace_sync_enabled()) {
      std::ostringstream oss;
      oss << "sync(stream=" << s.index
          << ") begin recording=" << stream->recording
          << " pending=" << stream->has_pending_work
          << " rec_epoch=" << stream->recording_epoch
          << " pending_epoch=" << stream->pending_epoch
          << " rec_ops=" << stream->recorded_ops;
      trace_sync(oss.str());
    }
    if (stream->recording) {
      submit_commands(stream);
    }

    if (!stream->has_pending_work) {
      return;
    }

    VkDevice device = VulkanContext::get().device();
    if (trace_sync_enabled()) {
      const auto status = vkGetFenceStatus(device, stream->fence);
      std::ostringstream oss;
      oss << "sync(stream=" << s.index
          << ") wait fence status=" << format_vk_result(status)
          << " pending_epoch=" << stream->pending_epoch;
      trace_sync(oss.str());
    }
    throw_if_vk_error(
        vkWaitForFences(device, 1, &stream->fence, VK_TRUE, UINT64_MAX),
        "[vulkan::synchronize] Failed waiting for stream fence");
    throw_if_vk_error(
        vkResetFences(device, 1, &stream->fence),
        "[vulkan::synchronize] Failed resetting stream fence");
    const uint64_t completed_epoch = stream->pending_epoch;
    KernelManager::get().reclaim_descriptor_sets(s.index, completed_epoch);
    stream->has_pending_work = false;
    stream->pending_epoch = 0;
    stream->in_flight_refs.clear();
    stream->unsynced_reads.clear();
    stream->unsynced_writes.clear();
    if (trace_sync_enabled()) {
      std::ostringstream oss;
      oss << "sync(stream=" << s.index
          << ") done reclaimed_epoch=" << completed_epoch;
      trace_sync(oss.str());
    }
  }

  void synchronize() {
    trace_sync("sync(all) begin");
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto& [_, stream] : streams_) {
        if (stream->recording) {
          submit_commands(stream.get());
        }
      }
    }

    {
      std::lock_guard<std::mutex> queue_lock(queue_mutex_);
      throw_if_vk_error(
          vkQueueWaitIdle(VulkanContext::get().compute_queue()),
          "[vulkan::synchronize] Failed waiting for compute queue idle");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [_, stream] : streams_) {
      stream->recording = false;
      stream->recording_epoch = 0;
      stream->pending_epoch = 0;
      stream->recorded_ops = 0;
      stream->recording_refs.clear();
      stream->recording_ref_ids.clear();
      stream->in_flight_refs.clear();
      stream->unsynced_reads.clear();
      stream->unsynced_writes.clear();
      stream->has_pending_work = false;
    }
    KernelManager::get().reclaim_all_descriptor_sets();
    trace_sync("sync(all) done");
  }

  void retain_array(int stream_index, const array& arr) {
    auto* stream = get_stream(stream_index);
    if (!stream->recording) {
      return;
    }

    auto data = arr.data_shared_ptr();
    if (!data) {
      return;
    }

    if (stream->recording_ref_ids.insert(data.get()).second) {
      stream->recording_refs.push_back(std::move(data));
    }
  }

  uint64_t descriptor_epoch(int stream_index) {
    auto* stream = get_stream(stream_index);
    if (stream->recording) {
      return stream->recording_epoch;
    }
    if (stream->has_pending_work) {
      return stream->pending_epoch;
    }
    return 0;
  }

  void begin_primitive(
      const Stream& s,
      const std::vector<array>& inputs,
      const std::vector<array>& outputs) {
    if (!deferred_submission_enabled()) {
      return;
    }

    auto* stream = get_stream(s.index);
    if (!stream->recording) {
      return;
    }

    const auto reads = make_access_ranges(inputs);
    const auto writes = make_access_ranges(outputs);
    if (!has_access_hazard(stream, reads, writes)) {
      return;
    }

    if (trace_sync_enabled()) {
      std::ostringstream oss;
      oss << "hazard(stream=" << s.index
          << ") detected rec_epoch=" << stream->recording_epoch
          << " rec_ops=" << stream->recorded_ops << " reads=" << reads.size()
          << " writes=" << writes.size()
          << " unsynced_reads=" << stream->unsynced_reads.size()
          << " unsynced_writes=" << stream->unsynced_writes.size();
      trace_sync(oss.str());
    }

    if (submit_on_hazard_boundary() && stream->recorded_ops > 0) {
      trace_sync("hazard boundary action=submit");
      submit_commands(stream);
      return;
    }

    trace_sync("hazard boundary action=barrier");
    insert_memory_barrier(stream->command_buffer);
    stream->unsynced_reads.clear();
    stream->unsynced_writes.clear();
  }

  void end_primitive(
      const Stream& s,
      const std::vector<array>& inputs,
      const std::vector<array>& outputs) {
    if (!deferred_submission_enabled()) {
      return;
    }

    auto* stream = get_stream(s.index);
    if (!stream->recording) {
      return;
    }

    auto reads = make_access_ranges(inputs);
    auto writes = make_access_ranges(outputs);
    stream->unsynced_reads.insert(
        stream->unsynced_reads.end(), reads.begin(), reads.end());
    stream->unsynced_writes.insert(
        stream->unsynced_writes.end(), writes.begin(), writes.end());
  }

  VkCommandBuffer begin_recording(int stream_index) {
    auto* stream = get_stream(stream_index);

    if (!stream->recording) {
      VkDevice device = VulkanContext::get().device();

      if (stream->has_pending_work) {
        if (trace_sync_enabled()) {
          const auto fence_status = vkGetFenceStatus(device, stream->fence);
          std::ostringstream oss;
          oss << "begin_recording(stream=" << stream_index
              << ") wait pending_epoch=" << stream->pending_epoch
              << " fence_status=" << format_vk_result(fence_status);
          trace_sync(oss.str());
        }
        throw_if_vk_error(
            vkWaitForFences(device, 1, &stream->fence, VK_TRUE, UINT64_MAX),
            "[vulkan::begin_recording] Failed waiting for stream fence");
        throw_if_vk_error(
            vkResetFences(device, 1, &stream->fence),
            "[vulkan::begin_recording] Failed resetting stream fence");
        KernelManager::get().reclaim_descriptor_sets(
            stream_index, stream->pending_epoch);
        stream->has_pending_work = false;
        stream->pending_epoch = 0;
        stream->in_flight_refs.clear();
        stream->unsynced_reads.clear();
        stream->unsynced_writes.clear();
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
      stream->recording_epoch = stream->next_epoch++;
      stream->recorded_ops = 0;
      stream->recording_refs.clear();
      stream->recording_ref_ids.clear();
      if (trace_sync_enabled()) {
        std::ostringstream oss;
        oss << "begin_recording(stream=" << stream_index
            << ") rec_epoch=" << stream->recording_epoch
            << " next_epoch=" << stream->next_epoch;
        trace_sync(oss.str());
      }
    }

    return stream->command_buffer;
  }

  void end_recording(int stream_index) {
    auto* stream = get_stream(stream_index);
    if (!stream->recording) {
      return;
    }

    if (trace_sync_enabled()) {
      std::ostringstream oss;
      oss << "end_recording(stream=" << stream_index
          << ") rec_epoch=" << stream->recording_epoch
          << " rec_ops=" << stream->recorded_ops
          << " deferred=" << deferred_submission_enabled();
      trace_sync(oss.str());
    }

    if (!deferred_submission_enabled()) {
      trace_sync("end_recording action=submit immediate");
      submit_commands(stream);
      return;
    }

    stream->recorded_ops += 1;
    if (stream->recorded_ops >= max_deferred_ops()) {
      if (trace_sync_enabled()) {
        std::ostringstream oss;
        oss << "end_recording action=submit threshold stream=" << stream_index
            << " rec_epoch=" << stream->recording_epoch
            << " rec_ops=" << stream->recorded_ops
            << " threshold=" << max_deferred_ops();
        trace_sync(oss.str());
      }
      submit_commands(stream);
      return;
    }

    // Keep a conservative memory dependency between deferred primitive
    // dispatches in the same command buffer. Hazard tracking can still force
    // stronger boundaries (submit) when requested.
    insert_memory_barrier(stream->command_buffer);
  }

 private:
  VulkanDevice() = default;

  static std::vector<BufferAccessRange> make_access_ranges(
      const std::vector<array>& arrays) {
    std::vector<BufferAccessRange> ranges;
    ranges.reserve(arrays.size());

    for (const auto& arr : arrays) {
      auto data = arr.data_shared_ptr();
      if (!data || arr.data_size() == 0) {
        continue;
      }

      const void* buffer_ptr = data->buffer.ptr();
      if (buffer_ptr == nullptr) {
        continue;
      }

      const auto item_size = static_cast<uint64_t>(size_of(arr.dtype()));
      if (item_size == 0) {
        continue;
      }

      const int64_t offset = arr.offset();
      if (offset < 0) {
        continue;
      }

      const uint64_t begin = static_cast<uint64_t>(offset) * item_size;
      const uint64_t size_bytes =
          static_cast<uint64_t>(arr.data_size()) * item_size;
      ranges.push_back(
          BufferAccessRange{buffer_ptr, begin, begin + size_bytes});
    }

    return ranges;
  }

  static bool overlaps(const BufferAccessRange& a, const BufferAccessRange& b) {
    return a.buffer == b.buffer && a.begin < b.end && b.begin < a.end;
  }

  static bool has_access_hazard(
      StreamData* stream,
      const std::vector<BufferAccessRange>& reads,
      const std::vector<BufferAccessRange>& writes) {
    for (const auto& w : writes) {
      for (const auto& prev_w : stream->unsynced_writes) {
        if (overlaps(w, prev_w)) {
          return true;
        }
      }
      for (const auto& prev_r : stream->unsynced_reads) {
        if (overlaps(w, prev_r)) {
          return true;
        }
      }
    }

    for (const auto& r : reads) {
      for (const auto& prev_w : stream->unsynced_writes) {
        if (overlaps(r, prev_w)) {
          return true;
        }
      }
    }

    return false;
  }

  static void insert_memory_barrier(VkCommandBuffer command_buffer) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
        VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
        VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1,
        &barrier,
        0,
        nullptr,
        0,
        nullptr);
  }

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
      std::lock_guard<std::mutex> queue_lock(queue_mutex_);
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

    const uint64_t submit_rec_epoch = stream->recording_epoch;
    const uint32_t submit_rec_ops = stream->recorded_ops;
    const size_t submit_recording_refs = stream->recording_refs.size();
    const size_t submit_unsynced_reads = stream->unsynced_reads.size();
    const size_t submit_unsynced_writes = stream->unsynced_writes.size();
    int last_queue_submit_retry = -1;
    VkResult end_cmd_result = VK_SUCCESS;
    VkResult reset_fence_result = VK_SUCCESS;
    VkResult last_queue_submit_result = VK_SUCCESS;

    if (trace_sync_enabled()) {
      std::ostringstream oss;
      oss << "submit begin stream=" << stream->stream_index
          << " rec_epoch=" << stream->recording_epoch
          << " rec_ops=" << stream->recorded_ops
          << " pending=" << stream->has_pending_work;
      trace_sync(oss.str());
    }

    VkDevice device = VulkanContext::get().device();
    VkQueue queue = VulkanContext::get().compute_queue();

    auto fail_submit = [&](VkResult result, const std::string& context) {
      const VkResult fence_status = vkGetFenceStatus(device, stream->fence);
      VkPhysicalDeviceProperties props{};
      vkGetPhysicalDeviceProperties(
          VulkanContext::get().physical_device(), &props);

      stream->recording = false;
      stream->recording_epoch = 0;
      stream->pending_epoch = 0;
      stream->recorded_ops = 0;
      stream->recording_refs.clear();
      stream->recording_ref_ids.clear();
      stream->in_flight_refs.clear();
      stream->unsynced_reads.clear();
      stream->unsynced_writes.clear();
      stream->has_pending_work = false;
      KernelManager::get().reclaim_descriptor_sets(stream->stream_index);

      const VkResult reset_pool_result =
          vkResetCommandPool(device, stream->command_pool, 0);

      std::ostringstream details;
      details << " stream=" << stream->stream_index
              << " rec_epoch=" << submit_rec_epoch
              << " rec_ops=" << submit_rec_ops
              << " recording_refs=" << submit_recording_refs
              << " unsynced_reads=" << submit_unsynced_reads
              << " unsynced_writes=" << submit_unsynced_writes
              << " fence_status=" << format_vk_result(fence_status)
              << " end_cmd_result=" << format_vk_result(end_cmd_result)
              << " reset_fence_result=" << format_vk_result(reset_fence_result)
              << " last_submit_retry=" << last_queue_submit_retry
              << " last_submit_result="
              << format_vk_result(last_queue_submit_result)
              << " reset_pool=" << format_vk_result(reset_pool_result)
              << " device='" << props.deviceName << "'";

      trace_sync(context + details.str());

      throw std::runtime_error(
          context + " (VkResult=" + format_vk_result(result) + ";" +
          details.str() + ").");
    };

    VkResult result = vkEndCommandBuffer(stream->command_buffer);
    end_cmd_result = result;
    if (result != VK_SUCCESS) {
      fail_submit(
          result, "[vulkan::submit_commands] Failed ending command buffer");
    }
    trace_sync("submit end_command_buffer success");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &stream->command_buffer;

    result = vkResetFences(device, 1, &stream->fence);
    reset_fence_result = result;
    if (result != VK_SUCCESS) {
      fail_submit(
          result, "[vulkan::submit_commands] Failed resetting stream fence");
    }
    trace_sync("submit reset_fence success");

    constexpr int kSubmitRetryCount = 8;
    {
      std::lock_guard<std::mutex> queue_lock(queue_mutex_);
      for (int retry = 0; retry < kSubmitRetryCount; ++retry) {
        last_queue_submit_retry = retry;
        result = vkQueueSubmit(queue, 1, &submitInfo, stream->fence);
        last_queue_submit_result = result;
        if (result == VK_SUCCESS) {
          if (trace_sync_enabled()) {
            std::ostringstream oss;
            oss << "submit queue_submit success retry=" << retry
                << " stream=" << stream->stream_index
                << " rec_epoch=" << stream->recording_epoch;
            trace_sync(oss.str());
          }
          break;
        }
        if (trace_sync_enabled()) {
          std::ostringstream oss;
          oss << "submit queue_submit retry=" << retry
              << " result=" << format_vk_result(result)
              << " stream=" << stream->stream_index
              << " rec_epoch=" << stream->recording_epoch;
          trace_sync(oss.str());
        }
        if (result != VK_TIMEOUT && result != VK_NOT_READY) {
          break;
        }
        std::this_thread::yield();
      }
    }
    if (result != VK_SUCCESS) {
      fail_submit(
          result, "[vulkan::submit_commands] Failed submitting command buffer");
    }

    stream->recording = false;
    stream->recorded_ops = 0;
    stream->pending_epoch = stream->recording_epoch;
    stream->recording_epoch = 0;
    stream->in_flight_refs = std::move(stream->recording_refs);
    stream->recording_ref_ids.clear();
    stream->recording_refs.clear();
    stream->unsynced_reads.clear();
    stream->unsynced_writes.clear();
    stream->has_pending_work = true;
    if (trace_sync_enabled()) {
      std::ostringstream oss;
      oss << "submit done stream=" << stream->stream_index
          << " pending_epoch=" << stream->pending_epoch
          << " inflight_refs=" << stream->in_flight_refs.size();
      trace_sync(oss.str());
    }
  }

  std::mutex mutex_;
  std::mutex queue_mutex_;
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

void retain_array_for_stream(const Stream& s, const array& arr) {
  VulkanDevice::get().retain_array(s.index, arr);
}

uint64_t descriptor_epoch_for_stream(const Stream& s) {
  return VulkanDevice::get().descriptor_epoch(s.index);
}

void begin_primitive_tracking(
    const Stream& s,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs) {
  VulkanDevice::get().begin_primitive(s, inputs, outputs);
}

void end_primitive_tracking(
    const Stream& s,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs) {
  VulkanDevice::get().end_primitive(s, inputs, outputs);
}

void synchronize_stream(Stream s) {
  VulkanDevice::get().synchronize(s);
}

void synchronize_all() {
  VulkanDevice::get().synchronize();
}

} // namespace mlx::core::vulkan
