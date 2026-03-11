// Copyright © 2024 Apple Inc.

#include "mlx/device.h"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
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
        return 16u;
      }
    }
    return 16u;
  }();
  return value;
}

bool barrier_between_deferred_ops() {
  static const bool enabled = []() {
    if (const char* env =
            std::getenv("MLX_VULKAN_BARRIER_BETWEEN_DEFERRED_OPS");
        env != nullptr) {
      return std::string(env) != "0";
    }

    return true;
  }();
  return enabled;
}

bool submit_on_hazard_boundary() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_SUBMIT_ON_HAZARD");
        env != nullptr) {
      return std::string(env) != "0";
    }

    return true;
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

thread_local std::vector<std::string> sync_label_stack;

std::string current_sync_label() {
  return sync_label_stack.empty() ? std::string{} : sync_label_stack.back();
}

void push_sync_label(std::string label) {
  sync_label_stack.push_back(std::move(label));
}

void pop_sync_label() {
  if (!sync_label_stack.empty()) {
    sync_label_stack.pop_back();
  }
}

} // namespace

// Stream data structure for Vulkan
struct BufferAccessRange {
  const VulkanBuffer* storage{nullptr};
  VkBuffer buffer{VK_NULL_HANDLE};
  uint64_t begin{0};
  uint64_t end{0};
};

struct SubmissionResources {
  VkCommandPool command_pool{VK_NULL_HANDLE};
  VkCommandBuffer command_buffer{VK_NULL_HANDLE};
  VkFence fence{VK_NULL_HANDLE};
};

struct SubmissionRecord {
  std::unique_ptr<SubmissionResources> resources;
  uint64_t epoch{0};
  uint32_t recorded_ops{0};
  std::vector<std::shared_ptr<array::Data>> refs;
  std::unordered_set<const array::Data*> ref_ids;
  std::vector<std::shared_ptr<void>> keepalive_resources;
  std::vector<std::function<void()>> completion_callbacks;
  std::string submit_reason;
};

struct ScratchSlot {
  std::optional<array> owner;
  size_t bytes{0};
  bool needs_barrier{false};
};

std::shared_ptr<array::Data> make_owned_staging_allocation(size_t size) {
  auto data = std::make_shared<array::Data>(allocator::malloc(size));
  auto* buffer = static_cast<VulkanBuffer*>(data->buffer.ptr());
  if (buffer == nullptr || buffer->buffer == VK_NULL_HANDLE ||
      buffer->mapped_ptr == nullptr) {
    throw std::runtime_error(
        "[vulkan::staging] Failed to allocate host-visible staging buffer.");
  }
  return data;
}

const VulkanBuffer* get_vulkan_buffer(
    const std::shared_ptr<array::Data>& data) {
  return data ? static_cast<const VulkanBuffer*>(data->buffer.ptr()) : nullptr;
}

size_t scratch_nbytes(const Shape& shape, Dtype dtype) {
  size_t elements = 1;
  for (auto dim : shape) {
    elements *= static_cast<size_t>(dim);
  }
  return elements * size_of(dtype);
}

array make_scratch_owner(size_t bytes) {
  if (bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "[vulkan::scratch] Requested scratch allocation exceeds int32 array limit.");
  }
  array owner({static_cast<int>(bytes)}, uint8, nullptr, {});
  owner.set_status(array::Status::available);
  owner.set_data(allocator::malloc(bytes));
  return owner;
}

array make_scratch_view(const array& owner, Shape shape, Dtype dtype) {
  array scratch(shape, dtype, nullptr, {});
  auto strides = make_contiguous_strides(shape);
  auto [data_size, row_contiguous, col_contiguous] =
      check_contiguity(shape, strides);
  scratch.copy_shared_buffer(
      owner,
      std::move(strides),
      {true, row_contiguous, col_contiguous},
      data_size);
  scratch.set_status(array::Status::available);
  return scratch;
}

struct StreamData {
  std::unique_ptr<SubmissionResources> recording_resources;
  std::vector<std::unique_ptr<SubmissionResources>> available_resources;
  std::deque<SubmissionRecord> in_flight_submissions;
  bool recording{false};
  int stream_index{0};
  uint64_t recording_epoch{0};
  uint64_t next_epoch{1};
  uint32_t recorded_ops{0};
  std::vector<std::shared_ptr<array::Data>> recording_refs;
  std::unordered_set<const array::Data*> recording_ref_ids;
  std::vector<std::shared_ptr<void>> recording_keepalive_resources;
  std::vector<std::function<void()>> recording_completion_callbacks;
  std::vector<BufferAccessRange> unsynced_reads;
  std::vector<BufferAccessRange> unsynced_writes;
  std::unordered_map<std::string, ScratchSlot> scratch_slots;
  std::deque<std::string> recent_primitives;
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
          << " pending=" << !stream->in_flight_submissions.empty()
          << " rec_epoch=" << stream->recording_epoch
          << " inflight=" << stream->in_flight_submissions.size()
          << " rec_ops=" << stream->recorded_ops;
      auto label = current_sync_label();
      if (!label.empty()) {
        oss << " label='" << label << "'";
      }
      trace_sync(oss.str());
    }
    if (stream->recording) {
      auto label = current_sync_label();
      submit_commands(
          stream,
          label.empty() ? std::string("explicit synchronize")
                        : std::string("explicit synchronize:") + label);
    }

    retire_submissions(stream, true);
    clear_scratch_barriers(stream);
    if (trace_sync_enabled()) {
      std::ostringstream oss;
      oss << "sync(stream=" << s.index << ") done inflight=0";
      trace_sync(oss.str());
    }
  }

  void finalize(Stream s) {
    auto* stream = get_stream(s.index);
    if (trace_sync_enabled()) {
      std::ostringstream oss;
      oss << "finalize(stream=" << s.index
          << ") begin recording=" << stream->recording
          << " pending=" << !stream->in_flight_submissions.empty()
          << " rec_epoch=" << stream->recording_epoch
          << " inflight=" << stream->in_flight_submissions.size()
          << " rec_ops=" << stream->recorded_ops;
      trace_sync(oss.str());
    }
    if (stream->recording) {
      submit_commands(stream, "finalize");
    }
    retire_submissions(stream, false);
    if (trace_sync_enabled()) {
      std::ostringstream oss;
      oss << "finalize(stream=" << s.index
          << ") done inflight=" << stream->in_flight_submissions.size();
      trace_sync(oss.str());
    }
  }

  void synchronize() {
    trace_sync("sync(all) begin");
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto& [_, stream] : streams_) {
        if (stream->recording) {
          submit_commands(stream.get(), "synchronize_all");
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
      retire_submissions(stream.get(), true);
      stream->recording = false;
      stream->recording_epoch = 0;
      stream->recorded_ops = 0;
      stream->recording_refs.clear();
      stream->recording_ref_ids.clear();
      stream->recording_keepalive_resources.clear();
      stream->recording_completion_callbacks.clear();
      stream->unsynced_reads.clear();
      stream->unsynced_writes.clear();
      clear_scratch_barriers(stream.get());
    }
    trace_sync("sync(all) done");
  }

  void retain_array(int stream_index, const array& arr) {
    auto* stream = get_stream(stream_index);
    auto data = arr.data_shared_ptr();
    if (!data) {
      return;
    }

    if (stream->recording) {
      if (stream->recording_ref_ids.insert(data.get()).second) {
        stream->recording_refs.push_back(std::move(data));
      }
      return;
    }

    if (!stream->in_flight_submissions.empty()) {
      auto& submission = stream->in_flight_submissions.back();
      if (submission.ref_ids.insert(data.get()).second) {
        submission.refs.push_back(std::move(data));
      }
    }
  }

  void retain_shared(int stream_index, std::shared_ptr<void> resource) {
    if (!resource) {
      return;
    }

    auto* stream = get_stream(stream_index);
    if (stream->recording) {
      stream->recording_keepalive_resources.push_back(std::move(resource));
      return;
    }

    if (!stream->in_flight_submissions.empty()) {
      stream->in_flight_submissions.back().keepalive_resources.push_back(
          std::move(resource));
    }
  }

  void add_completion_callback(
      int stream_index,
      std::function<void()> callback) {
    auto* stream = get_stream(stream_index);
    if (stream->recording) {
      stream->recording_completion_callbacks.push_back(std::move(callback));
      return;
    }

    if (!stream->in_flight_submissions.empty()) {
      stream->in_flight_submissions.back().completion_callbacks.push_back(
          std::move(callback));
      return;
    }

    callback();
  }

  uint64_t descriptor_epoch(int stream_index) {
    auto* stream = get_stream(stream_index);
    if (stream->recording) {
      return stream->recording_epoch;
    }
    if (!stream->in_flight_submissions.empty()) {
      return stream->in_flight_submissions.back().epoch;
    }
    return 0;
  }

  array acquire_scratch(
      const Stream& s,
      const std::string& lane,
      Shape shape,
      Dtype dtype) {
    auto* stream = get_stream(s.index);
    const size_t bytes = scratch_nbytes(shape, dtype);
    auto& slot = stream->scratch_slots[lane];

    if (!slot.owner.has_value() || slot.bytes < bytes) {
      slot.owner = make_scratch_owner(bytes);
      slot.bytes = bytes;
      slot.needs_barrier = false;
    } else if (slot.needs_barrier && stream->recording) {
      trace_sync(
          "scratch barrier lane='" + lane +
          "' bytes=" + std::to_string(slot.bytes));
      insert_memory_barrier(stream->recording_resources->command_buffer);
      stream->unsynced_reads.clear();
      stream->unsynced_writes.clear();
      slot.needs_barrier = false;
    }

    return make_scratch_view(*slot.owner, std::move(shape), dtype);
  }

  void mark_scratch_written(const Stream& s, const std::string& lane) {
    auto* stream = get_stream(s.index);
    auto it = stream->scratch_slots.find(lane);
    if (it != stream->scratch_slots.end()) {
      it->second.needs_barrier = true;
    }
  }

  void record_primitive(const Stream& s, std::string name) {
    auto* stream = get_stream(s.index);
    stream->recent_primitives.push_back(std::move(name));
    constexpr size_t kRecentPrimitiveLimit = 8;
    while (stream->recent_primitives.size() > kRecentPrimitiveLimit) {
      stream->recent_primitives.pop_front();
    }
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
    auto writes = make_access_ranges(outputs);
    auto donation_writes = make_potential_donation_writes(inputs, outputs);
    writes.insert(writes.end(), donation_writes.begin(), donation_writes.end());
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
      trace_sync(
          "hazard boundary action=submit reason=overlapping-buffer-range");
      submit_commands(stream, "hazard overlap");
      return;
    }

    trace_sync(
        "hazard boundary action=barrier reason=overlapping-buffer-range");
    trace_sync("barrier action=recording-tail reason=deferred-op-boundary");
    insert_memory_barrier(stream->recording_resources->command_buffer);
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
      retire_submissions(stream, false);

      auto resources = acquire_submission_resources(stream);
      VkDevice device = VulkanContext::get().device();

      // Reset command pool to allow reuse
      throw_if_vk_error(
          vkResetCommandPool(device, resources->command_pool, 0),
          "[vulkan::begin_recording] Failed resetting command pool");

      // Begin recording
      VkCommandBufferBeginInfo beginInfo{};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

      throw_if_vk_error(
          vkBeginCommandBuffer(resources->command_buffer, &beginInfo),
          "[vulkan::begin_recording] Failed beginning command buffer");
      stream->recording_resources = std::move(resources);
      stream->recording = true;
      stream->recording_epoch = stream->next_epoch++;
      stream->recorded_ops = 0;
      stream->recording_refs.clear();
      stream->recording_ref_ids.clear();
      stream->recording_keepalive_resources.clear();
      stream->recording_completion_callbacks.clear();
      stream->unsynced_reads.clear();
      stream->unsynced_writes.clear();
      if (trace_sync_enabled()) {
        std::ostringstream oss;
        oss << "begin_recording(stream=" << stream_index
            << ") rec_epoch=" << stream->recording_epoch
            << " inflight=" << stream->in_flight_submissions.size()
            << " next_epoch=" << stream->next_epoch;
        trace_sync(oss.str());
      }
    }

    return stream->recording_resources->command_buffer;
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
      submit_commands(stream, "immediate");
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
      submit_commands(stream, "threshold reached");
      return;
    }

    if (barrier_between_deferred_ops()) {
      trace_sync("barrier action=recording-tail reason=deferred-op-boundary");
      insert_memory_barrier(stream->recording_resources->command_buffer);
    }
  }

 private:
  VulkanDevice() = default;

  static std::string format_access_range(const BufferAccessRange& range) {
    std::ostringstream oss;
    oss << "buffer=0x" << std::hex << reinterpret_cast<uintptr_t>(range.storage)
        << "/0x" << reinterpret_cast<uintptr_t>(range.buffer) << std::dec
        << " [" << range.begin << ", " << range.end << ")";
    return oss.str();
  }

  static std::vector<BufferAccessRange> make_access_ranges(
      const std::vector<array>& arrays) {
    std::vector<BufferAccessRange> ranges;
    ranges.reserve(arrays.size());

    for (const auto& arr : arrays) {
      auto data = arr.data_shared_ptr();
      if (!data || arr.data_size() == 0) {
        continue;
      }

      auto* storage = static_cast<const VulkanBuffer*>(
          const_cast<void*>(static_cast<const void*>(data->buffer.ptr())));
      if (storage == nullptr || storage->buffer == VK_NULL_HANDLE) {
        continue;
      }

      const auto item_size = static_cast<uint64_t>(size_of(arr.dtype()));
      if (item_size == 0) {
        continue;
      }

      const int64_t offset_bytes = arr.offset();
      if (offset_bytes < 0) {
        continue;
      }

      const uint64_t begin = static_cast<uint64_t>(offset_bytes);
      const uint64_t size_bytes =
          static_cast<uint64_t>(arr.data_size()) * item_size;
      const uint64_t buffer_size = static_cast<uint64_t>(storage->size);
      const uint64_t end = std::min(begin + size_bytes, buffer_size);
      if (begin >= end) {
        continue;
      }
      ranges.push_back(BufferAccessRange{storage, storage->buffer, begin, end});
    }

    return ranges;
  }

  static std::vector<BufferAccessRange> make_potential_donation_writes(
      const std::vector<array>& inputs,
      const std::vector<array>& outputs) {
    std::vector<BufferAccessRange> writes;
    writes.reserve(inputs.size());

    for (const auto& in : inputs) {
      bool can_donate = false;
      for (const auto& out : outputs) {
        if (is_donatable(in, out)) {
          can_donate = true;
          break;
        }
      }
      if (!can_donate) {
        continue;
      }

      auto input_writes = make_access_ranges({in});
      writes.insert(writes.end(), input_writes.begin(), input_writes.end());
    }

    return writes;
  }

  static void clear_scratch_barriers(StreamData* stream) {
    for (auto& [_, slot] : stream->scratch_slots) {
      slot.needs_barrier = false;
    }
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
          if (trace_sync_enabled()) {
            trace_sync(
                "hazard waw current=" + format_access_range(w) +
                " previous=" + format_access_range(prev_w));
          }
          return true;
        }
      }
      for (const auto& prev_r : stream->unsynced_reads) {
        if (overlaps(w, prev_r)) {
          if (trace_sync_enabled()) {
            trace_sync(
                "hazard war current=" + format_access_range(w) +
                " previous=" + format_access_range(prev_r));
          }
          return true;
        }
      }
    }

    for (const auto& r : reads) {
      for (const auto& prev_w : stream->unsynced_writes) {
        if (overlaps(r, prev_w)) {
          if (trace_sync_enabled()) {
            trace_sync(
                "hazard raw current=" + format_access_range(r) +
                " previous=" + format_access_range(prev_w));
          }
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

  std::unique_ptr<SubmissionResources> create_submission_resources() {
    VkDevice device = VulkanContext::get().device();
    uint32_t queue_family = VulkanContext::get().compute_queue_family_index();

    auto resources = std::make_unique<SubmissionResources>();

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(
            device, &pool_info, nullptr, &resources->command_pool) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool");
    }

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = resources->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(
            device, &alloc_info, &resources->command_buffer) != VK_SUCCESS) {
      vkDestroyCommandPool(device, resources->command_pool, nullptr);
      throw std::runtime_error("failed to allocate command buffer");
    }

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if (vkCreateFence(device, &fence_info, nullptr, &resources->fence) !=
        VK_SUCCESS) {
      vkFreeCommandBuffers(
          device, resources->command_pool, 1, &resources->command_buffer);
      vkDestroyCommandPool(device, resources->command_pool, nullptr);
      throw std::runtime_error("failed to create fence");
    }

    return resources;
  }

  static void destroy_submission_resources(
      VkDevice device,
      std::unique_ptr<SubmissionResources>& resources) {
    if (!resources) {
      return;
    }

    if (resources->command_buffer != VK_NULL_HANDLE &&
        resources->command_pool != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(
          device, resources->command_pool, 1, &resources->command_buffer);
      resources->command_buffer = VK_NULL_HANDLE;
    }
    if (resources->fence != VK_NULL_HANDLE) {
      vkDestroyFence(device, resources->fence, nullptr);
      resources->fence = VK_NULL_HANDLE;
    }
    if (resources->command_pool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device, resources->command_pool, nullptr);
      resources->command_pool = VK_NULL_HANDLE;
    }
    resources.reset();
  }

  std::unique_ptr<SubmissionResources> acquire_submission_resources(
      StreamData* stream) {
    if (!stream->available_resources.empty()) {
      auto resources = std::move(stream->available_resources.back());
      stream->available_resources.pop_back();
      return resources;
    }
    return create_submission_resources();
  }

  void retire_submissions(StreamData* stream, bool wait_all) {
    VkDevice device = VulkanContext::get().device();

    while (!stream->in_flight_submissions.empty()) {
      auto& submission = stream->in_flight_submissions.front();
      VkResult status = VK_SUCCESS;
      if (wait_all) {
        if (trace_sync_enabled()) {
          std::ostringstream oss;
          oss << "retire wait stream=" << stream->stream_index
              << " epoch=" << submission.epoch << " reason='"
              << submission.submit_reason << "'";
          trace_sync(oss.str());
        }
        status = vkWaitForFences(
            device, 1, &submission.resources->fence, VK_TRUE, UINT64_MAX);
      } else {
        status = vkGetFenceStatus(device, submission.resources->fence);
        if (status == VK_NOT_READY) {
          break;
        }
      }

      if (status != VK_SUCCESS) {
        throw_if_vk_error(
            status, "[vulkan::retire_submissions] Failed waiting for fence");
      }

      SubmissionRecord completed = std::move(submission);
      stream->in_flight_submissions.pop_front();

      for (auto& callback : completed.completion_callbacks) {
        callback();
      }
      KernelManager::get().reclaim_descriptor_sets(
          stream->stream_index, completed.epoch);
      stream->available_resources.push_back(std::move(completed.resources));

      if (trace_sync_enabled()) {
        std::ostringstream oss;
        oss << "retire done stream=" << stream->stream_index
            << " epoch=" << completed.epoch
            << " remaining_inflight=" << stream->in_flight_submissions.size();
        trace_sync(oss.str());
      }
    }
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
      destroy_submission_resources(device, stream->recording_resources);
      for (auto& resources : stream->available_resources) {
        destroy_submission_resources(device, resources);
      }
      stream->available_resources.clear();
      for (auto& submission : stream->in_flight_submissions) {
        destroy_submission_resources(device, submission.resources);
      }
      stream->in_flight_submissions.clear();
    }
  }

  std::unique_ptr<StreamData> create_stream(int index) {
    auto stream = std::make_unique<StreamData>();
    stream->stream_index = index;

    return stream;
  }

  void submit_commands(StreamData* stream, std::string submit_reason) {
    if (!stream->recording) {
      return;
    }

    auto resources = std::move(stream->recording_resources);
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
          << " inflight=" << stream->in_flight_submissions.size() << " reason='"
          << submit_reason << "'";
      if (!stream->recent_primitives.empty()) {
        oss << " recent_primitives='";
        for (size_t i = 0; i < stream->recent_primitives.size(); ++i) {
          if (i > 0) {
            oss << ",";
          }
          oss << stream->recent_primitives[i];
        }
        oss << "'";
      }
      trace_sync(oss.str());
    }

    VkDevice device = VulkanContext::get().device();
    VkQueue queue = VulkanContext::get().compute_queue();

    auto fail_submit = [&](VkResult result, const std::string& context) {
      const VkResult fence_status =
          resources && resources->fence != VK_NULL_HANDLE
          ? vkGetFenceStatus(device, resources->fence)
          : VK_SUCCESS;
      VkPhysicalDeviceProperties props{};
      vkGetPhysicalDeviceProperties(
          VulkanContext::get().physical_device(), &props);

      stream->recording = false;
      stream->recording_epoch = 0;
      stream->recorded_ops = 0;
      stream->recording_refs.clear();
      stream->recording_ref_ids.clear();
      stream->recording_keepalive_resources.clear();
      stream->recording_completion_callbacks.clear();
      stream->unsynced_reads.clear();
      stream->unsynced_writes.clear();
      clear_scratch_barriers(stream);
      stream->recent_primitives.clear();
      stream->recording_resources.reset();
      KernelManager::get().reclaim_descriptor_set_epoch(
          stream->stream_index, submit_rec_epoch);

      VkResult reset_pool_result = VK_SUCCESS;
      if (resources && resources->command_pool != VK_NULL_HANDLE) {
        reset_pool_result =
            vkResetCommandPool(device, resources->command_pool, 0);
      }
      if (resources) {
        stream->available_resources.push_back(std::move(resources));
      }

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
              << " submit_reason='" << submit_reason << "'"
              << " reset_pool=" << format_vk_result(reset_pool_result)
              << " device='" << props.deviceName << "'";
      if (!stream->recent_primitives.empty()) {
        details << " recent_primitives='";
        for (size_t i = 0; i < stream->recent_primitives.size(); ++i) {
          if (i > 0) {
            details << ",";
          }
          details << stream->recent_primitives[i];
        }
        details << "'";
      }

      trace_sync(context + details.str());

      throw std::runtime_error(
          context + " (VkResult=" + format_vk_result(result) + ";" +
          details.str() + ").");
    };

    VkResult result = vkEndCommandBuffer(resources->command_buffer);
    end_cmd_result = result;
    if (result != VK_SUCCESS) {
      fail_submit(
          result, "[vulkan::submit_commands] Failed ending command buffer");
    }
    trace_sync("submit end_command_buffer success");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &resources->command_buffer;

    result = vkResetFences(device, 1, &resources->fence);
    reset_fence_result = result;
    if (result != VK_SUCCESS) {
      fail_submit(
          result, "[vulkan::submit_commands] Failed resetting stream fence");
    }
    trace_sync("submit reset_fence success");

    constexpr int kSubmitRetryCount = 32;
    {
      std::lock_guard<std::mutex> queue_lock(queue_mutex_);
      for (int retry = 0; retry < kSubmitRetryCount; ++retry) {
        last_queue_submit_retry = retry;
        result = vkQueueSubmit(queue, 1, &submitInfo, resources->fence);
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
        const auto backoff_ms = std::min(8, 1 << std::min(retry, 3));
        std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
      }
    }
    if (result != VK_SUCCESS) {
      fail_submit(
          result, "[vulkan::submit_commands] Failed submitting command buffer");
    }

    stream->recording = false;
    stream->recorded_ops = 0;
    stream->recording_epoch = 0;

    SubmissionRecord submission;
    submission.resources = std::move(resources);
    submission.epoch = submit_rec_epoch;
    submission.recorded_ops = submit_rec_ops;
    submission.refs = std::move(stream->recording_refs);
    submission.ref_ids = std::move(stream->recording_ref_ids);
    submission.keepalive_resources =
        std::move(stream->recording_keepalive_resources);
    submission.completion_callbacks =
        std::move(stream->recording_completion_callbacks);
    submission.submit_reason = std::move(submit_reason);

    stream->recording_resources.reset();
    stream->recording_refs.clear();
    stream->recording_ref_ids.clear();
    stream->recording_keepalive_resources.clear();
    stream->recording_completion_callbacks.clear();
    stream->unsynced_reads.clear();
    stream->unsynced_writes.clear();
    clear_scratch_barriers(stream);
    stream->recent_primitives.clear();
    stream->in_flight_submissions.push_back(std::move(submission));

    if (trace_sync_enabled()) {
      std::ostringstream oss;
      oss << "submit done stream=" << stream->stream_index
          << " epoch=" << submit_rec_epoch
          << " inflight=" << stream->in_flight_submissions.size();
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

ScopedSyncLabel::ScopedSyncLabel(std::string label) : active_(!label.empty()) {
  if (active_) {
    push_sync_label(std::move(label));
  }
}

ScopedSyncLabel::~ScopedSyncLabel() {
  if (active_) {
    pop_sync_label();
  }
}

// Expose VulkanDevice methods to other files
VkCommandBuffer begin_command_recording(int stream_index) {
  return VulkanDevice::get().begin_recording(stream_index);
}

void end_command_recording(int stream_index) {
  VulkanDevice::get().end_recording(stream_index);
}

bool deferred_submission_active() {
  return deferred_submission_enabled();
}

void retain_array_for_stream(const Stream& s, const array& arr) {
  VulkanDevice::get().retain_array(s.index, arr);
}

void retain_shared_for_stream(const Stream& s, std::shared_ptr<void> resource) {
  VulkanDevice::get().retain_shared(s.index, std::move(resource));
}

void add_completion_callback_for_stream(
    const Stream& s,
    std::function<void()> callback) {
  VulkanDevice::get().add_completion_callback(s.index, std::move(callback));
}

void enqueue_owned_staging_upload(
    const Stream& s,
    const void* src,
    size_t size,
    VkBuffer dst_buffer,
    uint64_t dst_offset) {
  if (size == 0) {
    return;
  }
  if (src == nullptr) {
    throw std::invalid_argument(
        "[vulkan::enqueue_owned_staging_upload] Null host source.");
  }
  if (dst_buffer == VK_NULL_HANDLE) {
    throw std::invalid_argument(
        "[vulkan::enqueue_owned_staging_upload] Null destination buffer.");
  }

  auto staging = make_owned_staging_allocation(size);
  auto* staging_buffer = get_vulkan_buffer(staging);
  std::memcpy(
      static_cast<char*>(staging_buffer->mapped_ptr),
      src,
      static_cast<size_t>(size));

  VkCommandBuffer command_buffer = begin_command_recording(s.index);
  VkBufferCopy copy_region{};
  copy_region.srcOffset = 0;
  copy_region.dstOffset = static_cast<VkDeviceSize>(dst_offset);
  copy_region.size = static_cast<VkDeviceSize>(size);
  vkCmdCopyBuffer(
      command_buffer, staging_buffer->buffer, dst_buffer, 1, &copy_region);

  retain_shared_for_stream(s, std::static_pointer_cast<void>(staging));
  end_command_recording(s.index);
}

void enqueue_owned_staging_readback(
    const Stream& s,
    VkBuffer src_buffer,
    uint64_t src_offset,
    size_t size,
    std::function<void(const void*, size_t)> completion) {
  if (size == 0) {
    completion(nullptr, 0);
    return;
  }
  if (!completion) {
    throw std::invalid_argument(
        "[vulkan::enqueue_owned_staging_readback] Missing completion callback.");
  }
  if (src_buffer == VK_NULL_HANDLE) {
    throw std::invalid_argument(
        "[vulkan::enqueue_owned_staging_readback] Null source buffer.");
  }

  auto staging = make_owned_staging_allocation(size);
  auto* staging_buffer = get_vulkan_buffer(staging);

  VkCommandBuffer command_buffer = begin_command_recording(s.index);
  VkBufferCopy copy_region{};
  copy_region.srcOffset = static_cast<VkDeviceSize>(src_offset);
  copy_region.dstOffset = 0;
  copy_region.size = static_cast<VkDeviceSize>(size);
  vkCmdCopyBuffer(
      command_buffer, src_buffer, staging_buffer->buffer, 1, &copy_region);

  retain_shared_for_stream(s, std::static_pointer_cast<void>(staging));
  add_completion_callback_for_stream(
      s,
      [staging = std::move(staging),
       size,
       completion = std::move(completion)]() {
        auto* completed_buffer = get_vulkan_buffer(staging);
        completion(completed_buffer->mapped_ptr, size);
      });
  end_command_recording(s.index);
}

uint64_t descriptor_epoch_for_stream(const Stream& s) {
  return VulkanDevice::get().descriptor_epoch(s.index);
}

array acquire_scratch_array(
    const Stream& s,
    const std::string& lane,
    Shape shape,
    Dtype dtype) {
  return VulkanDevice::get().acquire_scratch(s, lane, std::move(shape), dtype);
}

void mark_scratch_array_written(const Stream& s, const std::string& lane) {
  VulkanDevice::get().mark_scratch_written(s, lane);
}

void record_primitive_for_stream(const Stream& s, std::string name) {
  VulkanDevice::get().record_primitive(s, std::move(name));
}

void begin_primitive_tracking(
    const Stream& s,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs) {
  VulkanDevice::get().begin_primitive(s, inputs, outputs);
}

void finalize_stream(Stream s) {
  VulkanDevice::get().finalize(s);
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
