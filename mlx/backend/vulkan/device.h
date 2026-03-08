// Copyright © 2024 Apple Inc.

#pragma once

#include <vulkan/vulkan.h>
#include <cstddef>
#include <functional>
#include <memory>

#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::vulkan {

// Command buffer management
VkCommandBuffer begin_command_recording(int stream_index);
void end_command_recording(int stream_index);
bool deferred_submission_active();
void retain_array_for_stream(const Stream& s, const array& arr);
void retain_shared_for_stream(const Stream& s, std::shared_ptr<void> resource);
void add_completion_callback_for_stream(
    const Stream& s,
    std::function<void()> callback);
void enqueue_owned_staging_upload(
    const Stream& s,
    const void* src,
    size_t size,
    VkBuffer dst_buffer,
    uint64_t dst_offset = 0);
void enqueue_owned_staging_readback(
    const Stream& s,
    VkBuffer src_buffer,
    uint64_t src_offset,
    size_t size,
    std::function<void(const void*, size_t)> completion);
uint64_t descriptor_epoch_for_stream(const Stream& s);

// Primitive-level hazard tracking for deferred recording
void begin_primitive_tracking(
    const Stream& s,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs);
void end_primitive_tracking(
    const Stream& s,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs);

// Stream synchronization
void synchronize_stream(Stream s);
void synchronize_all();

} // namespace mlx::core::vulkan
