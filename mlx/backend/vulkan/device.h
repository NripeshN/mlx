// Copyright © 2024 Apple Inc.

#pragma once

#include <vulkan/vulkan.h>

#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::vulkan {

// Command buffer management
VkCommandBuffer begin_command_recording(int stream_index);
void end_command_recording(int stream_index);
void retain_array_for_stream(const Stream& s, const array& arr);
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
