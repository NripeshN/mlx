// Copyright © 2024 Apple Inc.

#pragma once

#include <vulkan/vulkan.h>
#include "mlx/stream.h"

namespace mlx::core::vulkan {

// Command buffer management
VkCommandBuffer begin_command_recording(int stream_index);
void end_command_recording(int stream_index);

// Stream synchronization
void synchronize_stream(Stream s);
void synchronize_all();

} // namespace mlx::core::vulkan
