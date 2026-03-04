// Copyright © 2024 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

namespace mlx::core::vulkan {

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index);

} // namespace mlx::core::vulkan
