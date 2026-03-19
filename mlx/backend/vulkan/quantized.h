#pragma once

#include "mlx/array.h"

namespace mlx::core::vulkan {

bool affine_quantize_from_float32(
    const array& in,
    array& w,
    array& scales,
    array& biases,
    Stream s,
    int group_size,
    int bits);

bool affine_dequantize_to_float32(
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    Stream s,
    int group_size,
    int bits);

} // namespace mlx::core::vulkan
