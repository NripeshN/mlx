// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

bool trace_fallback_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_TRACE_FALLBACKS");
        env != nullptr) {
      return std::string(env) != "0";
    }
    return false;
  }();
  return enabled;
}

void trace_fallback(const std::string& msg) {
  if (!trace_fallback_enabled()) {
    return;
  }
  static std::mutex trace_mutex;
  std::lock_guard<std::mutex> lock(trace_mutex);
  std::cerr << "[vulkan-fallback] " << msg << "\n";
}

void trace_use_fallback(
    std::string_view primitive_name,
    Stream s,
    std::string_view reason,
    std::string_view details) {
  if (!trace_fallback_enabled()) {
    return;
  }
  std::ostringstream oss;
  oss << "primitive=" << primitive_name << " kind=use_fallback"
      << " stream=" << s.index << " reason=" << reason;
  if (!details.empty()) {
    oss << ' ' << details;
  }
  trace_fallback(oss.str());
}

void trace_vulkan_unsupported(
    std::string_view primitive_name,
    std::string_view reason) {
  if (!trace_fallback_enabled()) {
    return;
  }
  std::ostringstream oss;
  oss << "primitive=" << primitive_name << " kind=unsupported"
      << " reason=" << reason;
  trace_fallback(oss.str());
}

uint32_t checked_u32_size(int64_t value, const char* name) {
  if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        std::string("[vulkan::primitives] ") + name +
        " is out of uint32 range.");
  }
  return static_cast<uint32_t>(value);
}

uint32_t checked_mul_u32(uint32_t a, uint32_t b, const char* name) {
  const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
  if (product > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        std::string("[vulkan::primitives] ") + name +
        " is out of uint32 range.");
  }
  return static_cast<uint32_t>(product);
}

uint32_t checked_product_u32(const Shape& shape, const char* name) {
  uint32_t product = 1;
  for (auto dim : shape) {
    product = checked_mul_u32(product, checked_u32_size(dim, name), name);
  }
  return product;
}

bool is_vulkan_float_dtype(Dtype dtype) {
  return dtype == float16 || dtype == float32 || dtype == bfloat16;
}

std::string dtype_suffix(Dtype dtype) {
  switch (dtype) {
    case float16:
      return "f16";
    case float32:
      return "f32";
    case bfloat16:
      return "bf16";
    case int8:
      return "i8";
    case int16:
      return "i16";
    case int32:
      return "i32";
    case int64:
      return "i64";
    case uint8:
      return "u8";
    case uint16:
      return "u16";
    case uint32:
      return "u32";
    case uint64:
      return "u64";
    case bool_:
      return "bool";
    default:
      return {};
  }
}

std::string gather_index_suffix(Dtype dtype) {
  switch (dtype) {
    case int32:
      return "i32";
    case int64:
      return "i64";
    case uint32:
      return "u32";
    case uint64:
      return "u64";
    default:
      return {};
  }
}

std::string gather_shader_name(Dtype value_dtype, Dtype index_dtype) {
  auto value_suffix = dtype_suffix(value_dtype);
  auto index_suffix = gather_index_suffix(index_dtype);
  if (value_suffix.empty() || index_suffix.empty()) {
    return {};
  }
  return "gather_" + value_suffix + "_" + index_suffix;
}

std::string gather_axis_shader_name(Dtype value_dtype, Dtype index_dtype) {
  auto value_suffix = dtype_suffix(value_dtype);
  auto index_suffix = gather_index_suffix(index_dtype);
  if (value_suffix.empty() || index_suffix.empty()) {
    return {};
  }
  return "gather_axis_" + value_suffix + "_" + index_suffix;
}

bool is_supported_elementwise_layout(const array& arr) {
  if (arr.ndim() > 4 || !arr.flags().row_contiguous || arr.offset() != 0) {
    return false;
  }
  if (arr.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  for (auto dim : arr.shape()) {
    if (dim < 0 || dim > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  for (auto stride : arr.strides()) {
    if (stride < 0 || stride > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool is_supported_unary_layout(const array& arr) {
  if (arr.ndim() > 4 || arr.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  if (arr.offset() < 0 || arr.offset() > 0xFFFF) {
    return false;
  }
  for (auto dim : arr.shape()) {
    if (dim < 0 || dim > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  for (auto stride : arr.strides()) {
    if (stride < 0 || stride > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool is_supported_generic_unary_layout(const array& arr) {
  return arr.flags().row_contiguous && arr.offset() == 0 &&
      arr.size() <= std::numeric_limits<uint32_t>::max();
}

int normalize_axis(int axis, int ndim) {
  if (axis < 0) {
    axis += ndim;
  }
  return axis;
}

bool normalize_unique_axes(
    const std::vector<int>& axes,
    int ndim,
    std::vector<int>& normalized_axes) {
  normalized_axes.clear();
  normalized_axes.reserve(axes.size());
  for (int axis : axes) {
    int normalized_axis = normalize_axis(axis, ndim);
    if (normalized_axis < 0 || normalized_axis >= ndim) {
      return false;
    }
    if (std::find(
            normalized_axes.begin(), normalized_axes.end(), normalized_axis) !=
        normalized_axes.end()) {
      return false;
    }
    normalized_axes.push_back(normalized_axis);
  }
  return !normalized_axes.empty();
}

bool has_keepdims_axis_shape(const array& in, const array& out, int axis) {
  if (in.ndim() != out.ndim()) {
    return false;
  }

  for (int i = 0; i < in.ndim(); ++i) {
    const int64_t expected = (i == axis) ? 1 : in.shape(i);
    if (out.shape(i) != expected) {
      return false;
    }
  }
  return true;
}

bool has_squeezed_axis_shape(const array& in, const array& out, int axis) {
  if (in.ndim() - 1 != out.ndim()) {
    return false;
  }

  int out_dim = 0;
  for (int in_dim = 0; in_dim < in.ndim(); ++in_dim) {
    if (in_dim == axis) {
      continue;
    }
    if (out.shape(out_dim) != in.shape(in_dim)) {
      return false;
    }
    out_dim++;
  }
  return true;
}

Shape keepdims_shape_for_axis(const array& in, int axis) {
  auto shape = in.shape();
  shape[axis] = 1;
  return shape;
}

Shape keepdims_shape_for_axes(const array& in, const std::vector<int>& axes) {
  auto shape = in.shape();
  for (int axis : axes) {
    shape[axis] = 1;
  }
  return shape;
}

bool has_keepdims_axes_shape(
    const array& in,
    const array& out,
    const std::vector<int>& axes) {
  return out.shape() == keepdims_shape_for_axes(in, axes);
}

bool has_squeezed_axes_shape(
    const array& in,
    const array& out,
    const std::vector<int>& axes) {
  if (out.ndim() != in.ndim() - axes.size()) {
    return false;
  }

  std::vector<bool> reduce_dims(in.ndim(), false);
  for (int axis : axes) {
    reduce_dims[axis] = true;
  }

  int out_dim = 0;
  for (int in_dim = 0; in_dim < in.ndim(); ++in_dim) {
    if (reduce_dims[in_dim]) {
      continue;
    }
    if (out.shape(out_dim) != in.shape(in_dim)) {
      return false;
    }
    out_dim++;
  }
  return true;
}

} // namespace mlx::core
