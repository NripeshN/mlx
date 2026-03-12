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

#define MLX_VK_BINARY_CASE(OP, A, B, O, RTE, SHADER)              \
  if (op == BinaryShaderOp::OP && a_dtype == A && b_dtype == B && \
      out_dtype == O && rte == RTE) {                             \
    return vulkan::StaticShaderId::SHADER;                        \
  }

#define MLX_VK_FLOAT_BINARY_CASES(OP, PREFIX)                        \
  MLX_VK_BINARY_CASE(                                                \
      OP, float32, float32, float32, false, PREFIX##_f32_f32_f32)    \
  MLX_VK_BINARY_CASE(                                                \
      OP, float32, float32, float32, true, PREFIX##_f32_f32_f32_rte) \
  MLX_VK_BINARY_CASE(                                                \
      OP, float32, float32, float16, false, PREFIX##_f32_f32_f16)    \
  MLX_VK_BINARY_CASE(                                                \
      OP, float32, float32, float16, true, PREFIX##_f32_f32_f16_rte) \
  MLX_VK_BINARY_CASE(                                                \
      OP, float32, float16, float32, false, PREFIX##_f32_f16_f32)    \
  MLX_VK_BINARY_CASE(                                                \
      OP, float32, float16, float32, true, PREFIX##_f32_f16_f32_rte) \
  MLX_VK_BINARY_CASE(                                                \
      OP, float32, float16, float16, false, PREFIX##_f32_f16_f16)    \
  MLX_VK_BINARY_CASE(                                                \
      OP, float32, float16, float16, true, PREFIX##_f32_f16_f16_rte) \
  MLX_VK_BINARY_CASE(                                                \
      OP, float16, float32, float32, false, PREFIX##_f16_f32_f32)    \
  MLX_VK_BINARY_CASE(                                                \
      OP, float16, float32, float32, true, PREFIX##_f16_f32_f32_rte) \
  MLX_VK_BINARY_CASE(                                                \
      OP, float16, float32, float16, false, PREFIX##_f16_f32_f16)    \
  MLX_VK_BINARY_CASE(                                                \
      OP, float16, float32, float16, true, PREFIX##_f16_f32_f16_rte) \
  MLX_VK_BINARY_CASE(                                                \
      OP, float16, float16, float32, false, PREFIX##_f16_f16_f32)    \
  MLX_VK_BINARY_CASE(                                                \
      OP, float16, float16, float32, true, PREFIX##_f16_f16_f32_rte) \
  MLX_VK_BINARY_CASE(                                                \
      OP, float16, float16, float16, false, PREFIX##_f16_f16_f16)    \
  MLX_VK_BINARY_CASE(                                                \
      OP, float16, float16, float16, true, PREFIX##_f16_f16_f16_rte)

#define MLX_VK_INTEGER_BINARY_CASES(OP, PREFIX)                               \
  MLX_VK_BINARY_CASE(OP, int32, int32, int32, false, PREFIX##_i32_i32_i32)    \
  MLX_VK_BINARY_CASE(OP, int64, int64, int64, false, PREFIX##_i64_i64_i64)    \
  MLX_VK_BINARY_CASE(OP, uint32, uint32, uint32, false, PREFIX##_u32_u32_u32) \
  MLX_VK_BINARY_CASE(OP, uint64, uint64, uint64, false, PREFIX##_u64_u64_u64)

std::optional<vulkan::StaticShaderId> binary_shader_id(
    BinaryShaderOp op,
    Dtype a_dtype,
    Dtype b_dtype,
    Dtype out_dtype,
    bool rte) {
  MLX_VK_FLOAT_BINARY_CASES(Add, add);
  MLX_VK_FLOAT_BINARY_CASES(Divide, div);
  MLX_VK_FLOAT_BINARY_CASES(Maximum, maximum);
  MLX_VK_FLOAT_BINARY_CASES(Minimum, minimum);
  MLX_VK_FLOAT_BINARY_CASES(Multiply, mul);
  MLX_VK_FLOAT_BINARY_CASES(Subtract, sub);

  MLX_VK_INTEGER_BINARY_CASES(Add, add);
  MLX_VK_INTEGER_BINARY_CASES(Divide, div);
  MLX_VK_INTEGER_BINARY_CASES(Maximum, maximum);
  MLX_VK_INTEGER_BINARY_CASES(Minimum, minimum);
  MLX_VK_INTEGER_BINARY_CASES(Multiply, mul);
  MLX_VK_INTEGER_BINARY_CASES(Subtract, sub);

  MLX_VK_BINARY_CASE(
      GreaterEqual, float32, float32, uint8, false, greater_equal_f32_f32_u8);
  MLX_VK_BINARY_CASE(
      GreaterEqual, float32, float16, uint8, false, greater_equal_f32_f16_u8);
  MLX_VK_BINARY_CASE(
      GreaterEqual, float16, float32, uint8, false, greater_equal_f16_f32_u8);
  MLX_VK_BINARY_CASE(
      GreaterEqual, float16, float16, uint8, false, greater_equal_f16_f16_u8);
  MLX_VK_BINARY_CASE(
      GreaterEqual, int32, int32, uint8, false, greater_equal_i32_i32_u8);
  MLX_VK_BINARY_CASE(
      GreaterEqual, int64, int64, uint8, false, greater_equal_i64_i64_u8);
  MLX_VK_BINARY_CASE(
      GreaterEqual, uint32, uint32, uint8, false, greater_equal_u32_u32_u8);
  MLX_VK_BINARY_CASE(
      GreaterEqual, uint64, uint64, uint8, false, greater_equal_u64_u64_u8);

  MLX_VK_BINARY_CASE(Maximum, uint32, uint32, uint8, false, maximum_u32_u32_u8);
  return std::nullopt;
}

#undef MLX_VK_INTEGER_BINARY_CASES
#undef MLX_VK_FLOAT_BINARY_CASES
#undef MLX_VK_BINARY_CASE

#define MLX_VK_GENERIC_UNARY_CASE(OP, DTYPE, RTE, SHADER)               \
  if (op == GenericUnaryShaderOp::OP && dtype == DTYPE && rte == RTE) { \
    return vulkan::StaticShaderId::SHADER;                              \
  }

std::optional<vulkan::StaticShaderId>
generic_unary_shader_id(GenericUnaryShaderOp op, Dtype dtype, bool rte) {
  MLX_VK_GENERIC_UNARY_CASE(Abs, float32, false, abs_f32);
  MLX_VK_GENERIC_UNARY_CASE(Abs, float16, false, abs_f16);
  MLX_VK_GENERIC_UNARY_CASE(Ceil, float32, false, ceil_f32);
  MLX_VK_GENERIC_UNARY_CASE(Ceil, float16, false, ceil_f16);
  MLX_VK_GENERIC_UNARY_CASE(Exp, float32, false, exp_f32);
  MLX_VK_GENERIC_UNARY_CASE(Exp, float16, false, exp_f16);
  MLX_VK_GENERIC_UNARY_CASE(Exp, float16, true, exp_f16_rte);
  MLX_VK_GENERIC_UNARY_CASE(Floor, float32, false, floor_f32);
  MLX_VK_GENERIC_UNARY_CASE(Floor, float16, false, floor_f16);
  MLX_VK_GENERIC_UNARY_CASE(Negative, float32, false, neg_f32);
  MLX_VK_GENERIC_UNARY_CASE(Negative, float16, false, neg_f16);
  MLX_VK_GENERIC_UNARY_CASE(Round, float32, false, round_f32);
  MLX_VK_GENERIC_UNARY_CASE(Round, float16, false, round_f16);
  MLX_VK_GENERIC_UNARY_CASE(Sigmoid, float32, false, sigmoid_f32);
  MLX_VK_GENERIC_UNARY_CASE(Sigmoid, float16, false, sigmoid_f16);
  MLX_VK_GENERIC_UNARY_CASE(Tanh, float32, false, tanh_f32);
  MLX_VK_GENERIC_UNARY_CASE(Tanh, float16, false, tanh_f16);
  return std::nullopt;
}

#undef MLX_VK_GENERIC_UNARY_CASE

#define MLX_VK_UNARY_CASE(OP, DTYPE, SHADER)       \
  if (op == UnaryShaderOp::OP && dtype == DTYPE) { \
    return vulkan::StaticShaderId::SHADER;         \
  }

std::optional<vulkan::StaticShaderId> unary_shader_id(
    UnaryShaderOp op,
    Dtype dtype) {
  MLX_VK_UNARY_CASE(Cos, float32, cos_f32);
  MLX_VK_UNARY_CASE(Erf, float32, erf_f32);
  MLX_VK_UNARY_CASE(ErfInv, float32, erfinv_f32);
  MLX_VK_UNARY_CASE(Log, float32, log_f32);
  MLX_VK_UNARY_CASE(Log, float16, log_f16_rte);
  MLX_VK_UNARY_CASE(Sin, float32, sin_f32);
  MLX_VK_UNARY_CASE(Square, float32, sqr_f32);
  MLX_VK_UNARY_CASE(Square, float16, sqr_f16);
  MLX_VK_UNARY_CASE(Square, bfloat16, sqr_f32);
  MLX_VK_UNARY_CASE(Square, complex64, sqr_c64);
  MLX_VK_UNARY_CASE(Sqrt, float32, sqrt_f32);
  MLX_VK_UNARY_CASE(Sqrt, float16, sqrt_f16);
  MLX_VK_UNARY_CASE(Sqrt, bfloat16, sqrt_f32);
  MLX_VK_UNARY_CASE(Sqrt, complex64, sqrt_c64);
  MLX_VK_UNARY_CASE(Rsqrt, float32, rsqrt_f32);
  MLX_VK_UNARY_CASE(Rsqrt, float16, rsqrt_f16);
  MLX_VK_UNARY_CASE(Rsqrt, bfloat16, rsqrt_f32);
  MLX_VK_UNARY_CASE(Rsqrt, complex64, rsqrt_c64);
  return std::nullopt;
}

#undef MLX_VK_UNARY_CASE

#define MLX_VK_GATHER_CASE(VALUE, INDEX, SHADER)      \
  if (value_dtype == VALUE && index_dtype == INDEX) { \
    return vulkan::StaticShaderId::SHADER;            \
  }

std::optional<vulkan::StaticShaderId> gather_shader_id(
    Dtype value_dtype,
    Dtype index_dtype) {
  MLX_VK_GATHER_CASE(float32, int32, gather_f32_i32);
  MLX_VK_GATHER_CASE(float16, int32, gather_f16_i32);
  MLX_VK_GATHER_CASE(bfloat16, int32, gather_bf16_i32);
  MLX_VK_GATHER_CASE(int32, int32, gather_i32_i32);
  MLX_VK_GATHER_CASE(uint32, int32, gather_u32_i32);
  MLX_VK_GATHER_CASE(float32, int64, gather_f32_i64);
  MLX_VK_GATHER_CASE(float16, int64, gather_f16_i64);
  MLX_VK_GATHER_CASE(bfloat16, int64, gather_bf16_i64);
  MLX_VK_GATHER_CASE(int32, int64, gather_i32_i64);
  MLX_VK_GATHER_CASE(uint32, int64, gather_u32_i64);
  MLX_VK_GATHER_CASE(float32, uint32, gather_f32_u32);
  MLX_VK_GATHER_CASE(float16, uint32, gather_f16_u32);
  MLX_VK_GATHER_CASE(bfloat16, uint32, gather_bf16_u32);
  MLX_VK_GATHER_CASE(int32, uint32, gather_i32_u32);
  MLX_VK_GATHER_CASE(uint32, uint32, gather_u32_u32);
  MLX_VK_GATHER_CASE(float32, uint64, gather_f32_u64);
  MLX_VK_GATHER_CASE(float16, uint64, gather_f16_u64);
  MLX_VK_GATHER_CASE(bfloat16, uint64, gather_bf16_u64);
  MLX_VK_GATHER_CASE(int32, uint64, gather_i32_u64);
  MLX_VK_GATHER_CASE(uint32, uint64, gather_u32_u64);
  return std::nullopt;
}

std::optional<vulkan::StaticShaderId> gather_axis_shader_id(
    Dtype value_dtype,
    Dtype index_dtype) {
  MLX_VK_GATHER_CASE(float32, int32, gather_axis_f32_i32);
  MLX_VK_GATHER_CASE(float16, int32, gather_axis_f16_i32);
  MLX_VK_GATHER_CASE(bfloat16, int32, gather_axis_bf16_i32);
  MLX_VK_GATHER_CASE(int32, int32, gather_axis_i32_i32);
  MLX_VK_GATHER_CASE(uint32, int32, gather_axis_u32_i32);
  MLX_VK_GATHER_CASE(float32, int64, gather_axis_f32_i64);
  MLX_VK_GATHER_CASE(float16, int64, gather_axis_f16_i64);
  MLX_VK_GATHER_CASE(bfloat16, int64, gather_axis_bf16_i64);
  MLX_VK_GATHER_CASE(int32, int64, gather_axis_i32_i64);
  MLX_VK_GATHER_CASE(uint32, int64, gather_axis_u32_i64);
  MLX_VK_GATHER_CASE(float32, uint32, gather_axis_f32_u32);
  MLX_VK_GATHER_CASE(float16, uint32, gather_axis_f16_u32);
  MLX_VK_GATHER_CASE(bfloat16, uint32, gather_axis_bf16_u32);
  MLX_VK_GATHER_CASE(int32, uint32, gather_axis_i32_u32);
  MLX_VK_GATHER_CASE(uint32, uint32, gather_axis_u32_u32);
  MLX_VK_GATHER_CASE(float32, uint64, gather_axis_f32_u64);
  MLX_VK_GATHER_CASE(float16, uint64, gather_axis_f16_u64);
  MLX_VK_GATHER_CASE(bfloat16, uint64, gather_axis_bf16_u64);
  MLX_VK_GATHER_CASE(int32, uint64, gather_axis_i32_u64);
  MLX_VK_GATHER_CASE(uint32, uint64, gather_axis_u32_u64);
  return std::nullopt;
}

std::optional<vulkan::StaticShaderId> scatter_axis_shader_id(
    Dtype value_dtype,
    Dtype index_dtype) {
  MLX_VK_GATHER_CASE(float32, int32, scatter_axis_f32_i32);
  MLX_VK_GATHER_CASE(float16, int32, scatter_axis_f16_i32);
  MLX_VK_GATHER_CASE(bfloat16, int32, scatter_axis_bf16_i32);
  MLX_VK_GATHER_CASE(int32, int32, scatter_axis_i32_i32);
  MLX_VK_GATHER_CASE(uint32, int32, scatter_axis_u32_i32);
  MLX_VK_GATHER_CASE(float32, int64, scatter_axis_f32_i64);
  MLX_VK_GATHER_CASE(float16, int64, scatter_axis_f16_i64);
  MLX_VK_GATHER_CASE(bfloat16, int64, scatter_axis_bf16_i64);
  MLX_VK_GATHER_CASE(int32, int64, scatter_axis_i32_i64);
  MLX_VK_GATHER_CASE(uint32, int64, scatter_axis_u32_i64);
  MLX_VK_GATHER_CASE(float32, uint32, scatter_axis_f32_u32);
  MLX_VK_GATHER_CASE(float16, uint32, scatter_axis_f16_u32);
  MLX_VK_GATHER_CASE(bfloat16, uint32, scatter_axis_bf16_u32);
  MLX_VK_GATHER_CASE(int32, uint32, scatter_axis_i32_u32);
  MLX_VK_GATHER_CASE(uint32, uint32, scatter_axis_u32_u32);
  MLX_VK_GATHER_CASE(float32, uint64, scatter_axis_f32_u64);
  MLX_VK_GATHER_CASE(float16, uint64, scatter_axis_f16_u64);
  MLX_VK_GATHER_CASE(bfloat16, uint64, scatter_axis_bf16_u64);
  MLX_VK_GATHER_CASE(int32, uint64, scatter_axis_i32_u64);
  MLX_VK_GATHER_CASE(uint32, uint64, scatter_axis_u32_u64);
  return std::nullopt;
}

#undef MLX_VK_GATHER_CASE

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
