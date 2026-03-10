// Copyright © 2024 Apple Inc.

#pragma once

#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

namespace mlx::core {

inline std::vector<array> materialize_cpu_fallback_inputs(
    const std::vector<array>& inputs) {
  auto cpu_stream = default_stream(Device::cpu);
  std::vector<array> cpu_inputs;
  cpu_inputs.reserve(inputs.size());
  for (const auto& in : inputs) {
    cpu_inputs.push_back(astype(in, in.dtype(), cpu_stream));
  }
  eval(cpu_inputs);
  synchronize(cpu_stream);
  return cpu_inputs;
}

// Trace fallback utilities
bool trace_fallback_enabled();
void trace_fallback(const std::string& msg);
void trace_use_fallback(
    std::string_view primitive_name,
    Stream s,
    std::string_view reason,
    std::string_view details = {});
void trace_vulkan_unsupported(
    std::string_view primitive_name,
    std::string_view reason);

// Type/size checking utilities
uint32_t checked_u32_size(int64_t value, const char* name);
uint32_t checked_mul_u32(uint32_t a, uint32_t b, const char* name);
uint32_t checked_product_u32(const Shape& shape, const char* name);

// Layout checkers
bool is_vulkan_float_dtype(Dtype dtype);
std::string dtype_suffix(Dtype dtype);
std::string gather_index_suffix(Dtype dtype);
std::string gather_shader_name(Dtype value_dtype, Dtype index_dtype);
std::string gather_axis_shader_name(Dtype value_dtype, Dtype index_dtype);
bool is_supported_elementwise_layout(const array& arr);
bool is_supported_unary_layout(const array& arr);
bool is_supported_generic_unary_layout(const array& arr);

// Axis utilities
int normalize_axis(int axis, int ndim);
bool normalize_unique_axes(
    const std::vector<int>& axes,
    int ndim,
    std::vector<int>& normalized_axes);
bool has_keepdims_axis_shape(const array& in, const array& out, int axis);
bool has_squeezed_axis_shape(const array& in, const array& out, int axis);
Shape keepdims_shape_for_axis(const array& in, int axis);
Shape keepdims_shape_for_axes(const array& in, const std::vector<int>& axes);
bool has_keepdims_axes_shape(
    const array& in,
    const array& out,
    const std::vector<int>& axes);
bool has_squeezed_axes_shape(
    const array& in,
    const array& out,
    const std::vector<int>& axes);

// CPU fallback templates
template <typename Primitive, typename... Args>
void eval_cpu_fallback(
    const std::vector<array>& inputs,
    array& out,
    Args&&... args) {
  if (trace_fallback_enabled()) {
    std::ostringstream oss;
    oss << "primitive=" << typeid(Primitive).name() << " kind=unary"
        << " inputs=" << inputs.size() << " out_shape=" << out.shape();
    trace_fallback(oss.str());
  }
  auto cpu_stream = default_stream(Device::cpu);
  auto cpu_inputs = materialize_cpu_fallback_inputs(inputs);
  Primitive cpu_primitive(cpu_stream, std::forward<Args>(args)...);
  cpu_primitive.eval_cpu(cpu_inputs, out);
  synchronize(cpu_stream);
}

template <typename Primitive, typename... Args>
void eval_cpu_fallback_on_stream(
    const std::vector<array>& inputs,
    array& out,
    Stream stream,
    Args&&... args) {
  vulkan::ScopedSyncLabel sync_label(
      std::string("cpu_fallback:") + typeid(Primitive).name());
  ::mlx::core::gpu::synchronize(stream);
  eval_cpu_fallback<Primitive>(inputs, out, std::forward<Args>(args)...);
}

template <typename Primitive, typename... Args>
void eval_cpu_fallback_multi(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    Args&&... args) {
  if (trace_fallback_enabled()) {
    std::ostringstream oss;
    oss << "primitive=" << typeid(Primitive).name() << " kind=multi"
        << " inputs=" << inputs.size() << " outputs=" << outputs.size();
    trace_fallback(oss.str());
  }
  auto cpu_stream = default_stream(Device::cpu);
  auto cpu_inputs = materialize_cpu_fallback_inputs(inputs);
  Primitive cpu_primitive(cpu_stream, std::forward<Args>(args)...);
  cpu_primitive.eval_cpu(cpu_inputs, outputs);
  synchronize(cpu_stream);
}

template <typename Primitive, typename... Args>
void eval_cpu_fallback_multi_on_stream(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    Stream stream,
    Args&&... args) {
  vulkan::ScopedSyncLabel sync_label(
      std::string("cpu_fallback_multi:") + typeid(Primitive).name());
  ::mlx::core::gpu::synchronize(stream);
  eval_cpu_fallback_multi<Primitive>(
      inputs, outputs, std::forward<Args>(args)...);
}

template <typename T, typename = void>
struct is_tuple_like : std::false_type {};

template <typename T>
struct is_tuple_like<
    T,
    std::void_t<decltype(std::tuple_size<std::decay_t<T>>::value)>>
    : std::true_type {};

template <typename Primitive, typename State>
void eval_cpu_fallback_with_state(
    const std::vector<array>& inputs,
    array& out,
    State&& state) {
  if constexpr (is_tuple_like<State>::value) {
    std::apply(
        [&](auto&&... state_args) {
          eval_cpu_fallback<Primitive>(
              inputs, out, std::forward<decltype(state_args)>(state_args)...);
        },
        std::forward<State>(state));
  } else {
    eval_cpu_fallback<Primitive>(inputs, out, std::forward<State>(state));
  }
}

template <typename Primitive, typename State>
void eval_cpu_fallback_with_state_on_stream(
    const std::vector<array>& inputs,
    array& out,
    Stream stream,
    State&& state) {
  if constexpr (is_tuple_like<State>::value) {
    std::apply(
        [&](auto&&... state_args) {
          eval_cpu_fallback_on_stream<Primitive>(
              inputs,
              out,
              stream,
              std::forward<decltype(state_args)>(state_args)...);
        },
        std::forward<State>(state));
  } else {
    eval_cpu_fallback_on_stream<Primitive>(
        inputs, out, stream, std::forward<State>(state));
  }
}

template <typename Primitive, typename State>
void eval_cpu_fallback_multi_with_state(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    State&& state) {
  if constexpr (is_tuple_like<State>::value) {
    std::apply(
        [&](auto&&... state_args) {
          eval_cpu_fallback_multi<Primitive>(
              inputs,
              outputs,
              std::forward<decltype(state_args)>(state_args)...);
        },
        std::forward<State>(state));
  } else {
    eval_cpu_fallback_multi<Primitive>(
        inputs, outputs, std::forward<State>(state));
  }
}

template <typename Primitive, typename State>
void eval_cpu_fallback_multi_with_state_on_stream(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    Stream stream,
    State&& state) {
  if constexpr (is_tuple_like<State>::value) {
    std::apply(
        [&](auto&&... state_args) {
          eval_cpu_fallback_multi_on_stream<Primitive>(
              inputs,
              outputs,
              stream,
              std::forward<decltype(state_args)>(state_args)...);
        },
        std::forward<State>(state));
  } else {
    eval_cpu_fallback_multi_on_stream<Primitive>(
        inputs, outputs, stream, std::forward<State>(state));
  }
}

} // namespace mlx::core
