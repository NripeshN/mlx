// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

namespace {

bool try_eval_scatter_axis_vulkan(
    const std::vector<array>& inputs,
    array& out,
    ScatterAxis::ReduceType reduce_type,
    int axis,
    Stream s) {
  if (inputs.size() != 3) {
    return false;
  }

  if (reduce_type != ScatterAxis::None) {
    trace_vulkan_unsupported(
        "ScatterAxis", "only non-reducing put_along_axis is implemented");
    return false;
  }

  const auto& src = inputs[0];
  array idx = inputs[1];
  array upd = inputs[2];

  if (src.ndim() == 0 || idx.ndim() != src.ndim() ||
      upd.shape() != idx.shape() || out.shape() != src.shape() ||
      out.dtype() != src.dtype()) {
    return false;
  }

  axis = normalize_axis(axis, src.ndim());
  if (axis < 0 || axis >= src.ndim()) {
    trace_vulkan_unsupported("ScatterAxis", "axis is out of range");
    return false;
  }

  const auto shader_name = scatter_axis_shader_name(out.dtype(), idx.dtype());
  if (shader_name.empty()) {
    trace_vulkan_unsupported(
        "ScatterAxis",
        "value/index dtype combination is not supported by Vulkan scatter_axis");
    return false;
  }

  if (!idx.flags().contiguous || idx.offset() != 0 ||
      idx.strides().back() != 1) {
    idx = contiguous_copy_gpu(idx, s);
  }
  if (!upd.flags().contiguous || upd.offset() != 0 ||
      upd.strides().back() != 1) {
    upd = contiguous_copy_gpu(upd, s);
  }

  uint32_t size_pre = 1;
  for (int i = 0; i < axis; ++i) {
    size_pre = checked_mul_u32(
        size_pre,
        checked_u32_size(src.shape(i), "scatter_axis size_pre"),
        "scatter_axis size_pre");
  }
  const uint32_t size_axis =
      checked_u32_size(src.shape(axis), "scatter_axis size_axis");
  uint32_t size_post = 1;
  for (int i = axis + 1; i < src.ndim(); ++i) {
    size_post = checked_mul_u32(
        size_post,
        checked_u32_size(src.shape(i), "scatter_axis size_post"),
        "scatter_axis size_post");
  }
  const uint32_t idx_axis_size =
      checked_u32_size(idx.shape(axis), "scatter_axis idx_axis_size");

  array out_work = out;
  const bool staged_output =
      !out.flags().contiguous || out.offset() != 0 || out.strides().back() != 1;
  if (staged_output) {
    out_work = array(out.shape(), out.dtype(), nullptr, {});
  }

  CopyType copy_type;
  if (src.data_size() == 1) {
    copy_type = CopyType::Scalar;
  } else if (src.flags().row_contiguous) {
    copy_type = CopyType::Vector;
  } else {
    copy_type = CopyType::General;
  }
  copy_gpu(src, out_work, copy_type, s);

  if (upd.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  array upd_flat = reshape_in_eval(
      upd,
      Shape{
          static_cast<int>(size_pre),
          static_cast<int>(idx_axis_size),
          static_cast<int>(size_post)},
      s);
  array idx_flat = reshape_in_eval(
      idx,
      Shape{
          static_cast<int>(size_pre),
          static_cast<int>(idx_axis_size),
          static_cast<int>(size_post)},
      s);
  array out_flat = reshape_in_eval(
      out_work,
      Shape{
          static_cast<int>(size_pre),
          static_cast<int>(size_axis),
          static_cast<int>(size_post)},
      s);

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_scatter_axis_op(
        upd_flat,
        idx_flat,
        out_flat,
        shader_name,
        command_buffer,
        s,
        size_pre,
        size_axis,
        size_post,
        idx_axis_size);
    vulkan::end_command_recording(s.index);

    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "scatter_axis_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("Scatter has no Vulkan implementation.");
}

void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis] = state();
  if (!try_eval_scatter_axis_vulkan(inputs, out, reduce_type, axis, stream())) {
    throw std::runtime_error(
        "ScatterAxis operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
