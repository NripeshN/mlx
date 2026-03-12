// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/ops.h"

namespace mlx::core {

namespace {

std::vector<int> invert_perm(const std::vector<int>& perm) {
  std::vector<int> inv(perm.size());
  for (int i = 0; i < static_cast<int>(perm.size()); ++i) {
    inv[perm[i]] = i;
  }
  return inv;
}

bool try_eval_scatter_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Scatter::ReduceType reduce_type,
    const std::vector<int>& axes,
    Stream s) {
  if (inputs.size() < 3 || inputs.size() != axes.size() + 2) {
    return false;
  }

  if (reduce_type != Scatter::None || axes.size() != 2) {
    return false;
  }

  const auto& src = inputs[0];
  array idx0 = inputs[1];
  array idx1 = inputs[2];
  array upd = inputs.back();
  if (src.ndim() == 0 || out.shape() != src.shape() ||
      out.dtype() != src.dtype() || idx0.shape() != idx1.shape()) {
    return false;
  }

  const int axis0 = normalize_axis(axes[0], src.ndim());
  const int axis1 = normalize_axis(axes[1], src.ndim());
  if (axis0 < 0 || axis1 < 0 || axis0 >= src.ndim() || axis1 >= src.ndim() ||
      axis0 == axis1) {
    return false;
  }

  for (int i = 0; i < src.ndim(); ++i) {
    const int64_t expected = (i == axis0 || i == axis1) ? 1 : src.shape(i);
    if (upd.shape(i + idx0.ndim()) != expected) {
      return false;
    }
  }

  if (!idx0.flags().contiguous || idx0.offset() != 0 ||
      idx0.strides().back() != 1) {
    idx0 = contiguous_copy_gpu(idx0, s);
  }
  if (!idx1.flags().contiguous || idx1.offset() != 0 ||
      idx1.strides().back() != 1) {
    idx1 = contiguous_copy_gpu(idx1, s);
  }
  if (!upd.flags().contiguous || upd.offset() != 0 ||
      upd.strides().back() != 1) {
    upd = contiguous_copy_gpu(upd, s);
  }

  std::vector<int> perm = {axis0, axis1};
  for (int i = 0; i < src.ndim(); ++i) {
    if (i != axis0 && i != axis1) {
      perm.push_back(i);
    }
  }

  const uint32_t dim1 =
      checked_u32_size(src.shape(axis1), "scatter paired axis size");
  const uint32_t index_count =
      checked_u32_size(idx0.size(), "scatter paired index count");
  const uint32_t flat_axis_size = checked_mul_u32(
      checked_u32_size(src.shape(axis0), "scatter paired axis0 size"),
      dim1,
      "scatter paired flat axis size");

  uint32_t slice_size = 1;
  for (int i = 2; i < static_cast<int>(perm.size()); ++i) {
    slice_size = checked_mul_u32(
        slice_size,
        checked_u32_size(src.shape(perm[i]), "scatter paired slice size"),
        "scatter paired slice size");
  }

  array out_work(out.shape(), out.dtype(), nullptr, {});

  CopyType copy_type =
      src.flags().row_contiguous ? CopyType::Vector : CopyType::General;
  copy_gpu(src, out_work, copy_type, s);
  out_work.set_status(array::Status::available);

  auto transposed = transpose(out_work, perm, s);
  if (!transposed.flags().contiguous || transposed.offset() != 0 ||
      transposed.strides().back() != 1) {
    transposed = contiguous_copy_gpu(transposed, s);
  }
  eval(transposed);

  array out_flat = reshape_in_eval(
      transposed,
      Shape{1, static_cast<int>(flat_axis_size), static_cast<int>(slice_size)},
      s);
  array linear_idx = add(multiply(idx0, array(dim1, idx0.dtype()), s), idx1, s);
  eval(linear_idx);
  if (!linear_idx.flags().contiguous || linear_idx.offset() != 0 ||
      linear_idx.strides().back() != 1) {
    linear_idx = contiguous_copy_gpu(linear_idx, s);
  }
  eval(linear_idx);
  array idx_flat = reshape_in_eval(
      linear_idx, Shape{1, static_cast<int>(index_count), 1}, s);
  array upd_flat = reshape_in_eval(
      upd,
      Shape{1, static_cast<int>(index_count), static_cast<int>(slice_size)},
      s);

  const auto shader_id = scatter_axis_shader_id(out.dtype(), idx_flat.dtype());
  if (!shader_id.has_value()) {
    return false;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_scatter_axis_op(
        upd_flat,
        idx_flat,
        out_flat,
        *shader_id,
        command_buffer,
        s,
        1,
        flat_axis_size,
        slice_size,
        index_count);
    vulkan::end_command_recording(s.index);

    array restored = transpose(transposed, invert_perm(perm), s);
    eval(restored);
    copy_gpu(restored, out_work, CopyType::GeneralGeneral, s);
    copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    out.set_status(array::Status::available);
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "scatter_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

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

  const auto shader_id = scatter_axis_shader_id(out.dtype(), idx.dtype());
  if (!shader_id.has_value()) {
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
        *shader_id,
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
  auto [reduce_type, axes] = state();
  if (!try_eval_scatter_vulkan(inputs, out, reduce_type, axes, stream())) {
    throw std::runtime_error(
        "Scatter operation failed on Vulkan (unsupported dtype or layout).");
  }
}

void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis] = state();
  if (!try_eval_scatter_axis_vulkan(inputs, out, reduce_type, axis, stream())) {
    throw std::runtime_error(
        "ScatterAxis operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
