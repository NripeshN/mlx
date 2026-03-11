// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/ops.h"

namespace mlx::core {

namespace {

bool try_eval_gather_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::vector<int>& axes,
    const Shape& slice_sizes,
    Stream s) {
  if (inputs.size() < 2 || inputs.size() != axes.size() + 1) {
    return false;
  }

  if (axes.size() == 2) {
    const auto& src_input = inputs[0];
    array idx0 = inputs[1];
    array idx1 = inputs[2];
    if (src_input.ndim() == 0 || out.dtype() != src_input.dtype() ||
        idx0.shape() != idx1.shape()) {
      return false;
    }

    const int axis0 = normalize_axis(axes[0], src_input.ndim());
    const int axis1 = normalize_axis(axes[1], src_input.ndim());
    if (axis0 < 0 || axis1 < 0 || axis0 >= src_input.ndim() ||
        axis1 >= src_input.ndim() || axis0 == axis1) {
      return false;
    }

    for (int i = 0; i < src_input.ndim(); ++i) {
      const int64_t expected =
          (i == axis0 || i == axis1) ? 1 : src_input.shape(i);
      if (slice_sizes[i] != expected) {
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

    std::vector<int> perm = {axis0, axis1};
    for (int i = 0; i < src_input.ndim(); ++i) {
      if (i != axis0 && i != axis1) {
        perm.push_back(i);
      }
    }

    auto transposed = transpose(src_input, perm, s);
    if (!transposed.flags().contiguous || transposed.offset() != 0 ||
        transposed.strides().back() != 1) {
      transposed = contiguous_copy_gpu(transposed, s);
    }
    eval(transposed);

    const uint32_t dim1 =
        checked_u32_size(src_input.shape(axis1), "gather paired axis size");
    const uint32_t index_count =
        checked_u32_size(idx0.size(), "gather paired index count");
    const uint32_t flat_axis_size = checked_mul_u32(
        checked_u32_size(src_input.shape(axis0), "gather paired axis0 size"),
        dim1,
        "gather paired flat axis size");

    uint32_t slice_size = 1;
    for (int i = 2; i < static_cast<int>(perm.size()); ++i) {
      slice_size = checked_mul_u32(
          slice_size,
          checked_u32_size(
              src_input.shape(perm[i]), "gather paired slice size"),
          "gather paired slice size");
    }

    array src_2d = reshape_in_eval(
        transposed,
        Shape{static_cast<int>(flat_axis_size), static_cast<int>(slice_size)},
        s);
    array linear_idx =
        add(multiply(idx0, array(dim1, idx0.dtype()), s), idx1, s);
    eval(linear_idx);
    if (!linear_idx.flags().contiguous || linear_idx.offset() != 0 ||
        linear_idx.strides().back() != 1) {
      linear_idx = contiguous_copy_gpu(linear_idx, s);
    }
    eval(linear_idx);
    array idx_1d =
        reshape_in_eval(linear_idx, Shape{static_cast<int>(index_count)}, s);
    array out_work = out;
    const bool staged_output = !out.flags().contiguous || out.offset() != 0 ||
        out.strides().back() != 1;
    if (staged_output) {
      out_work = array(out.shape(), out.dtype(), nullptr, {});
    }
    out_work.set_data(allocator::malloc(out_work.nbytes()));
    array out_2d = reshape_in_eval(
        out_work,
        Shape{static_cast<int>(index_count), static_cast<int>(slice_size)},
        s);

    const auto shader_name =
        gather_shader_name(src_input.dtype(), idx_1d.dtype());
    if (shader_name.empty()) {
      return false;
    }

    try {
      auto command_buffer = vulkan::begin_command_recording(s.index);
      vulkan::dispatch_gather_op(
          src_2d,
          idx_1d,
          out_2d,
          shader_name,
          command_buffer,
          s,
          slice_size,
          flat_axis_size,
          index_count);
      vulkan::end_command_recording(s.index);

      if (staged_output) {
        copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
      }
      return true;
    } catch (const std::runtime_error& e) {
      if (trace_fallback_enabled()) {
        std::ostringstream oss;
        oss << "gather_dispatch_failed reason=" << e.what();
        trace_fallback(oss.str());
      }
      return false;
    }
  }

  if (axes.size() != 1) {
    return false;
  }

  const auto& src_input = inputs[0];
  array idx = inputs[1];
  if (src_input.ndim() == 0 || out.dtype() != src_input.dtype()) {
    return false;
  }
  const int axis = normalize_axis(axes[0], src_input.ndim());
  if (axis < 0 || axis >= src_input.ndim()) {
    trace_vulkan_unsupported("Gather", "axis is out of range");
    return false;
  }
  for (int i = 0; i < src_input.ndim(); ++i) {
    const int64_t expected = (i == axis) ? 1 : src_input.shape(i);
    if (slice_sizes[i] != expected) {
      trace_vulkan_unsupported(
          "Gather", "only take-like single-axis gathers are supported");
      return false;
    }
  }

  const auto shader_name = gather_shader_name(src_input.dtype(), idx.dtype());
  if (shader_name.empty()) {
    trace_vulkan_unsupported(
        "Gather",
        "value/index dtype combination is not supported by Vulkan gather");
    return false;
  }

  if (!idx.flags().contiguous || idx.offset() != 0 ||
      idx.strides().back() != 1) {
    idx = contiguous_copy_gpu(idx, s);
  }

  const uint32_t axis_size =
      checked_u32_size(src_input.shape(axis), "gather axis size");
  const uint32_t index_count =
      checked_u32_size(idx.size(), "gather index count");
  const uint32_t slice_size =
      checked_product_u32(slice_sizes, "gather slice size");

  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return true;
  }

  array src_2d(
      Shape{static_cast<int>(axis_size), static_cast<int>(slice_size)},
      src_input.dtype(),
      nullptr,
      {});
  if (axis == 0) {
    array src = src_input;
    if (!src.flags().contiguous || src.offset() != 0 ||
        src.strides().back() != 1) {
      src = contiguous_copy_gpu(src, s);
    }
    src_2d = reshape_in_eval(
        src,
        Shape{static_cast<int>(axis_size), static_cast<int>(slice_size)},
        s);
  } else {
    std::vector<int> perm(src_input.ndim());
    perm[0] = axis;
    int dst_axis = 1;
    for (int src_axis = 0; src_axis < src_input.ndim(); ++src_axis) {
      if (src_axis != axis) {
        perm[dst_axis++] = src_axis;
      }
    }
    auto transposed = transpose(src_input, perm, s);
    eval(transposed);
    copy_gpu(transposed, src_2d, CopyType::General, s);
  }
  array idx_1d = reshape_in_eval(idx, Shape{static_cast<int>(index_count)}, s);
  array out_2d(
      Shape{static_cast<int>(index_count), static_cast<int>(slice_size)},
      out.dtype(),
      nullptr,
      {});
  out_2d.set_data(allocator::malloc(out_2d.nbytes()));

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_gather_op(
        src_2d,
        idx_1d,
        out_2d,
        shader_name,
        command_buffer,
        s,
        slice_size,
        axis_size,
        index_count);
    vulkan::end_command_recording(s.index);

    array gathered = reshape_in_eval(out_2d, out.shape(), s);
    copy_gpu(gathered, out, CopyType::GeneralGeneral, s);
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "gather_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

bool try_eval_gather_axis_vulkan(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    Stream s) {
  if (inputs.size() != 2) {
    return false;
  }

  array src = inputs[0];
  array idx = inputs[1];
  if (src.ndim() == 0 || idx.ndim() != src.ndim() ||
      out.shape() != idx.shape() || out.dtype() != src.dtype()) {
    return false;
  }
  axis = normalize_axis(axis, src.ndim());
  if (axis < 0 || axis >= src.ndim()) {
    trace_vulkan_unsupported("GatherAxis", "axis is out of range");
    return false;
  }

  const auto shader_name = gather_axis_shader_name(src.dtype(), idx.dtype());
  if (shader_name.empty()) {
    trace_vulkan_unsupported(
        "GatherAxis",
        "value/index dtype combination is not supported by Vulkan gather_axis");
    return false;
  }

  if (!src.flags().contiguous || src.offset() != 0 ||
      src.strides().back() != 1) {
    src = contiguous_copy_gpu(src, s);
  }
  if (!idx.flags().contiguous || idx.offset() != 0 ||
      idx.strides().back() != 1) {
    idx = contiguous_copy_gpu(idx, s);
  }

  uint32_t size_pre = 1;
  for (int i = 0; i < axis; ++i) {
    size_pre = checked_mul_u32(
        size_pre,
        checked_u32_size(src.shape(i), "gather_axis size_pre"),
        "gather_axis size_pre");
  }
  const uint32_t size_axis =
      checked_u32_size(src.shape(axis), "gather_axis size_axis");
  uint32_t size_post = 1;
  for (int i = axis + 1; i < src.ndim(); ++i) {
    size_post = checked_mul_u32(
        size_post,
        checked_u32_size(src.shape(i), "gather_axis size_post"),
        "gather_axis size_post");
  }
  const uint32_t idx_axis_size =
      checked_u32_size(idx.shape(axis), "gather_axis idx_axis_size");

  array out_work = out;
  const bool staged_output =
      !out.flags().contiguous || out.offset() != 0 || out.strides().back() != 1;
  if (staged_output) {
    out_work = array(out.shape(), out.dtype(), nullptr, {});
  }

  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  array src_flat = reshape_in_eval(
      src,
      Shape{
          static_cast<int>(size_pre),
          static_cast<int>(size_axis),
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
          static_cast<int>(idx_axis_size),
          static_cast<int>(size_post)},
      s);

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_gather_axis_op(
        src_flat,
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
      oss << "gather_axis_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [axes, slice_sizes] = state();
  if (!try_eval_gather_vulkan(inputs, out, axes, slice_sizes, stream())) {
    throw std::runtime_error(
        "Gather operation failed on Vulkan (unsupported dtype or layout).");
  }
}

void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_gather_axis_vulkan(inputs, out, state(), stream())) {
    throw std::runtime_error(
        "GatherAxis operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
