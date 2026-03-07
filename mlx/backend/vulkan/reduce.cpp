// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

namespace {

bool try_eval_reduce_sum_rows_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    Stream s) {
  if (inputs.size() != 1 || reduce_type != Reduce::Sum || axes.empty()) {
    return false;
  }

  array in = inputs[0];
  const bool f32_io = in.dtype() == float32 && out.dtype() == float32;
  const bool f16_io = in.dtype() == float16 && out.dtype() == float16;
  const bool bf16_io = in.dtype() == bfloat16 && out.dtype() == bfloat16;
  if (!f32_io && !f16_io && !bf16_io) {
    return false;
  }

  const bool use_f32_staging_io = f16_io || bf16_io;
  if (use_f32_staging_io) {
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }

  array reduce_out_target =
      use_f32_staging_io ? array(out.shape(), float32, nullptr, {}) : out;

  if (in.ndim() == 0 || in.ndim() > 4 || reduce_out_target.ndim() > 4) {
    return false;
  }

  std::vector<int> normalized_axes;
  if (!normalize_unique_axes(axes, in.ndim(), normalized_axes)) {
    return false;
  }

  const bool out_is_keepdims =
      has_keepdims_axes_shape(in, reduce_out_target, normalized_axes);
  const bool out_is_squeezed =
      has_squeezed_axes_shape(in, reduce_out_target, normalized_axes);
  if (!out_is_keepdims && !out_is_squeezed) {
    return false;
  }

  array reduced = in;
  for (int axis : normalized_axes) {
    array in_kernel = reduced;
    if (axis != reduced.ndim() - 1) {
      in_kernel = swapaxes_in_eval(reduced, axis, reduced.ndim() - 1);
    }

    if (!in_kernel.flags().row_contiguous ||
        !is_supported_unary_layout(in_kernel)) {
      in_kernel = contiguous_copy_gpu(in_kernel, s);
    }

    array kernel_out(
        keepdims_shape_for_axis(in_kernel, in_kernel.ndim() - 1),
        in_kernel.dtype(),
        nullptr,
        {});

    const bool staged_output = !kernel_out.flags().row_contiguous ||
        !is_supported_unary_layout(kernel_out);
    array out_work = staged_output
        ? array(kernel_out.shape(), kernel_out.dtype(), nullptr, {})
        : kernel_out;

    out_work.set_data(allocator::malloc(out_work.nbytes()));
    if (out_work.size() == 0) {
      if (staged_output) {
        copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
      }
    } else {
      try {
        auto command_buffer = vulkan::begin_command_recording(s.index);
        vulkan::dispatch_sum_rows_op(
            in_kernel, out_work, "sum_rows_f32", command_buffer, s, 1.0f);
        vulkan::end_command_recording(s.index);
      } catch (const std::runtime_error&) {
        return false;
      }
      if (staged_output) {
        copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
      }
    }

    reduced = (axis == reduced.ndim() - 1)
        ? kernel_out
        : swapaxes_in_eval(kernel_out, axis, reduced.ndim() - 1);
  }

  if (out_is_squeezed) {
    auto squeezed = reshape_in_eval(reduced, reduce_out_target.shape(), s);
    copy_gpu(squeezed, reduce_out_target, CopyType::GeneralGeneral, s);
  } else {
    copy_gpu(reduced, reduce_out_target, CopyType::GeneralGeneral, s);
  }

  if (use_f32_staging_io) {
    copy_gpu(reduce_out_target, out, CopyType::General, s);
  }
  return true;
}

bool try_eval_arg_reduce_vulkan(
    const std::vector<array>& inputs,
    array& out,
    ArgReduce::ReduceType reduce_type,
    int axis,
    Stream s) {
  if (inputs.size() != 1 || reduce_type != ArgReduce::ArgMax) {
    return false;
  }

  array in = inputs[0];
  const bool f32_input = in.dtype() == float32;
  const bool f16_input = in.dtype() == float16;
  const bool bf16_input = in.dtype() == bfloat16;
  if (in.ndim() == 0 || (!f32_input && !f16_input && !bf16_input) ||
      out.dtype() != uint32) {
    return false;
  }

  if (f16_input || bf16_input) {
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }

  axis = normalize_axis(axis, in.ndim());

  const bool out_is_keepdims = has_keepdims_axis_shape(in, out, axis);
  const bool out_is_squeezed = has_squeezed_axis_shape(in, out, axis);
  if (!out_is_keepdims && !out_is_squeezed) {
    return false;
  }

  array in_kernel = in;
  if (axis != in.ndim() - 1) {
    in_kernel = swapaxes_in_eval(in, axis, in.ndim() - 1);
  }

  if (in_kernel.size() > std::numeric_limits<uint32_t>::max() ||
      out.size() > std::numeric_limits<uint32_t>::max() ||
      in_kernel.shape(in_kernel.ndim() - 1) >
          std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  if (!in_kernel.flags().row_contiguous || in_kernel.offset() != 0 ||
      !is_supported_unary_layout(in_kernel)) {
    in_kernel = contiguous_copy_gpu(in_kernel, s);
  }

  array kernel_out(
      keepdims_shape_for_axis(in_kernel, in_kernel.ndim() - 1),
      out.dtype(),
      nullptr,
      {});

  const bool staged_output = !kernel_out.flags().row_contiguous ||
      kernel_out.offset() != 0 || !is_supported_unary_layout(kernel_out);
  array out_work = staged_output
      ? array(kernel_out.shape(), kernel_out.dtype(), nullptr, {})
      : kernel_out;

  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
    }
    if (out_is_squeezed) {
      auto squeezed = reshape_in_eval(kernel_out, out.shape(), s);
      copy_gpu(squeezed, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_argmax_op(
        in_kernel, out_work, "argmax_f32", command_buffer, s);
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
    }

    array restored_keepdims = kernel_out;
    if (axis != in.ndim() - 1) {
      restored_keepdims = swapaxes_in_eval(kernel_out, axis, in.ndim() - 1);
    }

    if (out_is_squeezed) {
      auto squeezed = reshape_in_eval(restored_keepdims, out.shape(), s);
      copy_gpu(squeezed, out, CopyType::GeneralGeneral, s);
    } else {
      copy_gpu(restored_keepdims, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

} // namespace

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis] = state();
  if (try_eval_arg_reduce_vulkan(inputs, out, reduce_type, axis, stream())) {
    return;
  }
  eval_cpu_fallback_on_stream<ArgReduce>(
      inputs, out, stream(), reduce_type, axis);
}

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axes] = state();
  if (try_eval_reduce_sum_rows_vulkan(
          inputs, out, reduce_type, axes, stream())) {
    return;
  }
  eval_cpu_fallback_on_stream<Reduce>(inputs, out, stream(), reduce_type, axes);
}

} // namespace mlx::core
