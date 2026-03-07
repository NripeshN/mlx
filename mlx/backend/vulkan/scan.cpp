// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

namespace {

bool try_eval_scan_cumsum_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Scan::ReduceType reduce_type,
    int axis,
    bool reverse,
    bool inclusive,
    Stream s) {
  if (inputs.size() != 1 || reduce_type != Scan::Sum) {
    return false;
  }

  array in = inputs[0];
  if (in.ndim() == 0 || in.dtype() != float32 || out.dtype() != float32) {
    return false;
  }

  int normalized_axis = normalize_axis(axis, in.ndim());
  if (normalized_axis < 0 || normalized_axis >= in.ndim()) {
    return false;
  }

  array in_kernel = in;
  if (normalized_axis != in.ndim() - 1) {
    in_kernel = swapaxes_in_eval(in, normalized_axis, in.ndim() - 1);
  }

  auto reverse_last_axis_contiguous = [&](const array& arr) {
    Shape starts(arr.ndim(), 0);
    Shape strides(arr.ndim(), 1);
    starts[arr.ndim() - 1] = arr.shape(arr.ndim() - 1) - 1;
    strides[arr.ndim() - 1] = -1;
    array reversed_view(arr.shape(), arr.dtype(), nullptr, {});
    slice_gpu(arr, reversed_view, starts, strides, s);
    return contiguous_copy_gpu(reversed_view, s);
  };

  array scan_input =
      reverse ? reverse_last_axis_contiguous(in_kernel) : in_kernel;

  if (!scan_input.flags().contiguous || scan_input.offset() != 0 ||
      scan_input.strides().back() != 1 ||
      !is_supported_unary_layout(scan_input)) {
    scan_input = contiguous_copy_gpu(scan_input, s);
  }

  if (scan_input.shape() != out.shape()) {
    return false;
  }

  if (scan_input.size() > std::numeric_limits<uint32_t>::max() ||
      scan_input.shape(scan_input.ndim() - 1) >
          std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  array inclusive_out(scan_input.shape(), scan_input.dtype(), nullptr, {});
  inclusive_out.set_data(allocator::malloc(inclusive_out.nbytes()));
  if (inclusive_out.size() == 0) {
    copy_gpu(inclusive_out, out, CopyType::GeneralGeneral, s);
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);

    vulkan::dispatch_cumsum_op(
        scan_input, inclusive_out, "cumsum_f32", command_buffer, s);

    array scan_result = inclusive_out;
    if (!inclusive) {
      array exclusive_out(scan_input.shape(), scan_input.dtype(), nullptr, {});
      exclusive_out.set_data(allocator::malloc(exclusive_out.nbytes()));
      vulkan::dispatch_binary_op(
          inclusive_out,
          scan_input,
          exclusive_out,
          "sub_f32_f32_f32",
          command_buffer,
          s,
          vulkan::BinaryDispatchVariant::Standard);
      scan_result = exclusive_out;
    }

    vulkan::end_command_recording(s.index);

    array restored =
        reverse ? reverse_last_axis_contiguous(scan_result) : scan_result;
    if (normalized_axis != in.ndim() - 1) {
      restored = swapaxes_in_eval(restored, normalized_axis, in.ndim() - 1);
    }

    copy_gpu(restored, out, CopyType::GeneralGeneral, s);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

} // namespace

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis, reverse, inclusive] = state();
  if (try_eval_scan_cumsum_vulkan(
          inputs, out, reduce_type, axis, reverse, inclusive, stream())) {
    return;
  }
  eval_cpu_fallback_on_stream<Scan>(
      inputs, out, stream(), reduce_type, axis, reverse, inclusive);
}

} // namespace mlx::core
