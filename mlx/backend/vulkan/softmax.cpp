// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/dtype_utils.h"

namespace mlx::core {

namespace {

bool is_supported_softmax_layout(const array& arr) {
  return arr.flags().contiguous && arr.offset() == 0 && arr.ndim() > 0 &&
      arr.strides().back() == 1;
}

array collapse_softmax_leading_dims(const array& arr, Stream s) {
  if (arr.ndim() <= 4) {
    return arr;
  }
  return flatten_in_eval(arr, 0, arr.ndim() - 4, s);
}

bool try_eval_softmax_vulkan(
    const std::vector<array>& inputs,
    array& out,
    bool /*precise*/,
    Stream s) {
  auto trace_softmax_failure = [&](std::string_view reason) {
    if (!trace_fallback_enabled()) {
      return;
    }
    std::ostringstream oss;
    oss << "softmax_vulkan_unsupported reason=" << reason
        << " inputs=" << inputs.size() << " out_shape=" << out.shape()
        << " out_dtype=" << dtype_to_string(out.dtype());
    if (!inputs.empty()) {
      oss << " in_shape=" << inputs[0].shape()
          << " in_dtype=" << dtype_to_string(inputs[0].dtype())
          << " in_ndim=" << inputs[0].ndim()
          << " in_offset=" << inputs[0].offset();
      if (inputs[0].ndim() > 0) {
        oss << " in_last_stride=" << inputs[0].strides().back()
            << " in_last_dim=" << inputs[0].shape(inputs[0].ndim() - 1);
      }
    }
    oss << " out_ndim=" << out.ndim() << " out_offset=" << out.offset();
    if (out.ndim() > 0) {
      oss << " out_last_stride=" << out.strides().back()
          << " out_last_dim=" << out.shape(out.ndim() - 1);
    }
    trace_fallback(oss.str());
  };

  if (inputs.size() != 1) {
    trace_softmax_failure("expected_single_input");
    return false;
  }

  array in = inputs[0];
  const bool f32_io = in.dtype() == float32 && out.dtype() == float32;
  const bool f16_io = in.dtype() == float16 && out.dtype() == float16;
  const bool bf16_io = in.dtype() == bfloat16 && out.dtype() == bfloat16;
  if (in.ndim() == 0 || (!f32_io && !f16_io && !bf16_io)) {
    trace_softmax_failure("unsupported_dtype_or_scalar_input");
    return false;
  }

  const bool use_f16_variant = f16_io;
  const bool use_f32_staging_io = f16_io || bf16_io;
  if (use_f32_staging_io) {
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }

  array softmax_out_target =
      use_f32_staging_io ? array(out.shape(), float32, nullptr, {}) : out;

  if (!is_supported_softmax_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }

  array softmax_out_storage = softmax_out_target;
  const bool staged_output = !is_supported_softmax_layout(softmax_out_target);
  if (staged_output) {
    softmax_out_storage = array(
        softmax_out_target.shape(), softmax_out_target.dtype(), nullptr, {});
  }

  set_unary_output_data(in, softmax_out_storage);

  array out_work = softmax_out_storage;

  if (in.shape() != out_work.shape()) {
    trace_softmax_failure("input_output_shape_mismatch");
    return false;
  }

  if (in.size() > std::numeric_limits<uint32_t>::max() ||
      out_work.size() > std::numeric_limits<uint32_t>::max() ||
      in.shape(in.ndim() - 1) > std::numeric_limits<uint32_t>::max()) {
    trace_softmax_failure("shape_exceeds_uint32_limits");
    return false;
  }

  const uint32_t row_width = static_cast<uint32_t>(in.shape(in.ndim() - 1));
  const bool use_large_softmax = row_width > 16384u;

  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(
          softmax_out_storage, softmax_out_target, CopyType::GeneralGeneral, s);
    }
    if (use_f32_staging_io) {
      array target_kernel =
          collapse_softmax_leading_dims(softmax_out_target, s);
      array out_kernel = collapse_softmax_leading_dims(out, s);
      copy_gpu(target_kernel, out_kernel, CopyType::General, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    if (use_large_softmax) {
      vulkan::dispatch_softmax_large_op(
          in,
          out_work,
          use_f16_variant ? vulkan::StaticShaderId::soft_max_large1_f32_f16
                          : vulkan::StaticShaderId::soft_max_large1_f32,
          use_f16_variant ? vulkan::StaticShaderId::soft_max_large2_f32_f16
                          : vulkan::StaticShaderId::soft_max_large2_f32,
          use_f16_variant ? vulkan::StaticShaderId::soft_max_large3_f32_f16
                          : vulkan::StaticShaderId::soft_max_large3_f32,
          command_buffer,
          s);
    } else {
      vulkan::dispatch_softmax_op(
          in,
          out_work,
          use_f16_variant ? vulkan::StaticShaderId::soft_max_f32_f16
                          : vulkan::StaticShaderId::soft_max_f32,
          command_buffer,
          s);
    }
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(
          softmax_out_storage, softmax_out_target, CopyType::GeneralGeneral, s);
    }
    if (use_f32_staging_io) {
      array target_kernel =
          collapse_softmax_leading_dims(softmax_out_target, s);
      array out_kernel = collapse_softmax_leading_dims(out, s);
      copy_gpu(target_kernel, out_kernel, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "softmax_dispatch_failed reason=" << e.what()
          << " row_width=" << row_width
          << " use_large_softmax=" << use_large_softmax
          << " staged_output=" << staged_output
          << " use_f32_staging_io=" << use_f32_staging_io;
      trace_fallback(oss.str());
    }
    return false;
  }
}

bool try_eval_logsumexp_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 1) {
    return false;
  }

  array in = inputs[0];
  if (in.ndim() == 0 || !is_vulkan_float_dtype(in.dtype()) ||
      out.dtype() != in.dtype()) {
    return false;
  }

  if (!in.flags().contiguous || in.offset() != 0 || in.strides().back() != 1 ||
      !is_supported_unary_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }

  array out_work = out;
  const bool staged_output = !out.flags().contiguous || out.offset() != 0 ||
      out.strides().back() != 1 || !is_supported_unary_layout(out);
  if (staged_output) {
    out_work = array(out.shape(), out.dtype(), nullptr, {});
  }

  set_unary_output_data(in, out_work);
  if (!is_supported_unary_layout(in) || !is_supported_unary_layout(out_work)) {
    return false;
  }

  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    const auto shader_id = out.dtype() == bfloat16
        ? vulkan::StaticShaderId::logsumexp_bf16
        : (out.dtype() == float16 ? vulkan::StaticShaderId::logsumexp_f16
                                  : vulkan::StaticShaderId::logsumexp_f32);
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_sum_rows_op(in, out_work, shader_id, command_buffer, s);
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "logsumexp_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_softmax_vulkan(inputs, out, state(), stream())) {
    throw std::runtime_error(
        "Softmax operation failed on Vulkan (unsupported dtype or layout).");
  }
}

void LogSumExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_logsumexp_vulkan(inputs, out, stream())) {
    throw std::runtime_error(
        "LogSumExp operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
