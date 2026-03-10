// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

namespace {

bool try_eval_softmax_vulkan(
    const std::vector<array>& inputs,
    array& out,
    bool /*precise*/,
    Stream s) {
  if (inputs.size() != 1) {
    return false;
  }

  array in = inputs[0];
  const bool f32_io = in.dtype() == float32 && out.dtype() == float32;
  const bool f16_io = in.dtype() == float16 && out.dtype() == float16;
  const bool bf16_io = in.dtype() == bfloat16 && out.dtype() == bfloat16;
  if (in.ndim() == 0 || (!f32_io && !f16_io && !bf16_io)) {
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

  if (!in.flags().contiguous || in.offset() != 0 || in.strides().back() != 1 ||
      !is_supported_unary_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }

  const bool staged_output = !softmax_out_target.flags().contiguous ||
      softmax_out_target.offset() != 0 ||
      softmax_out_target.strides().back() != 1 ||
      !is_supported_unary_layout(softmax_out_target);
  array out_work = staged_output
      ? array(
            softmax_out_target.shape(), softmax_out_target.dtype(), nullptr, {})
      : softmax_out_target;

  set_unary_output_data(in, out_work);
  if (in.shape() != out_work.shape()) {
    return false;
  }

  if (in.size() > std::numeric_limits<uint32_t>::max() ||
      out_work.size() > std::numeric_limits<uint32_t>::max() ||
      in.shape(in.ndim() - 1) > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const uint32_t row_width = static_cast<uint32_t>(in.shape(in.ndim() - 1));
  const bool use_large_softmax = row_width > 16384u;

  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, softmax_out_target, CopyType::GeneralGeneral, s);
    }
    if (use_f32_staging_io) {
      copy_gpu(softmax_out_target, out, CopyType::General, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    if (use_large_softmax) {
      vulkan::dispatch_softmax_large_op(
          in,
          out_work,
          use_f16_variant ? "soft_max_large1_f32_f16" : "soft_max_large1_f32",
          use_f16_variant ? "soft_max_large2_f32_f16" : "soft_max_large2_f32",
          use_f16_variant ? "soft_max_large3_f32_f16" : "soft_max_large3_f32",
          command_buffer,
          s);
    } else {
      vulkan::dispatch_softmax_op(
          in,
          out_work,
          use_f16_variant ? "soft_max_f32_f16" : "soft_max_f32",
          command_buffer,
          s);
    }
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(out_work, softmax_out_target, CopyType::GeneralGeneral, s);
    }
    if (use_f32_staging_io) {
      copy_gpu(softmax_out_target, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error&) {
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
    const std::string shader_name = out.dtype() == bfloat16
        ? "logsumexp_bf16"
        : "logsumexp_" + dtype_suffix(out.dtype());
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_sum_rows_op(in, out_work, shader_name, command_buffer, s);
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
