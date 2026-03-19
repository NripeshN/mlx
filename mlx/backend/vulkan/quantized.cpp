// Copyright © 2026 Apple Inc.

#include "mlx/backend/vulkan/quantized.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/matmul.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {
namespace {

bool is_supported_quantized_bits(int bits) {
  return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
      bits == 8;
}

bool is_supported_quantized_output_dtype(Dtype dtype) {
  return dtype == float16 || dtype == bfloat16 || dtype == float32;
}

bool is_row_contiguous_zero_offset(const array& arr) {
  return arr.flags().row_contiguous && arr.offset() == 0 &&
      arr.strides(-1) == 1;
}

array ensure_row_contiguous_zero_offset(const array& arr, Stream s) {
  if (is_row_contiguous_zero_offset(arr)) {
    return arr;
  }
  return contiguous_copy_gpu(arr, s);
}

array ensure_float32_row_contiguous(const array& arr, Stream s) {
  if (arr.dtype() == float32 && is_row_contiguous_zero_offset(arr)) {
    return arr;
  }
  array out(arr.shape(), float32, nullptr, {});
  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu(arr, out, CopyType::General, s);
  if (!is_row_contiguous_zero_offset(out)) {
    out = contiguous_copy_gpu(out, s);
  }
  return out;
}

Shape expanded_quantized_shape(const array& w, int bits) {
  auto out_shape = w.shape();
  out_shape.back() = w.shape(-1) * 32 / bits;
  return out_shape;
}

} // namespace

namespace vulkan {

bool affine_quantize_from_float32(
    const array& in,
    array& w,
    array& scales,
    array& biases,
    Stream s,
    int group_size,
    int bits) {
  if (in.dtype() != float32 || w.dtype() != uint32 ||
      scales.dtype() != float32 || biases.dtype() != float32) {
    return false;
  }
  if (!is_supported_quantized_bits(bits)) {
    return false;
  }

  array in_work = ensure_row_contiguous_zero_offset(in, s);
  if (!is_row_contiguous_zero_offset(in_work)) {
    return false;
  }

  w.set_data(allocator::malloc(w.nbytes()));
  scales.set_data(allocator::malloc(scales.nbytes()));
  biases.set_data(allocator::malloc(biases.nbytes()));
  if (in.size() == 0) {
    return true;
  }

  AffineQuantPushConstants push_constants{};
  push_constants.ne = static_cast<uint32_t>(in.size());
  push_constants.bits = static_cast<uint32_t>(bits);
  push_constants.group_size = static_cast<uint32_t>(group_size);
  const uint32_t num_groups = static_cast<uint32_t>(scales.size());

  auto command_buffer = vulkan::begin_command_recording(s.index);
  dispatch_affine_quant_op(
      in_work,
      w,
      scales,
      biases,
      StaticShaderId::affine_quantize_f32,
      command_buffer,
      s,
      push_constants,
      {(num_groups + 255u) / 256u, 1, 1});
  vulkan::end_command_recording(s.index);
  return true;
}

bool affine_dequantize_to_float32(
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    Stream s,
    int group_size,
    int bits) {
  if (w.dtype() != uint32 || scales.dtype() != float32 ||
      biases.dtype() != float32 || out.dtype() != float32) {
    return false;
  }
  if (!is_supported_quantized_bits(bits)) {
    return false;
  }

  array w_work = ensure_row_contiguous_zero_offset(w, s);
  array scales_work = ensure_row_contiguous_zero_offset(scales, s);
  array biases_work = ensure_row_contiguous_zero_offset(biases, s);

  if (!is_row_contiguous_zero_offset(w_work) ||
      !is_row_contiguous_zero_offset(scales_work) ||
      !is_row_contiguous_zero_offset(biases_work)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  AffineDequantPushConstants push_constants{};
  push_constants.ne = static_cast<uint32_t>(out.size());
  push_constants.bits = static_cast<uint32_t>(bits);
  push_constants.group_size = static_cast<uint32_t>(group_size);

  auto command_buffer = vulkan::begin_command_recording(s.index);
  dispatch_affine_dequant_op(
      w_work,
      scales_work,
      biases_work,
      out,
      StaticShaderId::affine_dequantize_f32,
      command_buffer,
      s,
      push_constants,
      {(push_constants.ne + 255u) / 256u, 1, 1});
  vulkan::end_command_recording(s.index);
  return true;
}

} // namespace vulkan

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (mode_ != QuantizationMode::Affine) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Only affine mode is implemented on Vulkan.");
  }
  if (inputs.size() != 4) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Expected x, w, scales, biases.");
  }
  if (!is_supported_quantized_bits(bits_)) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Unsupported quantization bits on Vulkan.");
  }
  if (!is_supported_quantized_output_dtype(inputs[0].dtype()) ||
      !is_supported_quantized_output_dtype(out.dtype())) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Only float16, bfloat16, and float32 are supported.");
  }

  auto& s = stream();
  array x = ensure_float32_row_contiguous(inputs[0], s);
  array w = ensure_row_contiguous_zero_offset(inputs[1], s);
  array scales = ensure_float32_row_contiguous(inputs[2], s);
  array biases = ensure_float32_row_contiguous(inputs[3], s);

  const bool flatten_lhs_batches = x.ndim() > 2 && w.ndim() == 2;
  if (inputs[1].ndim() > 2) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Batched quantized weights are not implemented on Vulkan.");
  }
  array x_mat = x;
  if (flatten_lhs_batches) {
    Shape flat_shape = {static_cast<int>(x.size() / x.shape(-1)), x.shape(-1)};
    x_mat = array(flat_shape, float32, nullptr, {});
    x_mat.copy_shared_buffer(
        x, make_contiguous_strides(flat_shape), x.flags(), x.size());
  }

  array w_deq(expanded_quantized_shape(w, bits_), float32, nullptr, {});
  if (!vulkan::affine_dequantize_to_float32(
          w, scales, biases, w_deq, s, group_size_, bits_)) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Failed to dequantize weights on Vulkan.");
  }

  array rhs = transpose_ ? swapaxes_in_eval(w_deq, -1, -2) : w_deq;
  rhs = ensure_row_contiguous_zero_offset(rhs, s);

  array out_work(
      flatten_lhs_batches
          ? Shape{static_cast<int>(x_mat.shape(0)), out.shape(-1)}
          : out.shape(),
      float32,
      nullptr,
      {});
  if (!try_eval_matmul_vulkan({x_mat, rhs}, out_work, s)) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Failed to dispatch Vulkan matmul.");
  }

  if (flatten_lhs_batches) {
    array::Flags flags = out.flags();
    flags.contiguous = true;
    flags.row_contiguous = true;
    auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
    flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;

    if (out.dtype() == float32) {
      out.copy_shared_buffer(
          out_work, make_contiguous_strides(out.shape()), flags, out.size());
      out.detach();
      out.set_status(array::Status::evaluated);
      return;
    }

    array out_view(out.shape(), float32, nullptr, {});
    out_view.copy_shared_buffer(
        out_work, make_contiguous_strides(out.shape()), flags, out.size());
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu(out_view, out, CopyType::GeneralGeneral, s);
    out.detach();
    out.set_status(array::Status::evaluated);
    return;
  }

  if (out.dtype() == float32) {
    out.copy_shared_buffer(out_work);
    return;
  }

  array out_cast = astype(out_work, out.dtype(), s);
  out.copy_shared_buffer(out_cast);
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[QQMatmul::eval_gpu] Not implemented on Vulkan.");
}

} // namespace mlx::core
