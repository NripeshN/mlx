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
#include "mlx/transforms.h"

namespace mlx::core {
namespace {

bool is_supported_quantized_bits(int bits) {
  return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
      bits == 8;
}

bool is_supported_quantized_output_dtype(Dtype dtype) {
  return dtype == float16 || dtype == bfloat16 || dtype == float32;
}

std::optional<vulkan::StaticShaderId> fused_affine_matmul_shader_id(
    Dtype x_dtype) {
  switch (x_dtype) {
    case float32:
      return vulkan::StaticShaderId::fused_affine_matmul_f32_f32;
    case float16:
      return vulkan::StaticShaderId::fused_affine_matmul_f16_f32;
    case bfloat16:
      return vulkan::StaticShaderId::fused_affine_matmul_bf16_f32;
    default:
      return std::nullopt;
  }
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

array nvfp4_quantize_dequantize(
    const array& in,
    Stream s,
    const std::optional<array>& global_scale) {
  array xhat = reshape(in, {-1, 16}, s);
  array scales =
      divide(max(abs(xhat, s), -1, true, s), array(6.0f, in.dtype()), s);
  array scale_encode = global_scale.has_value()
      ? divide(array(448.0f * 6.0f, float32), *global_scale, s)
      : array(1.0f, float32);
  array scales_enc = to_fp8(multiply(scales, scale_encode, s), s);
  array scale = divide(from_fp8(scales_enc, in.dtype(), s), scale_encode, s);

  array lut = astype(
      array(
          {+0.0f,
           +0.5f,
           +1.0f,
           +1.5f,
           +2.0f,
           +3.0f,
           +4.0f,
           +6.0f,
           -0.0f,
           -0.5f,
           -1.0f,
           -1.5f,
           -2.0f,
           -3.0f,
           -4.0f,
           -6.0f}),
      in.dtype(),
      s);
  array idx = argmin(
      abs(subtract(expand_dims(divide(xhat, scale, s), -1, s), lut, s), s),
      -1,
      false,
      s);
  array dequant = multiply(gather(lut, idx, 0, {1}, s), scale, s);
  return reshape(dequant, in.shape(), s);
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

bool nvfp4_dequantize_to_float32(
    const array& w,
    const array& scales,
    array& out,
    Stream s) {
  if (w.dtype() != uint32 || scales.dtype() != uint8 ||
      out.dtype() != float32) {
    return false;
  }

  array w_work = ensure_row_contiguous_zero_offset(w, s);
  array scales_work = ensure_row_contiguous_zero_offset(scales, s);
  if (!is_row_contiguous_zero_offset(w_work) ||
      !is_row_contiguous_zero_offset(scales_work)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  Nvfp4DequantPushConstants push_constants{};
  push_constants.ne = static_cast<uint32_t>(out.size());

  auto command_buffer = vulkan::begin_command_recording(s.index);
  dispatch_nvfp4_dequant_op(
      w_work,
      scales_work,
      out,
      StaticShaderId::dequant_nvfp4_f32,
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
  array x = ensure_row_contiguous_zero_offset(inputs[0], s);
  array w = ensure_row_contiguous_zero_offset(inputs[1], s);
  array scales = ensure_float32_row_contiguous(inputs[2], s);
  array biases = ensure_float32_row_contiguous(inputs[3], s);

  const bool vector_lhs = x.ndim() == 1;
  const bool flatten_lhs_batches = x.ndim() > 2 && w.ndim() == 2;
  array x_mat = x;
  if (vector_lhs) {
    Shape mat_shape = {1, x.shape(0)};
    x_mat = array(mat_shape, x.dtype(), nullptr, {});
    x_mat.copy_shared_buffer(
        x, make_contiguous_strides(mat_shape), x.flags(), x.size());
  } else if (flatten_lhs_batches) {
    Shape flat_shape = {static_cast<int>(x.size() / x.shape(-1)), x.shape(-1)};
    x_mat = array(flat_shape, x.dtype(), nullptr, {});
    x_mat.copy_shared_buffer(
        x, make_contiguous_strides(flat_shape), x.flags(), x.size());
  }

  auto fused_shader = fused_affine_matmul_shader_id(x_mat.dtype());
  const bool enable_fused_decode_qmm = []() {
    if (const char* env = std::getenv("MLX_VULKAN_FUSED_AFFINE_QMM");
        env != nullptr) {
      return std::string_view(env) != "0";
    }
    return true;
  }();
  const Dtype out_work_dtype = float32;

  array out_work(
      (vector_lhs || flatten_lhs_batches)
          ? Shape{static_cast<int>(x_mat.shape(0)), out.shape(-1)}
          : out.shape(),
      out_work_dtype,
      nullptr,
      {});

  bool fused_dispatched = false;
  if (enable_fused_decode_qmm && transpose_ && fused_shader.has_value() &&
      x_mat.ndim() == 2 && w.ndim() == 2 && scales.ndim() == 2 &&
      biases.ndim() == 2 && is_row_contiguous_zero_offset(x_mat) &&
      is_row_contiguous_zero_offset(w) &&
      is_row_contiguous_zero_offset(scales) &&
      is_row_contiguous_zero_offset(biases)) {
    const uint32_t rows = static_cast<uint32_t>(out_work.shape(-2));
    const uint32_t cols = static_cast<uint32_t>(out_work.shape(-1));
    const uint32_t k = static_cast<uint32_t>(x_mat.shape(-1));
    const uint32_t num_groups = static_cast<uint32_t>(scales.shape(-1));
    const bool decode_like_rows = rows == 1;

    if (decode_like_rows && rows == static_cast<uint32_t>(x_mat.shape(-2)) &&
        cols == static_cast<uint32_t>(w.shape(-2)) &&
        num_groups ==
            static_cast<uint32_t>((k + group_size_ - 1) / group_size_)) {
      try {
        out_work.set_data(allocator::malloc(out_work.nbytes()));
        if (out_work.size() != 0) {
          vulkan::FusedAffineMatmulPushConstants push_constants{};
          push_constants.rows = rows;
          push_constants.cols = cols;
          push_constants.K = k;
          push_constants.packed_row_bytes =
              static_cast<uint32_t>(w.strides(-2) * sizeof(uint32_t));
          push_constants.x_row_stride =
              static_cast<uint32_t>(x_mat.strides(-2));
          push_constants.out_row_stride =
              static_cast<uint32_t>(out_work.strides(-2));
          push_constants.scale_row_stride =
              static_cast<uint32_t>(scales.strides(-2));
          push_constants.bias_row_stride =
              static_cast<uint32_t>(biases.strides(-2));
          push_constants.bits = static_cast<uint32_t>(bits_);
          push_constants.group_size = static_cast<uint32_t>(group_size_);
          push_constants.num_groups = num_groups;

          const std::array<uint32_t, 3> grid = {
              (cols + 15u) / 16u, (rows + 15u) / 16u, 1u};

          auto command_buffer = vulkan::begin_command_recording(s.index);
          vulkan::dispatch_fused_affine_matmul_op(
              w,
              scales,
              biases,
              x_mat,
              out_work,
              *fused_shader,
              command_buffer,
              s,
              push_constants,
              grid);
          vulkan::end_command_recording(s.index);
        }
        fused_dispatched = true;
      } catch (const std::runtime_error&) {
        fused_dispatched = false;
      }
    }
  }

  if (!fused_dispatched) {
    array x_mat_f32 = ensure_float32_row_contiguous(x_mat, s);
    array w_deq(expanded_quantized_shape(w, bits_), float32, nullptr, {});
    if (!vulkan::affine_dequantize_to_float32(
            w, scales, biases, w_deq, s, group_size_, bits_)) {
      throw std::runtime_error(
          "[QuantizedMatmul::eval_gpu] Failed to dequantize weights on Vulkan.");
    }

    array rhs = transpose_ ? swapaxes_in_eval(w_deq, -1, -2) : w_deq;
    rhs = ensure_row_contiguous_zero_offset(rhs, s);

    if (!try_eval_matmul_vulkan({x_mat_f32, rhs}, out_work, s)) {
      throw std::runtime_error(
          "[QuantizedMatmul::eval_gpu] Failed to dispatch Vulkan matmul.");
    }
  }

  if (vector_lhs || flatten_lhs_batches) {
    array::Flags flags = out.flags();
    flags.contiguous = true;
    flags.row_contiguous = true;
    if (vector_lhs) {
      flags.col_contiguous = true;
    } else {
      auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
      flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
    }

    if (out_work.dtype() == out.dtype()) {
      out.copy_shared_buffer(
          out_work, make_contiguous_strides(out.shape()), flags, out.size());
      out.detach();
      out.set_status(array::Status::evaluated);
      return;
    }

    array out_view(out.shape(), out_work.dtype(), nullptr, {});
    out_view.copy_shared_buffer(
        out_work, make_contiguous_strides(out.shape()), flags, out.size());
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu(out_view, out, CopyType::GeneralGeneral, s);
    out.detach();
    out.set_status(array::Status::evaluated);
    return;
  }

  if (out_work.dtype() == out.dtype()) {
    out.copy_shared_buffer(out_work);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu(out_work, out, CopyType::General, s);
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  bool w_quantized = inputs[1].dtype() == uint32;
  if (!w_quantized) {
    throw std::runtime_error("[QQMatmul::eval_gpu] Not implemented on Vulkan.");
  }

  auto& s = stream();
  auto mode = quantization_mode_to_string(mode_);

  std::optional<array> global_scale_x = std::nullopt;
  std::optional<array> global_scale_w = std::nullopt;
  if (mode_ == QuantizationMode::Nvfp4 && inputs.size() >= 5) {
    global_scale_x = inputs[3];
    global_scale_w = inputs[4];
  }

  if (mode_ != QuantizationMode::Nvfp4) {
    throw std::runtime_error(
        "[QQMatmul::eval_gpu] Only nvfp4 mode is implemented on Vulkan.");
  }

  array xhat = nvfp4_quantize_dequantize(inputs[0], s, global_scale_x);
  array w_hat = dequantize(
      inputs[1],
      inputs[2],
      std::nullopt,
      group_size_,
      bits_,
      mode,
      global_scale_w,
      inputs[0].dtype(),
      s);

  array result = matmul(xhat, swapaxes(w_hat, -1, -2, s), s);
  if (out.dtype() != result.dtype()) {
    result = astype(result, out.dtype(), s);
  }
  eval(result);
  out.copy_shared_buffer(result);
}

} // namespace mlx::core
