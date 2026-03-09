// Copyright © 2024 Apple Inc.

#include <cmath>
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/fast_primitives.h"

namespace mlx::core {

namespace fast {

namespace {

std::string rope_shader_name(Dtype dtype, bool traditional) {
  if (dtype == float32) {
    return traditional ? "rope_norm_f32" : "rope_neox_f32";
  }
  if (dtype == float16) {
    return traditional ? "rope_norm_f16_rte" : "rope_neox_f16_rte";
  }
  return {};
}

bool normalize_rope_input(
    const array& in,
    array& normalized,
    Shape& normalized_shape,
    Stream s) {
  if (in.ndim() < 3) {
    return false;
  }

  if (in.ndim() == 3) {
    if (!in.flags().row_contiguous || in.offset() != 0) {
      return false;
    }
    normalized_shape = {
        static_cast<int>(in.shape(0)),
        1,
        static_cast<int>(in.shape(1)),
        static_cast<int>(in.shape(2))};
    normalized = reshape_in_eval(in, normalized_shape, s);
    return true;
  }

  if (in.ndim() == 4) {
    normalized = in;
    normalized_shape = in.shape();
    return true;
  }

  if (!in.flags().row_contiguous || in.offset() != 0) {
    return false;
  }
  normalized_shape = Flatten::output_shape(in, 1, in.ndim() - 3);
  normalized = reshape_in_eval(in, normalized_shape, s);
  return true;
}

array normalize_rope_output(
    array& out,
    const Shape& normalized_shape,
    Stream s) {
  if (out.shape() == normalized_shape) {
    return out;
  }
  return reshape_in_eval(out, normalized_shape, s);
}

bool prepare_rope_offsets(
    const array& input,
    uint32_t batch,
    array& offsets,
    uint32_t& position_stride,
    Stream s) {
  if (input.dtype() != int32 || input.ndim() > 1) {
    return false;
  }
  if (input.size() != 1 && input.size() != batch) {
    return false;
  }

  offsets = input;
  if (offsets.offset() != 0 || !offsets.flags().row_contiguous) {
    offsets = contiguous_copy_gpu(offsets, s);
  }
  if (offsets.offset() != 0 || !offsets.flags().row_contiguous) {
    return false;
  }

  position_stride = offsets.size() == 1 ? 0u : 1u;
  return true;
}

bool prepare_rope_freqs(const array& input, int dims, array& freqs, Stream s) {
  if (input.dtype() != float32 || input.ndim() != 1 ||
      input.shape(0) != dims / 2) {
    return false;
  }

  freqs = input;
  if (freqs.offset() != 0 || !freqs.flags().row_contiguous) {
    freqs = contiguous_copy_gpu(freqs, s);
  }
  return freqs.offset() == 0 && freqs.flags().row_contiguous;
}

vulkan::RopePushConstants make_rope_push_constants(
    const array& in,
    const array& out,
    int dims,
    float base,
    float scale,
    bool forward,
    bool has_freqs,
    uint32_t position_stride,
    bool positions_are_offsets) {
  vulkan::RopePushConstants pc{};
  const uint32_t batch = checked_u32_size(in.shape(0), "rope batch");
  const uint32_t steps = checked_u32_size(in.shape(1), "rope steps");
  const uint32_t heads = checked_u32_size(in.shape(2), "rope heads");

  pc.rope_mode = 0;
  pc.nrows = checked_mul_u32(
      checked_mul_u32(batch, steps, "rope rows"), heads, "rope rows");
  pc.n_dims = checked_u32_size(dims, "rope dims");
  pc.freq_scale = scale;
  pc.freq_base = base;
  pc.ext_factor = 0.0f;
  pc.attn_factor = 1.0f;
  pc.corr_dims[0] = 0.0f;
  pc.corr_dims[1] = 0.0f;
  pc.theta_scale = std::pow(base, -2.0f / static_cast<float>(dims));
  pc.has_ff = has_freqs ? 1u : 0u;
  pc.sections[0] = 0;
  pc.sections[1] = 0;
  pc.sections[2] = 0;
  pc.sections[3] = 0;
  pc.is_imrope = 0;
  pc.is_back = forward ? 0u : 1u;
  pc.set_rows_stride = 0;
  pc.position_stride = position_stride;
  pc.positions_are_offsets = positions_are_offsets ? 1u : 0u;

  pc.ne00 = checked_u32_size(in.shape(3), "rope ne00");
  pc.ne01 = checked_u32_size(in.shape(2), "rope ne01");
  pc.ne02 = checked_u32_size(in.shape(1), "rope ne02");
  pc.nb01 = checked_u32_size(in.strides(2), "rope nb01");
  pc.nb02 = checked_u32_size(in.strides(1), "rope nb02");
  pc.nb03 = checked_u32_size(in.strides(0), "rope nb03");
  pc.nb11 = checked_u32_size(out.strides(2), "rope nb11");
  pc.nb12 = checked_u32_size(out.strides(1), "rope nb12");
  pc.nb13 = checked_u32_size(out.strides(0), "rope nb13");
  return pc;
}

bool try_eval_rope_vulkan(
    const std::vector<array>& inputs,
    array& out,
    int dims,
    bool traditional,
    float base,
    float scale,
    bool forward,
    Stream s) {
  if (inputs.size() != 2 && inputs.size() != 3) {
    return false;
  }

  array x = inputs[0];
  if (x.offset() != 0 || !x.flags().row_contiguous) {
    x = contiguous_copy_gpu(x, s);
  }

  const auto shader_name = rope_shader_name(x.dtype(), traditional);
  if (shader_name.empty() || x.dtype() != out.dtype() || dims <= 0 ||
      (dims % 2) != 0 || dims > x.shape(-1) || x.offset() != 0) {
    return false;
  }

  array x_norm = x;
  Shape normalized_shape = x.shape();
  if (!normalize_rope_input(x, x_norm, normalized_shape, s) ||
      x_norm.ndim() != 4 || x_norm.offset() != 0) {
    return false;
  }

  const uint32_t batch = checked_u32_size(x_norm.shape(0), "rope batch");
  const uint32_t steps = checked_u32_size(x_norm.shape(2), "rope steps");

  array offsets = inputs[1];
  uint32_t position_stride = 0;
  if (!prepare_rope_offsets(offsets, batch, offsets, position_stride, s)) {
    return false;
  }

  array freqs = offsets;
  const bool has_freqs = inputs.size() == 3;
  if (has_freqs && !prepare_rope_freqs(inputs[2], dims, freqs, s)) {
    return false;
  }

  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return true;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  array out_norm = normalize_rope_output(out, normalized_shape, s);

  array x_kernel = swapaxes_in_eval(x_norm, 1, 2);
  array out_kernel = swapaxes_in_eval(out_norm, 1, 2);
  if (x_kernel.offset() != 0 || out_kernel.offset() != 0) {
    return false;
  }
  auto pc = make_rope_push_constants(
      x_kernel,
      out_kernel,
      dims,
      base,
      scale,
      forward,
      has_freqs,
      position_stride,
      true);
  const std::array<uint32_t, 3> grid = {
      std::min(pc.nrows, 32768u),
      std::max(1u, (pc.ne00 + 511u) / 512u),
      std::max(1u, (pc.nrows + 32767u) / 32768u)};

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_rope_op(
        x_kernel,
        offsets,
        freqs,
        out_kernel,
        offsets,
        shader_name,
        command_buffer,
        s,
        pc,
        grid);
    vulkan::end_command_recording(s.index);
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "rope_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

bool RoPE::use_fallback(Stream s) {
  if (s.device == Device::cpu) {
    trace_use_fallback("RoPE", s, "CPU stream");
    return true;
  }
  return false;
}

void RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (outputs.size() != 1) {
    throw std::runtime_error(
        "[vulkan::RoPE::eval_gpu] Expected exactly one output.");
  }

  if (try_eval_rope_vulkan(
          inputs,
          outputs[0],
          dims_,
          traditional_,
          base_,
          scale_,
          forward_,
          stream())) {
    return;
  }

  ::mlx::core::gpu::synchronize(stream());
  auto fallback_outputs = fallback_(inputs);
  if (fallback_outputs.size() != outputs.size()) {
    throw std::runtime_error(
        "[vulkan::RoPE::eval_gpu] Fallback output count mismatch.");
  }
  eval(fallback_outputs);
  for (int i = 0; i < outputs.size(); ++i) {
    outputs[i].copy_shared_buffer(fallback_outputs[i]);
  }
}

} // namespace fast

} // namespace mlx::core
