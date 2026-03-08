// Copyright © 2024 Apple Inc.

#include <cmath>
#include <cstring>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/primitives_utils.h"
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

bool make_rope_positions(
    const array& offset,
    uint32_t batch,
    uint32_t steps,
    array& positions) {
  auto offset_eval = offset;
  if (offset_eval.status() == array::Status::unscheduled) {
    return false;
  }
  offset_eval.wait();
  if (offset_eval.dtype() != int32 || offset_eval.ndim() > 1 ||
      offset_eval.offset() != 0 || !offset_eval.flags().row_contiguous) {
    return false;
  }
  if (offset_eval.size() != 1 && offset_eval.size() != batch) {
    return false;
  }

  positions = array(
      {static_cast<int>(batch), static_cast<int>(steps)}, int32, nullptr, {});
  positions.set_data(allocator::malloc(positions.nbytes()));

  auto* dst = positions.data<int32_t>();
  const auto* src = offset_eval.data<int32_t>();
  if ((dst == nullptr || src == nullptr) && positions.size() != 0) {
    return false;
  }

  if (offset_eval.size() == 1) {
    const int32_t base = src[0];
    for (uint32_t b = 0; b < batch; ++b) {
      for (uint32_t t = 0; t < steps; ++t) {
        dst[b * steps + t] = base + static_cast<int32_t>(t);
      }
    }
  } else {
    for (uint32_t b = 0; b < batch; ++b) {
      const int32_t base = src[b];
      for (uint32_t t = 0; t < steps; ++t) {
        dst[b * steps + t] = base + static_cast<int32_t>(t);
      }
    }
  }

  return true;
}

bool stage_rope_freqs(const array& input, int dims, array& freqs) {
  auto input_eval = input;
  if (input_eval.status() == array::Status::unscheduled) {
    return false;
  }
  input_eval.wait();
  if (input_eval.dtype() != float32 || input_eval.ndim() != 1 ||
      input_eval.shape(0) != dims / 2 || input_eval.offset() != 0 ||
      !input_eval.flags().row_contiguous) {
    return false;
  }

  freqs = array({static_cast<int>(input_eval.shape(0))}, float32, nullptr, {});
  freqs.set_data(allocator::malloc(freqs.nbytes()));

  auto* dst = freqs.data<float>();
  const auto* src = input_eval.data<float>();
  if ((dst == nullptr || src == nullptr) && freqs.size() != 0) {
    return false;
  }
  std::memcpy(dst, src, freqs.nbytes());
  return true;
}

vulkan::RopePushConstants make_rope_push_constants(
    const array& in,
    const array& out,
    int dims,
    float base,
    float scale,
    bool forward,
    bool has_freqs) {
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
  if (x.status() == array::Status::unscheduled) {
    return false;
  }
  x.wait();

  if (x.size() != 0) {
    array x_private(x.shape(), x.dtype(), nullptr, {});
    x_private.set_data(allocator::malloc(x_private.nbytes()));
    if (x.flags().row_contiguous && x.offset() == 0) {
      auto* dst = x_private.data<char>();
      const auto* src = x.data<char>();
      if ((dst == nullptr || src == nullptr) && x_private.nbytes() != 0) {
        return false;
      }
      std::memcpy(dst, src, x_private.nbytes());
    } else {
      copy_gpu(x, x_private, CopyType::General, s);
    }
    x = x_private;
  }

  const auto shader_name = rope_shader_name(x.dtype(), traditional);
  if (shader_name.empty() || x.dtype() != out.dtype() || dims <= 0 ||
      (dims % 2) != 0 || dims > x.shape(-1) || x.offset() != 0) {
    return false;
  }

  array offset = inputs[1];
  if (offset.dtype() != int32 || offset.ndim() > 1) {
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

  array positions = offset;
  if (!make_rope_positions(offset, batch, steps, positions)) {
    return false;
  }

  array freqs = positions;
  const bool has_freqs = inputs.size() == 3;
  if (has_freqs && !stage_rope_freqs(inputs[2], dims, freqs)) {
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
      x_kernel, out_kernel, dims, base, scale, forward, has_freqs);
  const std::array<uint32_t, 3> grid = {
      std::min(pc.nrows, 32768u),
      std::max(1u, (pc.ne00 + 511u) / 512u),
      std::max(1u, (pc.nrows + 32767u) / 32768u)};

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_rope_op(
        x_kernel,
        positions,
        freqs,
        out_kernel,
        positions,
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
  assert(outputs.size() == 1);
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
