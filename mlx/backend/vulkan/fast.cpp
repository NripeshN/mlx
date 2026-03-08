// Copyright © 2024 Apple Inc.

#include <array>
#include <limits>

#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core {

namespace fast {

namespace {

bool sdpa_vulkan_supported(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    bool has_sinks,
    Stream s,
    std::string* reason = nullptr) {
  if (s.device == Device::cpu) {
    if (reason != nullptr) {
      *reason = "cpu stream";
    }
    return false;
  }
  if (is_training) {
    if (reason != nullptr) {
      *reason = "training unsupported";
    }
    return false;
  }
  if (output_logsumexp) {
    if (reason != nullptr) {
      *reason = "logsumexp output unsupported";
    }
    return false;
  }
  if (has_sinks) {
    if (reason != nullptr) {
      *reason = "sinks unsupported";
    }
    return false;
  }
  if (has_arr_mask) {
    if (reason != nullptr) {
      *reason = "array mask unsupported";
    }
    return false;
  }
  if (has_mask && !do_causal) {
    if (reason != nullptr) {
      *reason = "non-causal mask unsupported";
    }
    return false;
  }
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    if (reason != nullptr) {
      *reason = "rank != 4";
    }
    return false;
  }
  if (!is_vulkan_float_dtype(q.dtype()) || !is_vulkan_float_dtype(k.dtype()) ||
      !is_vulkan_float_dtype(v.dtype())) {
    if (reason != nullptr) {
      *reason = "unsupported dtype";
    }
    return false;
  }
  if (q.shape(0) != k.shape(0) || q.shape(0) != v.shape(0) ||
      q.shape(-1) != k.shape(-1) || k.shape(1) != v.shape(1) ||
      q.shape(1) % k.shape(1) != 0 || k.shape(2) != v.shape(2) ||
      q.shape(2) > k.shape(2) || v.shape(-1) <= 0) {
    if (reason != nullptr) {
      *reason = "incompatible attention shapes";
    }
    return false;
  }
  return true;
}

std::vector<array> eval_fallback_outputs(
    const std::function<std::vector<array>(std::vector<array>)>& fallback,
    const std::vector<array>& inputs) {
  auto outputs = fallback(inputs);
  eval(outputs);
  return outputs;
}

array apply_diag_mask_inf_vulkan(const array& scores, int n_past, Stream s) {
  eval(scores);

  array masked(scores.shape(), float32, nullptr, {});
  masked.set_data(allocator::malloc(masked.nbytes()));

  auto command_buffer = vulkan::begin_command_recording(s.index);
  vulkan::dispatch_diag_mask_inf_op(
      scores,
      masked,
      "diag_mask_inf_f32",
      command_buffer,
      s,
      checked_u32_size(scores.shape(scores.ndim() - 2), "rows_per_channel"),
      checked_u32_size(n_past, "n_past"));
  vulkan::end_command_recording(s.index);

  return masked;
}

bool try_eval_rms_norm_vulkan(
    const std::vector<array>& inputs,
    array& out,
    float eps,
    Stream s) {
  if (inputs.size() != 2) {
    return false;
  }

  array x = inputs[0];
  array w = inputs[1];
  if (x.ndim() == 0 || x.shape() != out.shape() || w.ndim() == 0) {
    return false;
  }

  if (!is_vulkan_float_dtype(x.dtype()) || !is_vulkan_float_dtype(w.dtype()) ||
      !is_vulkan_float_dtype(out.dtype())) {
    return false;
  }

  const bool use_f32_staging_io =
      x.dtype() != float32 || w.dtype() != float32 || out.dtype() != float32;
  if (use_f32_staging_io) {
    array x_f32(x.shape(), float32, nullptr, {});
    array w_f32(w.shape(), float32, nullptr, {});
    copy_gpu(x, x_f32, CopyType::General, s);
    copy_gpu(w, w_f32, CopyType::General, s);
    x = x_f32;
    w = w_f32;
  }

  if (x.ndim() > 4 || w.ndim() > 4) {
    return false;
  }

  if (!x.flags().contiguous || x.strides().back() != 1) {
    x = contiguous_copy_gpu(x, s);
  }
  if (!w.flags().row_contiguous || w.offset() != 0) {
    w = contiguous_copy_gpu(w, s);
  }

  if (!is_supported_unary_layout(x) || !is_supported_unary_layout(w)) {
    return false;
  }

  const bool staged_output = use_f32_staging_io || !out.flags().contiguous ||
      out.offset() != 0 || out.strides().back() != 1;
  array out_work =
      staged_output ? array(out.shape(), float32, nullptr, {}) : out;
  set_unary_output_data(x, out_work);
  if (!out_work.flags().contiguous || out_work.offset() != 0 ||
      out_work.strides().back() != 1 || !is_supported_unary_layout(out_work)) {
    return false;
  }

  if (x.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  const uint32_t axis_size =
      checked_u32_size(x.shape(x.ndim() - 1), "axis_size");
  if (axis_size == 0 || axis_size > 32u * 512u) {
    return false;
  }

  const uint32_t nrows =
      x.ndim() >= 2 ? checked_u32_size(x.shape(x.ndim() - 2), "nrows") : 1u;
  const uint32_t nchannels =
      x.ndim() >= 3 ? checked_u32_size(x.shape(x.ndim() - 3), "nchannels") : 1u;
  const uint32_t nsamples =
      x.ndim() >= 4 ? checked_u32_size(x.shape(x.ndim() - 4), "nsamples") : 1u;

  if (nrows > std::numeric_limits<uint32_t>::max() ||
      nchannels > std::numeric_limits<uint32_t>::max() ||
      nsamples > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_binary_op(
        x,
        w,
        out_work,
        "rms_norm_f32",
        command_buffer,
        s,
        vulkan::BinaryDispatchVariant::Standard,
        std::array<uint32_t, 3>{nrows, nchannels, nsamples},
        {0u, 1u});
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "rms_norm_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  std::string reason;
  const bool supported = sdpa_vulkan_supported(
      q,
      k,
      v,
      has_mask,
      has_arr_mask,
      do_causal,
      is_training,
      output_logsumexp,
      false,
      s,
      &reason);
  if (!supported && trace_fallback_enabled()) {
    std::ostringstream oss;
    oss << "q_shape=" << q.shape() << " k_shape=" << k.shape()
        << " v_shape=" << v.shape() << " has_mask=" << has_mask
        << " has_arr_mask=" << has_arr_mask << " do_causal=" << do_causal
        << " is_training=" << is_training
        << " output_logsumexp=" << output_logsumexp;
    trace_use_fallback("ScaledDotProductAttention", s, reason, oss.str());
  }
  return !supported;
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  if (trace_fallback_enabled()) {
    std::ostringstream oss;
    oss << "q_shape=" << q.shape();
    trace_use_fallback(
        "ScaledDotProductAttentionVJP",
        s,
        "no Vulkan implementation",
        oss.str());
  }
  return true;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs.size() != 3 || outputs.size() != 1) {
    throw std::runtime_error(
        "ScaledDotProductAttention expects 3 inputs and 1 output.");
  }

  auto fallback_outputs = [&]() {
    auto outs = eval_fallback_outputs(fallback_, inputs);
    copy_gpu(outs[0], outputs[0], CopyType::General, stream());
  };

  const array& q_in = inputs[0];
  const array& k_in = inputs[1];
  const array& v_in = inputs[2];

  std::string reason;
  if (!sdpa_vulkan_supported(
          q_in,
          k_in,
          v_in,
          do_causal_,
          false,
          do_causal_,
          false,
          false,
          has_sinks_,
          stream(),
          &reason)) {
    fallback_outputs();
    return;
  }

  try {
    auto s = stream();

    array q = multiply(array(scale_, q_in.dtype()), q_in, s);
    array k = k_in;
    array v = v_in;

    const int n_q_heads = q.shape(1);
    const int n_kv_heads = k.shape(1);
    const int n_repeats = n_q_heads / n_kv_heads;

    if (n_repeats > 1) {
      q = reshape_in_eval(
          q,
          Shape{q.shape(0), n_kv_heads, n_repeats, q.shape(2), q.shape(3)},
          s);
      k = reshape_in_eval(
          k, Shape{k.shape(0), k.shape(1), 1, k.shape(2), k.shape(3)}, s);
      v = reshape_in_eval(
          v, Shape{v.shape(0), v.shape(1), 1, v.shape(2), v.shape(3)}, s);
    }

    auto scores = matmul(q, swapaxes_in_eval(k, -1, -2), s);
    if (do_causal_) {
      if (scores.dtype() != float32) {
        scores = astype(scores, float32, s);
      }
      const int n_past = k_in.shape(2) - q_in.shape(2);
      scores = apply_diag_mask_inf_vulkan(scores, n_past, s);
    }

    scores = softmax(scores, std::vector<int>{-1}, true, s);

    array v_work = v;
    if (v_work.dtype() != scores.dtype()) {
      v_work = astype(v_work, scores.dtype(), s);
    }

    auto out = matmul(scores, v_work, s);
    if (n_repeats > 1) {
      out = reshape_in_eval(out, outputs[0].shape(), s);
    }
    if (out.dtype() != outputs[0].dtype()) {
      out = astype(out, outputs[0].dtype(), s);
    }

    eval(out);
    copy_gpu(out, outputs[0], CopyType::General, s);
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "sdpa_vulkan_eval_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    fallback_outputs();
  }
}

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error(
      "ScaledDotProductAttentionVJP has no Vulkan implementation.");
}

bool LayerNorm::use_fallback(Stream s) {
  trace_use_fallback("LayerNorm", s, "no Vulkan implementation");
  return true;
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("LayerNorm has no Vulkan implementation.");
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("LayerNormVJP has no Vulkan implementation.");
}

bool RMSNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (outputs.size() != 1) {
    throw std::runtime_error("RMSNorm expects a single output.");
  }
  if (try_eval_rms_norm_vulkan(inputs, outputs[0], eps_, stream())) {
    return;
  }
  eval_cpu_fallback_multi_on_stream<RMSNorm>(
      inputs, outputs, stream(), fallback_, eps_);
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("RMSNormVJP has no Vulkan implementation.");
}

void ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  ::mlx::core::gpu::synchronize(stream());
  auto cpu_stream = default_stream(Device::cpu);
  fast::ConvertFP8 cpu_convert(cpu_stream, state());
  cpu_convert.eval_cpu(inputs, outputs);
  synchronize(cpu_stream);
}

void Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  ::mlx::core::gpu::synchronize(stream());
  auto fallback_outputs = fallback_(inputs);
  if (fallback_outputs.size() != outputs.size()) {
    throw std::runtime_error(
        "[vulkan::Quantize::eval_gpu] Fallback output count mismatch.");
  }
  eval(fallback_outputs);
  for (int i = 0; i < outputs.size(); ++i) {
    outputs[i].copy_shared_buffer(fallback_outputs[i]);
  }
}

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("CustomKernel has no Vulkan implementation.");
}

} // namespace fast

} // namespace mlx::core
