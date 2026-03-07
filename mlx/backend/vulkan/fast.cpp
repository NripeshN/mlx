// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core {

namespace fast {

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
  if (trace_fallback_enabled()) {
    std::ostringstream oss;
    oss << "q_shape=" << q.shape() << " k_shape=" << k.shape()
        << " v_shape=" << v.shape() << " has_mask=" << has_mask
        << " has_arr_mask=" << has_arr_mask << " do_causal=" << do_causal
        << " is_training=" << is_training
        << " output_logsumexp=" << output_logsumexp;
    trace_use_fallback(
        "ScaledDotProductAttention", s, "no Vulkan implementation", oss.str());
  }
  return true;
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
  throw std::runtime_error(
      "ScaledDotProductAttention has no Vulkan implementation.");
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
  trace_use_fallback("RMSNorm", s, "no Vulkan implementation");
  return true;
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("RMSNorm has no Vulkan implementation.");
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
