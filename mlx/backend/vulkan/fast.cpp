// Copyright © 2024 Apple Inc.

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/transforms_impl.h"

namespace mlx::core {

namespace fast {

namespace {

array apply_diag_mask_inf_vulkan(const array& scores, int n_past, Stream s);

bool experimental_flash_attention_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_EXPERIMENTAL_FLASH_ATTN");
        env != nullptr) {
      return std::string(env) != "0";
    }
    return true;
  }();
  return enabled;
}

struct FlashAttentionTuningParams {
  uint32_t workgroup_size;
  uint32_t block_rows;
  uint32_t block_cols;
  uint32_t d_split;
  uint32_t row_split;
  uint32_t subgroup_size;
  uint32_t shmem_staging;
  uint32_t limit_occupancy_shmem;
};

uint32_t lowest_set_bit(uint32_t value) {
  return value & (~value + 1u);
}

std::pair<uint32_t, uint32_t> vulkan_device_vendor_and_subgroup_size() {
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(
      vulkan::VulkanContext::get().physical_device(), &props);

  VkPhysicalDeviceSubgroupProperties subgroup_props{};
  subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceProperties2 props2{};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &subgroup_props;
  vkGetPhysicalDeviceProperties2(
      vulkan::VulkanContext::get().physical_device(), &props2);

  return {props.vendorID, subgroup_props.subgroupSize};
}

FlashAttentionTuningParams get_flash_attention_tuning_params(
    uint32_t hsk,
    uint32_t hsv,
    uint32_t n_rows,
    uint32_t n_kv) {
  auto [vendor_id, device_subgroup_size] =
      vulkan_device_vendor_and_subgroup_size();
  (void)vendor_id;
  (void)hsv;
  (void)n_kv;
  const uint32_t subgroup_size = std::max(device_subgroup_size, 1u);

  FlashAttentionTuningParams result{};
  result.workgroup_size = 128u;
  result.row_split = 1u;
  result.block_rows = (n_rows <= 1) ? 1u : ((n_rows <= 4) ? 4u : 8u);
  result.block_cols = 32u;

  const uint32_t d_lsb = lowest_set_bit(hsk);
  result.d_split = std::min(8u, d_lsb / 4u);
  result.subgroup_size = subgroup_size;
  result.shmem_staging = 0u;
  result.limit_occupancy_shmem = 0u;

  return result;
}

array make_flash_attention_causal_mask(
    const array& q,
    const array& k,
    Stream s) {
  array mask = zeros({q.shape(0), 1, q.shape(2), k.shape(2)}, float32, s);
  if (mask.dtype() != float32) {
    throw std::runtime_error("Unexpected causal mask dtype.");
  }
  const int n_past = k.shape(2) - q.shape(2);
  mask = apply_diag_mask_inf_vulkan(mask, n_past, s);
  return astype(mask, float16, s);
}

bool try_dispatch_flash_attention_native_vulkan(
    const array& q,
    const array& k,
    const array& v,
    const array* mask,
    array& out_storage,
    Stream s) {
  const uint32_t batch = checked_u32_size(q.shape(0), "flash_attn batch");
  const uint32_t q_heads = checked_u32_size(q.shape(1), "flash_attn q_heads");
  const uint32_t q_len = checked_u32_size(q.shape(2), "flash_attn q_len");
  const uint32_t hsk = checked_u32_size(q.shape(3), "flash_attn hsk");
  const uint32_t kv_heads = checked_u32_size(k.shape(1), "flash_attn kv_heads");
  const uint32_t kv_len = checked_u32_size(k.shape(2), "flash_attn kv_len");
  const uint32_t hsv = checked_u32_size(v.shape(3), "flash_attn hsv");
  const bool has_mask = mask != nullptr;

  const auto tuning =
      get_flash_attention_tuning_params(hsk, hsv, q_len, kv_len);
  if (tuning.d_split == 0 || tuning.block_rows == 0 || tuning.block_cols == 0 ||
      tuning.row_split == 0 || hsk % tuning.d_split != 0 ||
      hsv % tuning.d_split != 0 ||
      tuning.block_cols %
              (tuning.workgroup_size / tuning.d_split / tuning.row_split) !=
          0) {
    return false;
  }

  const uint32_t q_stride =
      checked_u32_size(q.strides(2), "flash_attn q_stride");
  const uint32_t k_stride =
      checked_u32_size(k.strides(2), "flash_attn k_stride");
  const uint32_t v_stride =
      checked_u32_size(v.strides(2), "flash_attn v_stride");
  const bool aligned = (kv_len % tuning.block_cols) == 0 &&
      (q_stride & 7u) == 0 && (k_stride & 7u) == 0 && (v_stride & 7u) == 0;
  auto stride_bytes = [](const array& arr, int axis, const char* name) {
    return checked_u32_size(
        static_cast<int64_t>(arr.strides(axis)) * size_of(arr.dtype()), name);
  };

  vulkan::FlashAttentionPushConstants push_constants{};
  push_constants.N = q_len;
  push_constants.KV = kv_len;
  push_constants.ne1 = q_heads;
  push_constants.ne2 = q_len;
  push_constants.ne3 = batch;
  push_constants.neq2 = q_heads;
  push_constants.neq3 = batch;
  push_constants.nek2 = kv_heads;
  push_constants.nek3 = batch;
  push_constants.nev2 = checked_u32_size(v.shape(1), "flash_attn v_heads");
  push_constants.nev3 = checked_u32_size(v.shape(0), "flash_attn v_batch");
  push_constants.nem1 = has_mask
      ? checked_u32_size((*mask).shape(2), "flash_attn mask_rows")
      : 1u;
  push_constants.nem2 = has_mask
      ? checked_u32_size((*mask).shape(1), "flash_attn mask_heads")
      : 1u;
  push_constants.nem3 = has_mask
      ? checked_u32_size((*mask).shape(0), "flash_attn mask_batch")
      : 1u;
  push_constants.nb01 = q_stride;
  push_constants.nb02 = stride_bytes(q, 1, "flash_attn q_nb02");
  push_constants.nb03 = stride_bytes(q, 0, "flash_attn q_nb03");
  push_constants.nb11 = k_stride;
  push_constants.nb12 = stride_bytes(k, 1, "flash_attn k_nb12");
  push_constants.nb13 = stride_bytes(k, 0, "flash_attn k_nb13");
  push_constants.nb21 = v_stride;
  push_constants.nb22 = stride_bytes(v, 1, "flash_attn v_nb22");
  push_constants.nb23 = stride_bytes(v, 0, "flash_attn v_nb23");
  push_constants.scale = 1.0f;
  push_constants.max_bias = 0.0f;
  push_constants.logit_softcap = 0.0f;
  push_constants.mask_n_head_log2 = 0u;
  push_constants.m0 = 0.0f;
  push_constants.m1 = 0.0f;
  push_constants.gqa_ratio = 1u;
  push_constants.split_kv = kv_len;
  push_constants.k_num = 1u;

  const std::vector<uint32_t> specialization_constants = {
      tuning.workgroup_size,
      tuning.block_rows,
      tuning.block_cols,
      hsk,
      hsv,
      aligned ? 0u : 1u,
      tuning.d_split,
      tuning.row_split,
      tuning.subgroup_size,
      tuning.shmem_staging,
      has_mask ? 2u : 0u,
      tuning.limit_occupancy_shmem,
  };

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_flash_attention_op(
        q,
        k,
        v,
        has_mask ? *mask : q,
        q,
        out_storage,
        q,
        "flash_attn_f32_f16_f16_fp32",
        command_buffer,
        s,
        push_constants,
        {(q_len + tuning.block_rows - 1u) / tuning.block_rows, q_heads, batch},
        specialization_constants);
    vulkan::end_command_recording(s.index);
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "flash_attn_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

bool try_eval_flash_attention_vulkan(
    const std::vector<array>& inputs,
    array& out,
    float scale,
    bool do_causal,
    Stream s) {
  if (inputs.size() != 3) {
    return false;
  }
  if (!experimental_flash_attention_enabled()) {
    return false;
  }

  array q = inputs[0];
  array k = inputs[1];
  array v = inputs[2];

  if (k.dtype() != float16 || v.dtype() != float16 || out.ndim() != 4 ||
      q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    return false;
  }

  const uint32_t batch = checked_u32_size(q.shape(0), "flash_attn batch");
  const uint32_t q_heads = checked_u32_size(q.shape(1), "flash_attn q_heads");
  const uint32_t q_len = checked_u32_size(q.shape(2), "flash_attn q_len");
  const uint32_t hsk = checked_u32_size(q.shape(3), "flash_attn hsk");
  const uint32_t kv_heads = checked_u32_size(k.shape(1), "flash_attn kv_heads");
  const uint32_t kv_len = checked_u32_size(k.shape(2), "flash_attn kv_len");
  const uint32_t hsv = checked_u32_size(v.shape(3), "flash_attn hsv");

  if (batch == 0 || q_heads == 0 || q_len == 0 || hsk == 0 || kv_heads == 0 ||
      kv_len == 0 || hsv == 0 || hsk % 4 != 0 || hsv % 4 != 0 ||
      q_heads % kv_heads != 0) {
    return false;
  }

  q = multiply(array(scale, q.dtype()), q, s);
  if (q.dtype() != float32) {
    q = astype(q, float32, s);
  }

  auto make_contiguous_zero_offset = [&](array x) {
    if (!x.flags().row_contiguous || x.strides().back() != 1 ||
        x.offset() != 0) {
      x = contiguous_copy_gpu(x, s);
    }
    return x;
  };

  q = make_contiguous_zero_offset(q);
  k = make_contiguous_zero_offset(k);
  v = make_contiguous_zero_offset(v);
  eval(q);
  eval(k);
  eval(v);

  if (q.dtype() != float32 || k.dtype() != float16 || v.dtype() != float16) {
    return false;
  }

  array out_storage(
      {out.shape(0), out.shape(2), out.shape(1), out.shape(3)},
      float32,
      nullptr,
      {});
  out_storage.set_status(array::Status::available);
  out_storage.set_data(allocator::malloc(out_storage.nbytes()));

  try {
    std::optional<array> causal_mask;
    if (do_causal && q_len > 1) {
      causal_mask = make_flash_attention_causal_mask(q, k, s);
      *causal_mask = make_contiguous_zero_offset(*causal_mask);
      eval(*causal_mask);
      if (causal_mask->dtype() != float16) {
        return false;
      }
    }

    if (!try_dispatch_flash_attention_native_vulkan(
            q, k, v, causal_mask ? &(*causal_mask) : nullptr, out_storage, s)) {
      return false;
    }

    vulkan::synchronize_stream(s);

    array out_transposed = swapaxes(out_storage, 1, 2, s);
    eval(out_transposed);
    array out_final = out_transposed;
    if (out.dtype() != float32) {
      out_final = astype(out_final, out.dtype(), s);
    }
    eval(out_final);
    copy_gpu(out_final, out, CopyType::General, s);
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "flash_attn_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

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
  masked.set_status(array::Status::available);
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
  if (detail::in_grad_tracing() && trace_fallback_enabled()) {
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

    if (try_eval_flash_attention_vulkan(
            inputs, outputs[0], scale_, do_causal_, s)) {
      return;
    }

    array q = multiply(array(scale_, q_in.dtype()), q_in, s);
    array k = k_in;
    array v = v_in;

    const int n_q_heads = q.shape(1);
    const int n_kv_heads = k.shape(1);
    const int n_repeats = n_q_heads / n_kv_heads;

    if (n_repeats > 1) {
      q = unflatten(q, 1, {n_kv_heads, n_repeats}, s);
      k = expand_dims(k, 2, s);
      v = expand_dims(v, 2, s);
    }

    auto scores = matmul(q, swapaxes(k, -1, -2, s), s);
    if (do_causal_) {
      if (scores.dtype() != float32) {
        scores = astype(scores, float32, s);
      }
      const int n_past = k_in.shape(2) - q_in.shape(2);
      scores = apply_diag_mask_inf_vulkan(scores, n_past, s);
    }

    const Shape scores_shape = scores.shape();
    const bool collapsed_scores = scores.ndim() > 4;
    if (collapsed_scores) {
      scores = flatten(scores, 0, scores.ndim() - 3, s);
    }
    scores = softmax(scores, std::vector<int>{-1}, true, s);
    if (collapsed_scores) {
      scores = unflatten(
          scores, 0, Shape(scores_shape.begin(), scores_shape.end() - 2), s);
    }

    array v_work = v;
    if (v_work.dtype() != scores.dtype()) {
      v_work = astype(v_work, scores.dtype(), s);
    }

    auto out = matmul(scores, v_work, s);
    if (n_repeats > 1) {
      out = flatten(out, 1, 2, s);
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
