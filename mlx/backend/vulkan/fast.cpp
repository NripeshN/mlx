// Copyright © 2024 Apple Inc.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/quantized.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/transforms_impl.h"

namespace mlx::core {

using namespace vk;

namespace fast {

namespace {

constexpr char kFlashAttnMaskOptScratchLane[] = "flash_attn.mask_opt";
constexpr char kFlashAttnSplitKScratchLane[] = "flash_attn.split_k";
constexpr char kFlashAttnOutScratchLane[] = "flash_attn.out_storage";
constexpr char kFlashAttnCausalMaskScratchLane[] = "flash_attn.causal_mask";
constexpr char kFlashAttnKCastScratchLane[] = "flash_attn.k_cast";
constexpr char kFlashAttnVCastScratchLane[] = "flash_attn.v_cast";

array apply_diag_mask_inf_vulkan(const array& scores, int n_past, Stream s);

struct FlashAttentionCausalMaskCacheEntry {
  Shape shape;
  bool valid{false};
};

std::mutex flash_attention_causal_mask_cache_mutex;
std::unordered_map<int, FlashAttentionCausalMaskCacheEntry>
    flash_attention_causal_mask_cache;

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

bool trace_flash_attention_debug_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_TRACE_FLASH_ATTN");
        env != nullptr) {
      return std::string(env) != "0";
    }
    return false;
  }();
  return enabled;
}

void trace_flash_attention_debug(const std::string& msg) {
  if (!trace_flash_attention_debug_enabled()) {
    return;
  }
  std::cerr << "[vulkan-fa-debug] " << msg << std::endl;
}

void trace_flash_attention_array(const char* label, const array& arr) {
  if (!trace_flash_attention_debug_enabled()) {
    return;
  }
  std::ostringstream oss;
  oss << label << " shape=" << arr.shape() << " dtype=" << arr.dtype()
      << " status=" << arr.status() << " has_primitive=" << arr.has_primitive()
      << " row_contig=" << arr.flags().row_contiguous
      << " offset=" << arr.offset();
  trace_flash_attention_debug(oss.str());
}

array cast_flash_attention_kv_to_f16(
    const array& x,
    const char* scratch_lane,
    Stream s) {
  if (x.dtype() == float16) {
    return x;
  }
  array out =
      vulkan::acquire_scratch_array(s, scratch_lane, x.shape(), float16);
  copy_gpu(x, out, CopyType::General, s);
  vulkan::mark_scratch_array_written(s, scratch_lane);
  return out;
}

struct FlashAttentionTuningParams {
  enum class Path {
    Scalar,
    CoopMat1,
  };

  Path path;
  uint32_t workgroup_size;
  uint32_t block_rows;
  uint32_t block_cols;
  uint32_t d_split;
  uint32_t row_split;
  uint32_t subgroup_size;
  uint32_t shmem_staging;
  uint32_t limit_occupancy_shmem;
};

vulkan::StaticShaderId flash_attention_main_shader(
    Dtype q_dtype,
    FlashAttentionTuningParams::Path path) {
  if (const char* env = std::getenv("MLX_VULKAN_FLASH_ATTN_SHADER");
      env != nullptr) {
    const std::string value(env);
    if (value == "cm1") {
      switch (q_dtype) {
        case float16:
          return vulkan::StaticShaderId::flash_attn_f16_f16_f16_cm1;
        case bfloat16:
          return vulkan::StaticShaderId::flash_attn_bf16_f16_f16_cm1;
        default:
          return vulkan::StaticShaderId::flash_attn_f32_f16_f16_cm1;
      }
    }
    if (value == "fp32") {
      switch (q_dtype) {
        case float16:
          return vulkan::StaticShaderId::flash_attn_f16_f16_f16_fp32;
        case bfloat16:
          return vulkan::StaticShaderId::flash_attn_bf16_f16_f16_fp32;
        default:
          return vulkan::StaticShaderId::flash_attn_f32_f16_f16_fp32;
      }
    }
    if (value == "f16acc") {
      if (path == FlashAttentionTuningParams::Path::CoopMat1) {
        switch (q_dtype) {
          case float16:
            return vulkan::StaticShaderId::flash_attn_f16_f16_f16_f16acc_cm1;
          case bfloat16:
            return vulkan::StaticShaderId::flash_attn_bf16_f16_f16_f16acc_cm1;
          default:
            return vulkan::StaticShaderId::flash_attn_f32_f16_f16_f16acc_cm1;
        }
      }
      switch (q_dtype) {
        case float16:
          return vulkan::StaticShaderId::flash_attn_f16_f16_f16_f16acc;
        case bfloat16:
          return vulkan::StaticShaderId::flash_attn_bf16_f16_f16_f16acc;
        default:
          return vulkan::StaticShaderId::flash_attn_f32_f16_f16_f16acc;
      }
    }
  }
  if (path == FlashAttentionTuningParams::Path::CoopMat1) {
    switch (q_dtype) {
      case float16:
        return vulkan::StaticShaderId::flash_attn_f16_f16_f16_cm1;
      case bfloat16:
        return vulkan::StaticShaderId::flash_attn_bf16_f16_f16_cm1;
      default:
        return vulkan::StaticShaderId::flash_attn_f32_f16_f16_cm1;
    }
  }
  if (vulkan::VulkanContext::get().shader_float16_supported()) {
    switch (q_dtype) {
      case float16:
        return vulkan::StaticShaderId::flash_attn_f16_f16_f16;
      case bfloat16:
        return vulkan::StaticShaderId::flash_attn_bf16_f16_f16;
      default:
        return vulkan::StaticShaderId::flash_attn_f32_f16_f16;
    }
  }
  switch (q_dtype) {
    case float16:
      return vulkan::StaticShaderId::flash_attn_f16_f16_f16_fp32;
    case bfloat16:
      return vulkan::StaticShaderId::flash_attn_bf16_f16_f16_fp32;
    default:
      return vulkan::StaticShaderId::flash_attn_f32_f16_f16_fp32;
  }
}

struct FlashAttentionExecutionPlan {
  FlashAttentionTuningParams tuning;
  bool aligned;
  bool use_mask_opt;
  uint32_t split_kv;
  uint32_t split_k;
};

constexpr uint32_t kVendorIdAmd = 0x1002u;
constexpr uint32_t kVendorIdIntel = 0x8086u;
constexpr uint32_t kVendorIdNvidia = 0x10DEu;

uint32_t lowest_set_bit(uint32_t value) {
  return value & (~value + 1u);
}

std::pair<uint32_t, uint32_t> vulkan_device_vendor_and_subgroup_size() {
  vk::PhysicalDeviceProperties props =
      vulkan::VulkanContext::get().physical_device().getProperties();

  VkPhysicalDeviceSubgroupProperties subgroup_props{};
  subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  PhysicalDeviceProperties2 props2{};
  props2.pNext = &subgroup_props;
  vulkan::VulkanContext::get().physical_device().getProperties2(&props2);

  return {props.vendorID, subgroup_props.subgroupSize};
}

uint32_t max_compute_shared_memory_size() {
  vk::PhysicalDeviceProperties props =
      vulkan::VulkanContext::get().physical_device().getProperties();
  return props.limits.maxComputeSharedMemorySize;
}

bool flash_attention_scalar_shmem_supported(
    const FlashAttentionTuningParams& params,
    uint32_t hsk,
    uint32_t hsv) {
  const uint32_t float_type_size =
      vulkan::VulkanContext::get().shader_float16_supported() ? 2u : 4u;
  const uint32_t tmpsh = params.workgroup_size * sizeof(float);
  const uint32_t tmpshv4 = params.workgroup_size * 4u * float_type_size;
  const uint32_t masksh =
      params.block_cols * (params.block_rows + 1u) * float_type_size;
  const uint32_t qf =
      params.block_rows * (hsk / 4u + 1u) * 4u * float_type_size;
  const uint32_t d = std::max(hsk, hsv);
  const uint32_t kvsh = params.shmem_staging
      ? params.block_cols * (d / 4u + 1u) * 4u * float_type_size
      : 4u * float_type_size;
  return tmpsh + tmpshv4 + masksh + qf + kvsh <=
      max_compute_shared_memory_size();
}

bool flash_attention_coopmat_shmem_supported(
    const FlashAttentionTuningParams& params,
    uint32_t hsk,
    uint32_t hsv,
    bool f32acc) {
  const uint32_t block_rows = params.block_rows;
  const uint32_t block_cols = params.block_cols;
  const uint32_t mat_block_rows = 16u;
  const uint32_t mat_block_cols = 16u;
  const uint32_t row_split = block_cols / mat_block_cols;
  const uint32_t hsk_pad = ((hsk + 15u) / 16u) * 16u;
  const uint32_t hsv_pad = ((hsv + 15u) / 16u) * 16u;
  const uint32_t acc_type_size = f32acc ? 4u : 2u;
  const uint32_t f16vec4_size = 8u;
  const uint32_t tmpsh = (block_cols / mat_block_cols) * sizeof(float);
  const uint32_t qstride = hsk_pad / 4u + 2u;
  const uint32_t qf = block_rows * qstride * f16vec4_size;
  const uint32_t psh_stride = block_rows / 4u + 2u;
  const uint32_t psh = block_cols * psh_stride * f16vec4_size;
  const uint32_t sfshstride = (hsk <= 128u) ? (block_rows + 8u) : block_rows;
  const uint32_t sfsh = block_cols * sfshstride * acc_type_size;
  const uint32_t kvshstride =
      (params.shmem_staging ? std::max(hsk_pad, hsv_pad) : mat_block_rows) /
          4u +
      2u;
  const uint32_t vsh_stride = mat_block_cols / 4u * row_split;
  const uint32_t ksh =
      ((kvshstride >= vsh_stride) ? (block_cols * kvshstride)
                                  : (block_cols * vsh_stride)) *
      f16vec4_size;
  const uint32_t osh_stride = params.row_split * mat_block_rows / 4u;
  const uint32_t pvsh = mat_block_cols * osh_stride * f16vec4_size;
  const uint32_t slope = block_rows * acc_type_size;
  return tmpsh + qf + psh + sfsh + ksh + pvsh + slope <=
      max_compute_shared_memory_size();
}

FlashAttentionTuningParams get_flash_attention_tuning_params_scalar(
    uint32_t hsk,
    uint32_t hsv,
    uint32_t n_rows,
    uint32_t n_kv) {
  auto [vendor_id, device_subgroup_size] =
      vulkan_device_vendor_and_subgroup_size();
  const uint32_t subgroup_size = std::max(device_subgroup_size, 1u);
  const bool unified_memory = vulkan::VulkanContext::get().is_unified_memory();
  const uint32_t d = hsk | hsv;

  FlashAttentionTuningParams result{};
  result.path = FlashAttentionTuningParams::Path::Scalar;
  result.subgroup_size = subgroup_size;

  uint32_t row_split_max_hsk = 64u;
  if (vendor_id == kVendorIdAmd && !unified_memory) {
    row_split_max_hsk = n_rows <= 8 ? 64u : 128u;
  }
  result.row_split = (n_rows < 4 || hsk <= row_split_max_hsk) ? 1u : 4u;

  if (subgroup_size > 32u &&
      (n_rows < 4 || hsk < (result.row_split == 1 ? 128u : 64u))) {
    result.workgroup_size = subgroup_size * 2u;
  } else {
    result.workgroup_size = subgroup_size * 4u;
  }
  result.workgroup_size = std::clamp(result.workgroup_size, 32u, 256u);

  const bool reduce_block_rows =
      (d & 8u) != 0 || n_kv < 1024u || vendor_id == kVendorIdIntel;
  if (n_rows == 1) {
    result.block_rows = 1u;
    result.block_cols = 64u;
  } else if (result.row_split == 1u) {
    result.block_rows =
        n_rows == 2 ? 2u : ((n_rows <= 4 || reduce_block_rows) ? 4u : 8u);
    result.block_cols = (d & 8u) ? 64u : 32u;
  } else {
    result.block_rows =
        n_rows <= 4 ? 4u : ((n_rows <= 8 || reduce_block_rows) ? 8u : 16u);
    result.block_cols = (d & 8u) ? 64u : 32u;
  }

  const uint32_t d_lsb = lowest_set_bit(d);
  result.d_split = std::min(std::min(subgroup_size, 8u), d_lsb / 4u);
  result.shmem_staging =
      (vendor_id == kVendorIdNvidia && hsk < 256u && hsv < 256u) ? 1u : 0u;

  if (const char* env = std::getenv("MLX_VULKAN_FLASH_ATTN_OCCUPANCY_LIMIT");
      env != nullptr) {
    result.limit_occupancy_shmem = std::stoul(env);
  } else {
    result.limit_occupancy_shmem = 0u;
    if (vendor_id == kVendorIdAmd && n_rows >= 64u && hsk <= 128u) {
      // Subgroups per WG
      const uint32_t num_subgroups =
          result.workgroup_size / result.subgroup_size;
      // Approx. maximum limit 4 subgroups per SIMD via dummy shmem allocation
      const uint32_t target_subgroups_per_simd = 4;
      result.limit_occupancy_shmem =
          30 * 1024 / target_subgroups_per_simd / num_subgroups;
    }
  }

  if (!reduce_block_rows &&
      !flash_attention_scalar_shmem_supported(result, hsk, hsv)) {
    result.block_rows = std::max(result.block_rows / 2u, 1u);
  }

  return result;
}

FlashAttentionTuningParams get_flash_attention_tuning_params_coopmat1(
    uint32_t hsk,
    uint32_t hsv) {
  auto [vendor_id, device_subgroup_size] =
      vulkan_device_vendor_and_subgroup_size();
  const uint32_t subgroup_size = std::max(device_subgroup_size, 1u);
  const uint32_t d = hsk | hsv;

  FlashAttentionTuningParams result{};
  result.path = FlashAttentionTuningParams::Path::CoopMat1;
  result.block_rows = 16u;
  result.block_cols = 64u;
  result.row_split = 4u;
  result.subgroup_size = subgroup_size;
  result.workgroup_size = 4u * subgroup_size;
  result.d_split =
      std::min(std::min(subgroup_size, 8u), lowest_set_bit(d) / 4u);
  result.shmem_staging =
      (vendor_id == kVendorIdNvidia && hsk < 256u && hsv < 256u) ? 1u : 0u;

  if (const char* env = std::getenv("MLX_VULKAN_FLASH_ATTN_OCCUPANCY_LIMIT");
      env != nullptr) {
    result.limit_occupancy_shmem = std::stoul(env);
  } else {
    result.limit_occupancy_shmem = 0u;
  }
  return result;
}

FlashAttentionTuningParams get_flash_attention_tuning_params(
    uint32_t hsk,
    uint32_t hsv,
    uint32_t n_rows,
    uint32_t n_kv) {
  if (n_rows > 1 &&
      vulkan::VulkanContext::get().coopmat_flash_attention_f32acc_supported()) {
    auto coopmat = get_flash_attention_tuning_params_coopmat1(hsk, hsv);
    if (flash_attention_coopmat_shmem_supported(coopmat, hsk, hsv, true)) {
      return coopmat;
    }
  }
  return get_flash_attention_tuning_params_scalar(hsk, hsv, n_rows, n_kv);
}

uint32_t round_up_multiple(uint32_t value, uint32_t multiple) {
  if (multiple == 0) {
    return value;
  }
  return ((value + multiple - 1u) / multiple) * multiple;
}

FlashAttentionExecutionPlan make_flash_attention_execution_plan(
    uint32_t hsk,
    uint32_t hsv,
    uint32_t q_len,
    uint32_t q_heads,
    uint32_t kv_len,
    uint32_t batch,
    bool has_mask,
    bool do_causal,
    uint32_t q_stride,
    uint32_t k_stride,
    uint32_t v_stride) {
  auto tuning = get_flash_attention_tuning_params(hsk, hsv, q_len, kv_len);
  const bool aligned = (kv_len % tuning.block_cols) == 0 &&
      (q_stride & 7u) == 0 && (k_stride & 7u) == 0 && (v_stride & 7u) == 0;

  FlashAttentionExecutionPlan plan{
      tuning,
      aligned,
      false,
      kv_len,
      1u,
  };

  if (has_mask && !do_causal) {
    plan.use_mask_opt = q_len >= 32u && kv_len >= 128u &&
        static_cast<uint64_t>(q_len) * static_cast<uint64_t>(kv_len) >= 32768u;
  }

  const uint32_t tr = (q_len + tuning.block_rows - 1u) / tuning.block_rows;
  const uint32_t total_wgs_no_split = std::max(tr * q_heads * batch, 1u);
  const uint32_t target_workgroups = 32u;
  uint32_t split_k = total_wgs_no_split < target_workgroups
      ? (target_workgroups + total_wgs_no_split - 1u) / total_wgs_no_split
      : 1u;
  split_k = std::min(split_k, 8u);
  if (has_mask && split_k > 1u) {
    split_k = std::min(split_k, 4u);
  }
  if (split_k > 1u && kv_len >= 2u * tuning.block_cols) {
    uint32_t split_kv =
        round_up_multiple(std::max(1u, kv_len / split_k), tuning.block_cols);
    split_k = (kv_len + split_kv - 1u) / split_kv;
    if (split_k > 1u) {
      plan.split_kv = split_kv;
      plan.split_k = split_k;
    }
  }

  return plan;
}

void insert_compute_barrier(VkCommandBuffer command_buffer) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask =
      static_cast<VkAccessFlags>(vk::AccessFlagBits::eShaderWrite);
  barrier.dstAccessMask = static_cast<VkAccessFlags>(
      vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
  vkCmdPipelineBarrier(
      command_buffer,
      static_cast<VkPipelineStageFlags>(
          vk::PipelineStageFlagBits::eComputeShader),
      static_cast<VkPipelineStageFlags>(
          vk::PipelineStageFlagBits::eComputeShader),
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);
}

array make_flash_attention_causal_mask(
    const array& q,
    const array& k,
    Stream s) {
  Shape shape = {q.shape(0), 1, q.shape(2), k.shape(2)};
  array mask = vulkan::acquire_scratch_array(
      s, kFlashAttnCausalMaskScratchLane, shape, float16);

  bool needs_refresh = true;
  {
    std::lock_guard<std::mutex> lock(flash_attention_causal_mask_cache_mutex);
    auto& entry = flash_attention_causal_mask_cache[s.index];
    needs_refresh = !entry.valid || entry.shape != shape;
    entry.shape = shape;
    entry.valid = true;
  }

  if (!needs_refresh) {
    mask.set_status(array::Status::available);
    return mask;
  }

  array mask_f32 = zeros(shape, float32, s);
  if (mask_f32.dtype() != float32) {
    throw std::runtime_error("Unexpected causal mask dtype.");
  }
  const int n_past = k.shape(2) - q.shape(2);
  mask_f32 = apply_diag_mask_inf_vulkan(mask_f32, n_past, s);
  array mask_f16 = astype(mask_f32, float16, s);
  eval(mask_f16);
  copy_gpu_inplace(mask_f16, mask, CopyType::General, s);
  mask.set_status(array::Status::evaluated);
  vulkan::mark_scratch_array_written(s, kFlashAttnCausalMaskScratchLane);
  return mask;
}

bool try_dispatch_flash_attention_native_vulkan(
    const array& q,
    const array& k,
    const array& v,
    const array* mask,
    bool do_causal,
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
  const uint32_t q_stride =
      checked_u32_size(q.strides(2), "flash_attn q_stride");
  const uint32_t k_stride =
      checked_u32_size(k.strides(2), "flash_attn k_stride");
  const uint32_t v_stride =
      checked_u32_size(v.strides(2), "flash_attn v_stride");

  const auto plan = make_flash_attention_execution_plan(
      hsk,
      hsv,
      q_len,
      q_heads,
      kv_len,
      batch,
      has_mask,
      do_causal,
      q_stride,
      k_stride,
      v_stride);
  const auto& tuning = plan.tuning;
  const auto shader_id = flash_attention_main_shader(q.dtype(), tuning.path);
  if (tuning.d_split == 0 || tuning.block_rows == 0 || tuning.block_cols == 0 ||
      tuning.row_split == 0 || hsk % tuning.d_split != 0 ||
      hsv % tuning.d_split != 0 ||
      tuning.block_cols %
              (tuning.workgroup_size / tuning.d_split / tuning.row_split) !=
          0) {
    return false;
  }

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
  push_constants.split_kv = plan.split_kv;
  push_constants.k_num = plan.split_k;

  std::optional<array> mask_opt;
  if (plan.use_mask_opt) {
    const uint32_t mask_opt_num_dwords =
        (kv_len + 16u * tuning.block_cols - 1u) / (16u * tuning.block_cols);
    const uint32_t mask_rows =
        (push_constants.nem1 + tuning.block_rows - 1u) / tuning.block_rows;
    const uint64_t mask_opt_elems = static_cast<uint64_t>(mask_opt_num_dwords) *
        mask_rows * push_constants.nem2 * push_constants.nem3;
    if (mask_opt_elems >
        static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return false;
    }
    mask_opt = vulkan::acquire_scratch_array(
        s,
        kFlashAttnMaskOptScratchLane,
        {static_cast<int>(mask_opt_elems)},
        uint32);
  }

  std::optional<array> split_k_tmp;
  if (plan.split_k > 1u) {
    const uint64_t out_elems =
        static_cast<uint64_t>(hsv) * q_heads * q_len * batch;
    const uint64_t lm_elems =
        static_cast<uint64_t>(q_heads) * 2u * q_len * batch;
    const uint64_t split_k_elems = (out_elems + lm_elems) * plan.split_k;
    if (split_k_elems >
        static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return false;
    }
    split_k_tmp = vulkan::acquire_scratch_array(
        s,
        kFlashAttnSplitKScratchLane,
        {static_cast<int>(split_k_elems)},
        float32);
  }

  const std::vector<uint32_t> specialization_constants = {
      tuning.workgroup_size,
      tuning.block_rows,
      tuning.block_cols,
      hsk,
      hsv,
      plan.aligned ? 0u : 1u,
      tuning.d_split,
      tuning.row_split,
      tuning.subgroup_size,
      tuning.shmem_staging,
      (plan.use_mask_opt ? 1u : 0u) | (has_mask ? 2u : 0u) |
          (do_causal ? 8u : 0u),
      tuning.limit_occupancy_shmem,
  };

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);

    if (plan.use_mask_opt) {
      vulkan::FlashAttentionMaskOptPushConstants mask_opt_push_constants{};
      mask_opt_push_constants.nem0 = kv_len;
      mask_opt_push_constants.nem1 = push_constants.nem1;
      mask_opt_push_constants.nem2 = push_constants.nem2;
      mask_opt_push_constants.nbm1 =
          checked_u32_size((*mask).strides(2), "flash_attn mask_nbm1");
      mask_opt_push_constants.nbm2 =
          checked_u32_size((*mask).strides(1), "flash_attn mask_nbm2");
      mask_opt_push_constants.nbm3 =
          checked_u32_size((*mask).strides(0), "flash_attn mask_nbm3");
      mask_opt_push_constants.nbd1 =
          (kv_len + 16u * tuning.block_cols - 1u) / (16u * tuning.block_cols);
      mask_opt_push_constants.nbd2 = mask_opt_push_constants.nbd1 *
          ((push_constants.nem1 + tuning.block_rows - 1u) / tuning.block_rows);
      mask_opt_push_constants.nbd3 =
          mask_opt_push_constants.nbd2 * push_constants.nem2;

      vulkan::dispatch_flash_attention_mask_opt_op(
          *mask,
          *mask_opt,
          vulkan::StaticShaderId::fa_mask_opt,
          command_buffer,
          s,
          mask_opt_push_constants,
          {
              mask_opt_push_constants.nbd1,
              (push_constants.nem1 + tuning.block_rows - 1u) /
                  tuning.block_rows,
              push_constants.nem2 * push_constants.nem3,
          },
          {
              128u,
              std::max(1u, 128u / std::max(tuning.subgroup_size, 1u)),
              tuning.block_rows,
              tuning.block_cols,
          });
      insert_compute_barrier(command_buffer);
      vulkan::mark_scratch_array_written(s, kFlashAttnMaskOptScratchLane);
    }

    array& flash_output = plan.split_k > 1u ? *split_k_tmp : out_storage;
    vulkan::dispatch_flash_attention_op(
        q,
        k,
        v,
        has_mask ? *mask : q,
        q,
        flash_output,
        plan.use_mask_opt ? *mask_opt : q,
        shader_id,
        command_buffer,
        s,
        push_constants,
        {
            ((q_len + tuning.block_rows - 1u) / tuning.block_rows) *
                plan.split_k,
            q_heads,
            batch,
        },
        specialization_constants);

    if (plan.split_k > 1u) {
      insert_compute_barrier(command_buffer);
      vulkan::FlashAttentionSplitKReducePushConstants reduce_push_constants{};
      reduce_push_constants.D = hsv;
      reduce_push_constants.ne1 = q_heads;
      reduce_push_constants.ne2 = q_len;
      reduce_push_constants.ne3 = batch;
      reduce_push_constants.k_num = plan.split_k;
      reduce_push_constants.sinks = 0u;
      vulkan::dispatch_flash_attention_split_k_reduce_op(
          *split_k_tmp,
          q,
          out_storage,
          vulkan::StaticShaderId::fa_split_k_reduce,
          command_buffer,
          s,
          reduce_push_constants,
          {q_heads, (hsv + 31u) / 32u, q_len * batch},
          {32u});
      vulkan::mark_scratch_array_written(s, kFlashAttnSplitKScratchLane);
    }

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

  const bool supported_kv_dtype =
      (k.dtype() == float16 || k.dtype() == bfloat16) &&
      (v.dtype() == float16 || v.dtype() == bfloat16);
  if (!supported_kv_dtype || out.ndim() != 4 || q.ndim() != 4 ||
      k.ndim() != 4 || v.ndim() != 4) {
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
  if (q.dtype() == float16) {
    q = astype(q, float32, s);
  }

  auto make_contiguous_zero_offset = [&](array x) {
    if (!x.flags().row_contiguous || x.strides().back() != 1 ||
        x.offset() != 0) {
      x = contiguous_copy_gpu(x, s);
    }
    return x;
  };

  auto debug_eval = [&](array& x, const char* label) {
    trace_flash_attention_array(label, x);
    eval(x);
    trace_flash_attention_array(label, x);
  };

  q = make_contiguous_zero_offset(q);
  k = make_contiguous_zero_offset(k);
  v = make_contiguous_zero_offset(v);
  k = cast_flash_attention_kv_to_f16(k, kFlashAttnKCastScratchLane, s);
  v = cast_flash_attention_kv_to_f16(v, kFlashAttnVCastScratchLane, s);
  debug_eval(q, "q_ready");
  debug_eval(k, "k_ready");
  debug_eval(v, "v_ready");

  const bool supported_q_dtype = q.dtype() == float32 || q.dtype() == bfloat16;
  if (!supported_q_dtype || k.dtype() != float16 || v.dtype() != float16) {
    return false;
  }

  array out_storage = vulkan::acquire_scratch_array(
      s,
      kFlashAttnOutScratchLane,
      {out.shape(0), out.shape(2), out.shape(1), out.shape(3)},
      float32);

  try {
    const bool use_causal_shader = do_causal && q_len > 1;

    if (!try_dispatch_flash_attention_native_vulkan(
            q, k, v, nullptr, use_causal_shader, out_storage, s)) {
      return false;
    }
    vulkan::mark_scratch_array_written(s, kFlashAttnOutScratchLane);

    array out_transposed = swapaxes_in_eval(out_storage, 1, 2);
    trace_flash_attention_array("out_storage", out_storage);
    trace_flash_attention_array("out_transposed", out_transposed);
    if (out.dtype() == float32) {
      copy_gpu(out_transposed, out, CopyType::General, s);
      out.set_status(array::Status::evaluated);
      return true;
    }

    array out_final = astype(out_transposed, out.dtype(), s);
    debug_eval(out_final, "out_final");
    if (out.shape() == out_final.shape()) {
      out.copy_shared_buffer(out_final);
    } else {
      copy_gpu(out_final, out, CopyType::General, s);
      out.set_status(array::Status::evaluated);
    }
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
      vulkan::StaticShaderId::diag_mask_inf_f32,
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
  if (x.ndim() == 0 || x.shape() != out.shape()) {
    return false;
  }

  const bool has_weight = w.ndim() != 0;

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

  if (w.ndim() > 4) {
    return false;
  }

  const uint32_t axis_size =
      checked_u32_size(x.shape(x.ndim() - 1), "axis_size");
  if (axis_size == 0 || axis_size > 32u * 512u) {
    return false;
  }

  if (!x.flags().contiguous || x.strides().back() != 1) {
    x = contiguous_copy_gpu(x, s);
  }
  if (has_weight && (!w.flags().row_contiguous || w.offset() != 0)) {
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
        vulkan::StaticShaderId::rms_norm_f32,
        command_buffer,
        s,
        vulkan::BinaryDispatchVariant::Standard,
        std::array<uint32_t, 3>{nrows, nchannels, nsamples},
        {0u, has_weight ? 1u : 0u});
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
  if (inputs.size() < 3 || outputs.empty()) {
    throw std::runtime_error(
        "ScaledDotProductAttention expects at least 3 inputs and 1 output.");
  }

  const array& q_in = inputs[0];
  const array& k_in = inputs[1];
  const array& v_in = inputs[2];
  const bool has_arr_mask = inputs.size() > static_cast<size_t>(3 + has_sinks_);

  std::string reason;
  if (!sdpa_vulkan_supported(
          q_in,
          k_in,
          v_in,
          has_arr_mask || do_causal_,
          has_arr_mask,
          do_causal_,
          false,
          output_logsumexp_,
          has_sinks_,
          stream(),
          &reason)) {
    throw std::runtime_error(
        std::string("ScaledDotProductAttention not supported on Vulkan: ") +
        reason);
  }

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

  array scores_work = scores;
  if (n_repeats > 1) {
    Shape v_shape = scores.shape();
    v_shape.back() = v_work.shape(-1);
    v_work = broadcast_to(v_work, v_shape, s);
    scores_work = flatten(scores_work, 1, 2, s);
    v_work = flatten(v_work, 1, 2, s);
    if (!scores_work.flags().row_contiguous || scores_work.offset() != 0 ||
        scores_work.strides().back() != 1) {
      scores_work = contiguous_copy_gpu(scores_work, s);
    }
    if (!v_work.flags().row_contiguous || v_work.offset() != 0 ||
        v_work.strides().back() != 1) {
      v_work = contiguous_copy_gpu(v_work, s);
    }
  }

  auto result = matmul(scores_work, v_work, s);
  if (result.dtype() != outputs[0].dtype()) {
    result = astype(result, outputs[0].dtype(), s);
  }

  eval(result);

  copy_gpu(result, outputs[0], CopyType::General, s);

  if (output_logsumexp_) {
    throw std::runtime_error(
        "ScaledDotProductAttention with logsumexp output is not supported on Vulkan.");
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
  if (!try_eval_rms_norm_vulkan(inputs, outputs[0], eps_, stream())) {
    throw std::runtime_error("RMSNorm failed on Vulkan.");
  }
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("[RMSNormVJP::eval_gpu] Not implemented.");
}

void ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval_cpu_fallback_multi_with_state_on_stream<ConvertFP8>(
      inputs, outputs, stream(), state());
}

void Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (dequantize_) {
    if (mode_ != QuantizationMode::Affine) {
      if (mode_ == QuantizationMode::Nvfp4 && inputs.size() == 2 &&
          outputs.size() == 1) {
        auto& out = outputs[0];
        array out_f32(out.shape(), float32, nullptr, {});
        if (!vulkan::nvfp4_dequantize_to_float32(
                inputs[0], inputs[1], out_f32, stream())) {
          throw std::runtime_error(
              "[Quantize::eval_gpu] Nvfp4 dequantize failed on Vulkan.");
        }

        out.set_data(allocator::malloc(out.nbytes()));
        copy_gpu(out_f32, out, CopyType::General, stream());
        return;
      }

      auto fallback_inputs = inputs;
      if (fallback_inputs.size() > 1 &&
          !fallback_inputs[1].flags().row_contiguous) {
        fallback_inputs[1] = contiguous_copy_gpu(fallback_inputs[1], stream());
      }
      auto fallback_outputs = fallback_(fallback_inputs);
      for (size_t i = 0; i < outputs.size(); ++i) {
        array staged(
            fallback_outputs[i].shape(),
            fallback_outputs[i].dtype(),
            nullptr,
            {});
        copy_gpu(fallback_outputs[i], staged, CopyType::General, stream());
        outputs[i].overwrite_descriptor(staged);
      }
      return;
    }
    if (inputs.size() != 3) {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Expected affine dequantize inputs to include biases.");
    }

    auto& wq = inputs[0];
    auto& scales = inputs[1];
    auto& biases = inputs[2];
    auto& out = outputs[0];

    auto& s = stream();

    array scales_f32 = scales;
    if (scales_f32.dtype() != float32) {
      scales_f32 = array(scales.shape(), float32, nullptr, {});
      scales_f32.set_data(allocator::malloc(scales_f32.nbytes()));
      copy_gpu(scales, scales_f32, CopyType::General, s);
    }

    array biases_f32 = biases;
    if (biases_f32.dtype() != float32) {
      biases_f32 = array(biases.shape(), float32, nullptr, {});
      biases_f32.set_data(allocator::malloc(biases_f32.nbytes()));
      copy_gpu(biases, biases_f32, CopyType::General, s);
    }

    array out_f32(out.shape(), float32, nullptr, {});
    if (!vulkan::affine_dequantize_to_float32(
            wq, scales_f32, biases_f32, out_f32, s, group_size_, bits_)) {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Affine dequantize failed on Vulkan.");
    }

    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu(out_f32, out, CopyType::General, s);
  } else {
    if (mode_ != QuantizationMode::Affine) {
      auto fallback_outputs = eval_fallback_outputs(fallback_, inputs);
      for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i].copy_shared_buffer(fallback_outputs[i]);
      }
      return;
    }
    if (outputs.size() != 3 || inputs.size() != 1) {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Expected affine quantize outputs for weights, scales, and biases.");
    }

    auto& s = stream();
    array in_f32 = inputs[0];
    if (in_f32.dtype() != float32) {
      in_f32 = array(inputs[0].shape(), float32, nullptr, {});
      in_f32.set_data(allocator::malloc(in_f32.nbytes()));
      copy_gpu(inputs[0], in_f32, CopyType::General, s);
    }

    array scales_f32(outputs[1].shape(), float32, nullptr, {});
    array biases_f32(outputs[2].shape(), float32, nullptr, {});
    if (!vulkan::affine_quantize_from_float32(
            in_f32,
            outputs[0],
            scales_f32,
            biases_f32,
            s,
            group_size_,
            bits_)) {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Affine quantize failed on Vulkan.");
    }

    outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));
    outputs[2].set_data(allocator::malloc(outputs[2].nbytes()));
    copy_gpu(scales_f32, outputs[1], CopyType::General, s);
    copy_gpu(biases_f32, outputs[2], CopyType::General, s);
  }
}

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("CustomKernel has no Vulkan implementation.");
}

} // namespace fast

} // namespace mlx::core
