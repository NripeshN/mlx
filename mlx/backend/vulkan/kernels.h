// Copyright © 2024 Apple Inc.

#pragma once

#include <vulkan/vulkan.hpp>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "mlx/array.h"
#include "mlx/backend/vulkan/allocator.h"
#include "vulkan_shaders.hpp"

// Forward declaration for generated enum (will be overridden by
// vulkan_shaders.hpp)
enum class StaticShaderId : uint32_t {
  Count // Placeholder - actual values defined in generated header
};

namespace mlx::core::vulkan {

// Hash function for Vulkan handle types
struct VulkanHandleHash {
  size_t operator()(vk::DescriptorSet handle) const {
    return std::hash<uintptr_t>{}(
        reinterpret_cast<uintptr_t>(static_cast<VkDescriptorSet>(handle)));
  }

  size_t operator()(vk::DescriptorSetLayout handle) const {
    return std::hash<uintptr_t>{}(reinterpret_cast<uintptr_t>(
        static_cast<VkDescriptorSetLayout>(handle)));
  }

  size_t operator()(vk::Pipeline handle) const {
    return std::hash<uintptr_t>{}(
        reinterpret_cast<uintptr_t>(static_cast<VkPipeline>(handle)));
  }

  size_t operator()(vk::PipelineLayout handle) const {
    return std::hash<uintptr_t>{}(
        reinterpret_cast<uintptr_t>(static_cast<VkPipelineLayout>(handle)));
  }

  size_t operator()(vk::ShaderModule handle) const {
    return std::hash<uintptr_t>{}(
        reinterpret_cast<uintptr_t>(static_cast<VkShaderModule>(handle)));
  }
};

// Shader SPIR-V data container
struct ShaderModule {
  std::vector<uint32_t> spirv_code;
  vk::ShaderModule module;
  bool compiled{false};
  std::string debug_name;

  ~ShaderModule();
};

// Compute pipeline wrapper
struct ComputePipeline {
  vk::Pipeline pipeline;
  vk::PipelineLayout layout;
  vk::DescriptorSetLayout descriptor_layout;
  uint32_t push_constant_size{0};

  ~ComputePipeline();
};

// Kernel manager for loading and caching shaders/pipelines
class KernelManager {
 public:
  static KernelManager& get();

  void initialize_static_registry();

  // Get or create a compute pipeline for a shader
  ComputePipeline* get_pipeline(
      StaticShaderId shader_id,
      const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
      uint32_t push_constant_size = 0,
      const std::vector<uint32_t>& specialization_constants = {});
  ComputePipeline* get_pipeline(
      const std::string& shader_name,
      const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
      uint32_t push_constant_size = 0,
      const std::vector<uint32_t>& specialization_constants = {});

  // Backward compatibility overloads for C API
  ComputePipeline* get_pipeline(
      StaticShaderId shader_id,
      const std::vector<VkDescriptorSetLayoutBinding>& bindings,
      uint32_t push_constant_size = 0,
      const std::vector<uint32_t>& specialization_constants = {});
  ComputePipeline* get_pipeline(
      const std::string& shader_name,
      const std::vector<VkDescriptorSetLayoutBinding>& bindings,
      uint32_t push_constant_size = 0,
      const std::vector<uint32_t>& specialization_constants = {});

  // Get or load a shader module
  ShaderModule* get_shader(StaticShaderId id);
  ShaderModule* get_shader(const std::string& name);

  // Register a dynamic shader from SPIR-V data.
  void
  register_shader(const std::string& name, const void* data, size_t size_bytes);

  // Descriptor set management
  vk::DescriptorSet allocate_descriptor_set(vk::DescriptorSetLayout layout);
  void free_descriptor_set(vk::DescriptorSet set);
  void defer_descriptor_set_free(int stream_index, vk::DescriptorSet set);
  void defer_descriptor_set_free(
      int stream_index,
      uint64_t submission_epoch,
      vk::DescriptorSet set);

  // Backward compatibility overloads for C API
  vk::DescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout);
  void free_descriptor_set(VkDescriptorSet set);
  void defer_descriptor_set_free(int stream_index, VkDescriptorSet set);
  void defer_descriptor_set_free(
      int stream_index,
      uint64_t submission_epoch,
      VkDescriptorSet set);
  void reclaim_descriptor_set_epoch(
      int stream_index,
      uint64_t submission_epoch);
  void reclaim_descriptor_sets(int stream_index);
  void reclaim_descriptor_sets(int stream_index, uint64_t completed_epoch);
  void reclaim_all_descriptor_sets();

  // Clean up all resources
  void cleanup();

 private:
  struct DescriptorBindingKey {
    uint32_t binding{0};
    uint32_t descriptor_type{0};
    uint32_t descriptor_count{0};
    uint32_t stage_flags{0};
    bool has_immutable_samplers{false};

    bool operator==(const DescriptorBindingKey& other) const = default;
  };

  struct PipelineKey {
    bool is_dynamic{false};
    StaticShaderId static_shader_id{StaticShaderId::Count};
    std::string dynamic_shader_name;
    std::vector<DescriptorBindingKey> bindings;
    uint32_t push_constant_size{0};
    std::vector<uint32_t> specialization_constants;

    bool operator==(const PipelineKey& other) const = default;
  };

  struct PipelineKeyHash {
    size_t operator()(const PipelineKey& key) const;
  };

  KernelManager();
  ~KernelManager();

  void ensure_static_registry_initialized();
  void register_static_shader(
      StaticShaderId id,
      const void* data,
      size_t size_bytes);
  vk::ShaderModule compile_shader(const std::vector<uint32_t>& spirv);
  static DescriptorBindingKey make_descriptor_binding_key(
      const vk::DescriptorSetLayoutBinding& binding);
  void purge_descriptor_sets_for_layouts(
      const std::unordered_set<vk::DescriptorSetLayout, VulkanHandleHash>&
          layouts);

  std::vector<std::unique_ptr<ShaderModule>> static_shaders_;
  std::unordered_map<std::string, std::unique_ptr<ShaderModule>>
      dynamic_shaders_;
  std::unordered_map<
      PipelineKey,
      std::unique_ptr<ComputePipeline>,
      PipelineKeyHash>
      pipelines_;
  bool static_registry_initialized_{false};
  std::mutex static_registry_mutex_;

  struct DescriptorSetRecord {
    vk::DescriptorSet set;
    vk::DescriptorSetLayout layout;
  };

  // Descriptor pool for allocating descriptor sets
  vk::DescriptorPool descriptor_pool_;
  bool descriptor_pool_initialized_{false};
  std::unordered_map<
      int,
      std::unordered_map<uint64_t, std::vector<DescriptorSetRecord>>>
      deferred_descriptor_sets_;
  std::mutex deferred_descriptor_sets_mutex_;
  std::unordered_map<
      vk::DescriptorSetLayout,
      std::vector<vk::DescriptorSet>,
      VulkanHandleHash>
      reusable_descriptor_sets_;
  std::unordered_map<
      vk::DescriptorSet,
      vk::DescriptorSetLayout,
      VulkanHandleHash>
      descriptor_set_layouts_;
  std::mutex descriptor_sets_mutex_;

  void init_descriptor_pool();
};

// Push constant structures used by shaders
struct BinaryPushConstants {
  uint32_t ne;
  uint32_t ne00;
  uint32_t ne01;
  uint32_t ne02;
  uint32_t ne03;
  uint32_t nb00;
  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;
  uint32_t ne10;
  uint32_t ne11;
  uint32_t ne12;
  uint32_t ne13;
  uint32_t nb10;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
  uint32_t ne20;
  uint32_t ne21;
  uint32_t ne22;
  uint32_t ne23;
  uint32_t nb20;
  uint32_t nb21;
  uint32_t nb22;
  uint32_t nb23;
  uint32_t misalign_offsets;
  float param1;
  float param2;
  int32_t param3;
};

struct TernaryPushConstants {
  uint32_t ne;
};

struct UnaryPushConstants {
  uint32_t ne;
  uint32_t ne00;
  uint32_t ne01;
  uint32_t ne02;
  uint32_t ne03;
  uint32_t nb00;
  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;
  uint32_t ne10;
  uint32_t ne11;
  uint32_t ne12;
  uint32_t ne13;
  uint32_t nb10;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
  uint32_t misalign_offsets;
  float param1;
  float param2;
  uint32_t ne0_012mp;
  uint32_t ne0_012L;
  uint32_t ne0_01mp;
  uint32_t ne0_01L;
  uint32_t ne0_0mp;
  uint32_t ne0_0L;
  uint32_t ne1_012mp;
  uint32_t ne1_012L;
  uint32_t ne1_01mp;
  uint32_t ne1_01L;
  uint32_t ne1_0mp;
  uint32_t ne1_0L;
};

struct GenericPushConstants {
  uint32_t KX;
  uint32_t KY;
  float param1;
  float param2;
  float param3;
  float param4;
};

struct SumRowsPushConstants {
  uint32_t n_cols;
  uint32_t ne01;
  uint32_t ne02;
  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
  float weight;
  uint32_t misalign_offsets;
  uint32_t ne0_12mp;
  uint32_t ne0_12L;
  uint32_t ne0_1mp;
  uint32_t ne0_1L;
};

struct SoftmaxPushConstants {
  uint32_t KX;
  uint32_t KY;
  uint32_t ne00;
  uint32_t ne01;
  uint32_t ne02;
  uint32_t ne12;
  uint32_t ne13;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
  float scale;
  float max_bias;
  float m0;
  float m1;
  uint32_t n_head_log2;
  uint32_t nrows_x;
  uint32_t has_sinks;
};

struct DiagMaskInfPushConstants {
  uint32_t ncols;
  uint32_t rows_per_channel;
  uint32_t n_past;
};

struct FlashAttentionPushConstants {
  uint32_t N;
  uint32_t KV;

  uint32_t ne1;
  uint32_t ne2;
  uint32_t ne3;

  uint32_t neq2;
  uint32_t neq3;
  uint32_t nek2;
  uint32_t nek3;
  uint32_t nev2;
  uint32_t nev3;
  uint32_t nem1;
  uint32_t nem2;
  uint32_t nem3;

  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
  uint32_t nb21;
  uint32_t nb22;
  uint32_t nb23;

  float scale;
  float max_bias;
  float logit_softcap;

  uint32_t mask_n_head_log2;
  float m0;
  float m1;

  uint32_t gqa_ratio;
  uint32_t split_kv;
  uint32_t k_num;
};

struct FlashAttentionSplitKReducePushConstants {
  uint32_t D;
  uint32_t ne1;
  uint32_t ne2;
  uint32_t ne3;
  uint32_t k_num;
  uint32_t sinks;
};

struct FlashAttentionMaskOptPushConstants {
  uint32_t nem0;
  uint32_t nem1;
  uint32_t nem2;
  uint32_t nbm1;
  uint32_t nbm2;
  uint32_t nbm3;
  uint32_t nbd1;
  uint32_t nbd2;
  uint32_t nbd3;
};

struct MatmulPushConstants {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t stride_a;
  uint32_t stride_b;
  uint32_t stride_d;
  uint32_t batch_stride_a;
  uint32_t batch_stride_b;
  uint32_t batch_stride_d;
  uint32_t base_work_group_z;
  uint32_t num_batches;
  uint32_t k_split;
  uint32_t ne02;
  uint32_t ne12;
  uint32_t broadcast2;
  uint32_t broadcast3;
  uint32_t padded_N;
};

struct MatVecPushConstants {
  uint32_t ncols;
  uint32_t stride_a;
  uint32_t stride_b;
  uint32_t stride_d;
  uint32_t batch_stride_a;
  uint32_t batch_stride_b;
  uint32_t batch_stride_d;
  uint32_t fusion_flags;
  uint32_t base_work_group_y;
  uint32_t ne02;
  uint32_t ne12;
  uint32_t broadcast2;
  uint32_t broadcast3;
};

struct RandomBitsPushConstants {
  uint32_t num_keys;
  uint32_t bytes_per_key;
  uint32_t odd;
  uint32_t out_skip;
};

struct GatherPushConstants {
  uint32_t ne;
  uint32_t slice_size;
  uint32_t axis_size;
  uint32_t index_count;
};

struct GatherAxisPushConstants {
  uint32_t ne;
  uint32_t size_pre;
  uint32_t size_axis;
  uint32_t size_post;
  uint32_t idx_axis_size;
};

struct RopePushConstants {
  uint32_t rope_mode;
  uint32_t nrows;
  uint32_t n_dims;
  float freq_scale;
  float freq_base;
  float ext_factor;
  float attn_factor;
  float corr_dims[2];
  float theta_scale;
  uint32_t has_ff;
  int32_t sections[4];
  uint32_t is_imrope;
  uint32_t is_back;
  uint32_t set_rows_stride;
  uint32_t position_stride;
  uint32_t positions_are_offsets;
  uint32_t ne00;
  uint32_t ne01;
  uint32_t ne02;
  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
};

struct AffineDequantPushConstants {
  uint32_t ne;
  uint32_t bits;
  uint32_t group_size;
};

struct AffineQuantPushConstants {
  uint32_t ne;
  uint32_t bits;
  uint32_t group_size;
};

struct Nvfp4DequantPushConstants {
  uint32_t ne;
};

enum class BinaryDispatchVariant {
  Standard,
  AddWithPartials,
};

// Helper functions for kernel dispatch
void dispatch_binary_op(
    const array& a,
    const array& b,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    BinaryDispatchVariant variant = BinaryDispatchVariant::Standard,
    std::optional<std::array<uint32_t, 3>> explicit_grid = std::nullopt,
    const std::vector<uint32_t>& specialization_constants = {});

void dispatch_binary_op(
    const array& a,
    const array& b,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    BinaryDispatchVariant variant,
    std::optional<std::array<uint32_t, 3>> explicit_grid,
    const std::vector<uint32_t>& specialization_constants,
    float param1,
    float param2 = 0.0f,
    int32_t param3 = 0);

void dispatch_ternary_op(
    const array& cond,
    const array& x,
    const array& y,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s);

void dispatch_unary_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    float param1 = 0.0f,
    float param2 = 0.0f);

void dispatch_generic_unary_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    float param1 = 0.0f,
    float param2 = 0.0f,
    float param3 = 0.0f,
    float param4 = 0.0f);

void dispatch_arange_op(
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    float start,
    float step);

void dispatch_sum_rows_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    float weight = 1.0f);

void dispatch_argmax_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s);

void dispatch_softmax_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s);

void dispatch_softmax_large_op(
    const array& in,
    array& out,
    StaticShaderId shader_id_pass1,
    StaticShaderId shader_id_pass2,
    StaticShaderId shader_id_pass3,
    vk::CommandBuffer cmd_buffer,
    const Stream& s);

void dispatch_diag_mask_inf_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t rows_per_channel,
    uint32_t n_past);

void dispatch_flash_attention_op(
    const array& q,
    const array& k,
    const array& v,
    const array& mask,
    const array& sinks,
    array& out,
    const array& mask_opt,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const FlashAttentionPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid,
    const std::vector<uint32_t>& specialization_constants);

void dispatch_flash_attention_split_k_reduce_op(
    const array& in,
    const array& sinks,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const FlashAttentionSplitKReducePushConstants& push_constants,
    const std::array<uint32_t, 3>& grid,
    const std::vector<uint32_t>& specialization_constants = {});

void dispatch_flash_attention_mask_opt_op(
    const array& mask,
    array& mask_opt,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const FlashAttentionMaskOptPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid,
    const std::vector<uint32_t>& specialization_constants = {});

void dispatch_cumsum_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s);

void dispatch_mul_mm_op(
    const array& a,
    const array& b,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const MatmulPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid);

void dispatch_mul_mat_vec_op(
    const array& matrix,
    const array& vec,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s);

void dispatch_random_bits_op(
    const array& keys,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const RandomBitsPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid);

void dispatch_gather_op(
    const array& src,
    const array& indices,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t slice_size,
    uint32_t axis_size,
    uint32_t index_count);

void dispatch_gather_axis_op(
    const array& src,
    const array& indices,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t size_pre,
    uint32_t size_axis,
    uint32_t size_post,
    uint32_t idx_axis_size);

void dispatch_scatter_axis_op(
    const array& updates,
    const array& indices,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t size_pre,
    uint32_t size_axis,
    uint32_t size_post,
    uint32_t idx_axis_size);

void dispatch_rope_op(
    const array& in,
    const array& positions,
    const array& freqs,
    array& out,
    const array& indices,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const RopePushConstants& push_constants,
    const std::array<uint32_t, 3>& grid);

void dispatch_affine_dequant_op(
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const AffineDequantPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid);

void dispatch_affine_quant_op(
    const array& in,
    array& w,
    array& scales,
    array& biases,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const AffineQuantPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid);

void dispatch_nvfp4_dequant_op(
    const array& w,
    const array& scales,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const Nvfp4DequantPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid);

// Get workgroup dimensions for element-wise operations.
// Returns (workgroup_count_x, workgroup_count_y, workgroup_count_z)
// using ggml's 512-element tiling expected by get_idx().
std::tuple<uint32_t, uint32_t, uint32_t> get_element_wise_grid_dims(
    size_t num_elements,
    uint32_t tile_size);

// Logical tile size used by generic Vulkan indexing helpers.
constexpr uint32_t VULKAN_INDEX_TILE_SIZE = 512;

} // namespace mlx::core::vulkan
