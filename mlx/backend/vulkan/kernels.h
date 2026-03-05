// Copyright © 2024 Apple Inc.

#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "mlx/array.h"
#include "mlx/backend/vulkan/allocator.h"

namespace mlx::core::vulkan {

// Shader SPIR-V data container
struct ShaderModule {
  std::vector<uint32_t> spirv_code;
  VkShaderModule module{VK_NULL_HANDLE};
  bool compiled{false};

  ~ShaderModule();
};

// Compute pipeline wrapper
struct ComputePipeline {
  VkPipeline pipeline{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptor_layout{VK_NULL_HANDLE};
  uint32_t push_constant_size{0};

  ~ComputePipeline();
};

// Kernel manager for loading and caching shaders/pipelines
class KernelManager {
 public:
  static KernelManager& get();

  // Get or create a compute pipeline for a shader
  ComputePipeline* get_pipeline(
      const std::string& shader_name,
      const std::vector<VkDescriptorSetLayoutBinding>& bindings,
      uint32_t push_constant_size = 0);

  // Get or load a shader module
  ShaderModule* get_shader(const std::string& name);

  // Register a shader from SPIR-V data (called by generated shader code)
  // data: pointer to SPIR-V bytecode (can be uint8_t or uint32_t)
  // size_bytes: size of the data in bytes
  void
  register_shader(const std::string& name, const void* data, size_t size_bytes);

  // Descriptor set management
  VkDescriptorSet allocate_descriptor_set(VkDescriptorSetLayout layout);
  void free_descriptor_set(VkDescriptorSet set);
  void defer_descriptor_set_free(int stream_index, VkDescriptorSet set);
  void reclaim_descriptor_sets(int stream_index);
  void reclaim_all_descriptor_sets();

  // Clean up all resources
  void cleanup();

 private:
  KernelManager() = default;
  ~KernelManager();

  VkShaderModule compile_shader(const std::vector<uint32_t>& spirv);

  std::unordered_map<std::string, std::unique_ptr<ShaderModule>> shaders_;
  std::unordered_map<std::string, std::unique_ptr<ComputePipeline>> pipelines_;

  // Descriptor pool for allocating descriptor sets
  VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
  bool descriptor_pool_initialized_{false};
  std::unordered_map<int, std::vector<VkDescriptorSet>>
      deferred_descriptor_sets_;
  std::mutex deferred_descriptor_sets_mutex_;

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

enum class BinaryDispatchVariant {
  Standard,
  AddWithPartials,
};

// Helper functions for kernel dispatch
void dispatch_binary_op(
    const array& a,
    const array& b,
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s,
    BinaryDispatchVariant variant = BinaryDispatchVariant::Standard);

void dispatch_unary_op(
    const array& in,
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s,
    float param1 = 0.0f,
    float param2 = 0.0f);

void dispatch_generic_unary_op(
    const array& in,
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s,
    float param1 = 0.0f,
    float param2 = 0.0f,
    float param3 = 0.0f,
    float param4 = 0.0f);

void dispatch_arange_op(
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s,
    float start,
    float step);

// Get workgroup dimensions for element-wise operations.
// Returns (workgroup_count_x, workgroup_count_y, workgroup_count_z)
// using ggml's 512-element tiling expected by get_idx().
std::tuple<uint32_t, uint32_t, uint32_t> get_element_wise_grid_dims(
    size_t num_elements,
    uint32_t tile_size);

// Logical tile size used by generic Vulkan indexing helpers.
constexpr uint32_t VULKAN_INDEX_TILE_SIZE = 512;

} // namespace mlx::core::vulkan

// Registration macro for shader data
#define MLX_VULKAN_REGISTER_SHADER(name, data, size)           \
  namespace {                                                  \
  struct ShaderRegistrar_##name {                              \
    ShaderRegistrar_##name() {                                 \
      mlx::core::vulkan::KernelManager::get().register_shader( \
          #name, data, size);                                  \
    }                                                          \
  } shader_registrar_##name;                                   \
  }
