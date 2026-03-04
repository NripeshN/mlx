// Copyright © 2024 Apple Inc.

#pragma once

#include <vulkan/vulkan.h>
#include <memory>
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

  void init_descriptor_pool();
};

// Push constant structures used by shaders
struct BinaryPushConstants {
  uint32_t ne; // Total number of elements
  uint32_t ne00; // Dimension 0 of src0
  uint32_t ne01; // Dimension 1 of src0
  uint32_t ne02; // Dimension 2 of src0
  uint32_t ne03; // Dimension 3 of src0
  uint32_t nb00; // Stride 0 of src0
  uint32_t nb01; // Stride 1 of src0
  uint32_t nb02; // Stride 2 of src0
  uint32_t nb03; // Stride 3 of src0
  uint32_t ne10; // Dimension 0 of src1
  uint32_t ne11; // Dimension 1 of src1
  uint32_t ne12; // Dimension 2 of src1
  uint32_t ne13; // Dimension 3 of src1
  uint32_t nb10; // Stride 0 of src1
  uint32_t nb11; // Stride 1 of src1
  uint32_t nb12; // Stride 2 of src1
  uint32_t nb13; // Stride 3 of src1
  uint32_t ne0; // Dimension 0 of dst
  uint32_t ne1; // Dimension 1 of dst
  uint32_t ne2; // Dimension 2 of dst
  uint32_t ne3; // Dimension 3 of dst
  uint32_t nb0; // Stride 0 of dst
  uint32_t nb1; // Stride 1 of dst
  uint32_t nb2; // Stride 2 of dst
  uint32_t nb3; // Stride 3 of dst
  uint32_t a_offset; // Offset into src0 buffer
  uint32_t b_offset; // Offset into src1 buffer
  uint32_t d_offset; // Offset into dst buffer
  float param1; // Extra parameter 1
  float param2; // Extra parameter 2
  float param3; // Extra parameter 3
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
  uint32_t ne0;
  uint32_t ne1;
  uint32_t ne2;
  uint32_t ne3;
  uint32_t nb0;
  uint32_t nb1;
  uint32_t nb2;
  uint32_t nb3;
  uint32_t a_offset;
  uint32_t d_offset;
  float param1;
  float param2;
};

// Helper functions for kernel dispatch
void dispatch_binary_op(
    const array& a,
    const array& b,
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s);

void dispatch_unary_op(
    const array& in,
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s,
    float param1 = 0.0f,
    float param2 = 0.0f);

// Get workgroup dimensions for element-wise operations
// Returns (workgroup_count_x, workgroup_count_y, workgroup_count_z)
std::tuple<uint32_t, uint32_t, uint32_t> get_element_wise_grid_dims(
    size_t num_elements,
    uint32_t workgroup_size);

// Standard workgroup size for element-wise operations
constexpr uint32_t VULKAN_WORKGROUP_SIZE = 256;

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
