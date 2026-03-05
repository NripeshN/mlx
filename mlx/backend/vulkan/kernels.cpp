// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/kernels.h"
#include <cstring>
#include <sstream>
#include <stdexcept>
#include "mlx/backend/vulkan/vulkan.h"

namespace mlx::core::vulkan {

namespace {

std::string make_pipeline_key(
    const std::string& shader_name,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    uint32_t push_constant_size) {
  std::ostringstream key;
  key << shader_name << "|pc=" << push_constant_size
      << "|n=" << bindings.size();
  for (const auto& binding : bindings) {
    key << "|b=" << binding.binding << ",t=" << binding.descriptorType
        << ",c=" << binding.descriptorCount << ",s=" << binding.stageFlags
        << ",i=" << (binding.pImmutableSamplers != nullptr ? 1 : 0);
  }
  return key.str();
}

} // namespace

ShaderModule::~ShaderModule() {
  if (module != VK_NULL_HANDLE) {
    VkDevice device = VulkanContext::get().device();
    vkDestroyShaderModule(device, module, nullptr);
  }
}

ComputePipeline::~ComputePipeline() {
  VkDevice device = VulkanContext::get().device();
  if (pipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(device, pipeline, nullptr);
  }
  if (layout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device, layout, nullptr);
  }
  if (descriptor_layout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
  }
}

KernelManager::~KernelManager() {
  cleanup();
}

KernelManager& KernelManager::get() {
  static auto* manager = new KernelManager();
  return *manager;
}

void KernelManager::register_shader(
    const std::string& name,
    const void* data,
    size_t size_bytes) {
  auto& shader = shaders_[name];
  if (!shader) {
    shader = std::make_unique<ShaderModule>();
  }
  // Convert bytes to uint32_t words (SPIR-V is word-based)
  size_t num_words = size_bytes / sizeof(uint32_t);
  const uint32_t* words = static_cast<const uint32_t*>(data);
  shader->spirv_code.assign(words, words + num_words);
}

VkShaderModule KernelManager::compile_shader(
    const std::vector<uint32_t>& spirv) {
  VkDevice device = VulkanContext::get().device();

  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = spirv.size() * sizeof(uint32_t);
  createInfo.pCode = spirv.data();

  VkShaderModule module;
  if (vkCreateShaderModule(device, &createInfo, nullptr, &module) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module");
  }

  return module;
}

ShaderModule* KernelManager::get_shader(const std::string& name) {
  auto it = shaders_.find(name);
  if (it == shaders_.end()) {
    return nullptr;
  }

  auto* shader = it->second.get();
  if (!shader->compiled) {
    shader->module = compile_shader(shader->spirv_code);
    shader->compiled = true;
  }

  return shader;
}

ComputePipeline* KernelManager::get_pipeline(
    const std::string& shader_name,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    uint32_t push_constant_size) {
  std::string pipeline_key =
      make_pipeline_key(shader_name, bindings, push_constant_size);

  auto it = pipelines_.find(pipeline_key);
  if (it != pipelines_.end()) {
    return it->second.get();
  }

  VkDevice device = VulkanContext::get().device();

  // Get or compile shader
  ShaderModule* shader = get_shader(shader_name);
  if (!shader) {
    throw std::runtime_error("Shader not found: " + shader_name);
  }

  // Create descriptor set layout
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  if (!bindings.empty()) {
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            device, &layoutInfo, nullptr, &descriptor_layout) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create descriptor set layout");
    }
  }

  // Create pipeline layout
  VkPipelineLayout pipeline_layout;
  VkPushConstantRange push_constant_range{};

  if (push_constant_size > 0) {
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = push_constant_size;
  }

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  if (descriptor_layout != VK_NULL_HANDLE) {
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptor_layout;
  }
  if (push_constant_size > 0) {
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &push_constant_range;
  }

  if (vkCreatePipelineLayout(
          device, &pipelineLayoutInfo, nullptr, &pipeline_layout) !=
      VK_SUCCESS) {
    if (descriptor_layout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    }
    throw std::runtime_error("Failed to create pipeline layout");
  }

  // Create compute pipeline
  VkComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = shader->module;
  pipelineInfo.stage.pName = "main";
  pipelineInfo.layout = pipeline_layout;

  VkPipeline pipeline;
  if (vkCreateComputePipelines(
          device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) !=
      VK_SUCCESS) {
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    if (descriptor_layout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    }
    throw std::runtime_error("Failed to create compute pipeline");
  }

  auto pipeline_ptr = std::make_unique<ComputePipeline>();
  pipeline_ptr->pipeline = pipeline;
  pipeline_ptr->layout = pipeline_layout;
  pipeline_ptr->descriptor_layout = descriptor_layout;
  pipeline_ptr->push_constant_size = push_constant_size;

  auto* result = pipeline_ptr.get();
  pipelines_[pipeline_key] = std::move(pipeline_ptr);

  return result;
}

void KernelManager::init_descriptor_pool() {
  if (descriptor_pool_initialized_) {
    return;
  }

  VkDevice device = VulkanContext::get().device();

  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 1024; // Arbitrary initial size

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = 512;
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

  if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptor_pool_) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor pool");
  }

  descriptor_pool_initialized_ = true;
}

VkDescriptorSet KernelManager::allocate_descriptor_set(
    VkDescriptorSetLayout layout) {
  if (!descriptor_pool_initialized_) {
    init_descriptor_pool();
  }

  VkDevice device = VulkanContext::get().device();

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptor_pool_;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &layout;

  VkDescriptorSet descriptor_set;
  if (vkAllocateDescriptorSets(device, &allocInfo, &descriptor_set) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate descriptor set");
  }

  return descriptor_set;
}

void KernelManager::free_descriptor_set(VkDescriptorSet set) {
  if (descriptor_pool_ != VK_NULL_HANDLE) {
    VkDevice device = VulkanContext::get().device();
    vkFreeDescriptorSets(device, descriptor_pool_, 1, &set);
  }
}

void KernelManager::cleanup() {
  pipelines_.clear();
  shaders_.clear();

  if (descriptor_pool_ != VK_NULL_HANDLE) {
    VkDevice device = VulkanContext::get().device();
    vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
    descriptor_pool_ = VK_NULL_HANDLE;
    descriptor_pool_initialized_ = false;
  }
}

std::tuple<uint32_t, uint32_t, uint32_t> get_element_wise_grid_dims(
    size_t num_elements,
    uint32_t workgroup_size) {
  uint32_t workgroups = static_cast<uint32_t>(
      (num_elements + workgroup_size - 1) / workgroup_size);
  return {workgroups, 1, 1};
}

void dispatch_binary_op(
    const array&,
    const array&,
    array&,
    const std::string&,
    VkCommandBuffer,
    const Stream&) {
  throw std::runtime_error(
      "[vulkan::dispatch_binary_op] Not implemented for Vulkan backend.");
}

void dispatch_unary_op(
    const array&,
    array&,
    const std::string&,
    VkCommandBuffer,
    const Stream&,
    float,
    float) {
  throw std::runtime_error(
      "[vulkan::dispatch_unary_op] Not implemented for Vulkan backend.");
}

} // namespace mlx::core::vulkan
