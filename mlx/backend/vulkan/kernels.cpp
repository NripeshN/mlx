// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/kernels.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <span>
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

VkDescriptorSetLayoutBinding make_storage_buffer_binding(uint32_t binding) {
  VkDescriptorSetLayoutBinding layout_binding{};
  layout_binding.binding = binding;
  layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  layout_binding.descriptorCount = 1;
  layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  layout_binding.pImmutableSamplers = nullptr;
  return layout_binding;
}

uint32_t checked_u32(int64_t value, const char* name) {
  if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        std::string("[vulkan::kernels] ") + name + " is out of uint32 range.");
  }
  return static_cast<uint32_t>(value);
}

uint32_t checked_mul_u32(uint32_t a, uint32_t b, const char* name) {
  const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
  if (product > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        std::string("[vulkan::kernels] ") + name + " is out of uint32 range.");
  }
  return static_cast<uint32_t>(product);
}

struct TensorLayout4D {
  uint32_t ne00{1};
  uint32_t ne01{1};
  uint32_t ne02{1};
  uint32_t ne03{1};
  uint32_t nb00{0};
  uint32_t nb01{0};
  uint32_t nb02{0};
  uint32_t nb03{0};
};

TensorLayout4D make_tensor_layout_4d(
    const array& arr,
    const char* tensor_name) {
  if (arr.ndim() > 4) {
    throw std::runtime_error(
        std::string("[vulkan::kernels] ") + tensor_name +
        " rank > 4 is not supported.");
  }

  TensorLayout4D layout;
  for (size_t i = 0; i < arr.ndim(); ++i) {
    const int src_dim = static_cast<int>(arr.ndim() - 1 - i);
    const auto dim_size = checked_u32(arr.shape(src_dim), "shape");
    const auto stride = checked_u32(arr.strides(src_dim), "stride");

    switch (i) {
      case 0:
        layout.ne00 = dim_size;
        layout.nb00 = stride;
        break;
      case 1:
        layout.ne01 = dim_size;
        layout.nb01 = stride;
        break;
      case 2:
        layout.ne02 = dim_size;
        layout.nb02 = stride;
        break;
      case 3:
        layout.ne03 = dim_size;
        layout.nb03 = stride;
        break;
    }
  }

  return layout;
}

uint32_t
checked_offset(const array& arr, const char* tensor_name, uint32_t max_offset) {
  if (arr.offset() < 0 || arr.offset() > max_offset) {
    throw std::runtime_error(
        std::string("[vulkan::kernels] ") + tensor_name +
        " offset is out of supported range.");
  }
  return static_cast<uint32_t>(arr.offset());
}

void init_fastdiv_values(uint32_t d, uint32_t& mp, uint32_t& L) {
  if (d == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] fastdiv divisor must be non-zero.");
  }

  L = 0;
  while (L < 32 && (uint32_t{1} << L) < d) {
    L++;
  }
  mp = static_cast<uint32_t>(
      ((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d) + 1);
}

void init_unary_fastdiv(UnaryPushConstants& p) {
  init_fastdiv_values(p.ne02 * p.ne01 * p.ne00, p.ne0_012mp, p.ne0_012L);
  init_fastdiv_values(p.ne01 * p.ne00, p.ne0_01mp, p.ne0_01L);
  init_fastdiv_values(p.ne00, p.ne0_0mp, p.ne0_0L);
  init_fastdiv_values(p.ne12 * p.ne11 * p.ne10, p.ne1_012mp, p.ne1_012L);
  init_fastdiv_values(p.ne11 * p.ne10, p.ne1_01mp, p.ne1_01L);
  init_fastdiv_values(p.ne10, p.ne1_0mp, p.ne1_0L);
}

VkDescriptorBufferInfo make_buffer_info(const array& arr, const char* name) {
  auto* vulkan_buffer = static_cast<const VulkanBuffer*>(
      static_cast<const void*>(arr.buffer().ptr()));
  if (vulkan_buffer == nullptr || vulkan_buffer->buffer == VK_NULL_HANDLE) {
    throw std::runtime_error(
        std::string("[vulkan::kernels] Missing Vulkan buffer for ") + name +
        ".");
  }

  VkDescriptorBufferInfo info{};
  info.buffer = vulkan_buffer->buffer;
  info.offset = 0;
  info.range = VK_WHOLE_SIZE;
  return info;
}

enum class DispatchGridKind {
  ElementWise,
  Linear1D,
  RowWise,
};

enum class KernelSpecId {
  Binary,
  BinaryAddWithPartials,
  Unary,
  GenericUnary,
  Arange,
  SumRows,
  Argmax,
  Softmax,
};

struct KernelSpec {
  std::array<uint32_t, 4> bindings{};
  uint32_t binding_count{0};
  uint32_t push_constant_size{0};
  DispatchGridKind grid_kind{DispatchGridKind::ElementWise};
};

struct BoundArray {
  const array* arr;
  const char* name;
};

constexpr KernelSpec make_kernel_spec(
    std::array<uint32_t, 4> bindings,
    uint32_t binding_count,
    uint32_t push_constant_size,
    DispatchGridKind grid_kind) {
  return {bindings, binding_count, push_constant_size, grid_kind};
}

constexpr std::array<KernelSpec, 8> kKernelSpecs = {
    make_kernel_spec(
        {0, 1, 2, 0},
        3,
        sizeof(BinaryPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1, 2, 3},
        4,
        sizeof(BinaryPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1, 0, 0},
        2,
        sizeof(UnaryPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1, 0, 0},
        2,
        sizeof(GenericPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 0, 0, 0},
        1,
        sizeof(GenericPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 0, 0},
        2,
        sizeof(SumRowsPushConstants),
        DispatchGridKind::RowWise),
    make_kernel_spec(
        {0, 1, 0, 0},
        2,
        sizeof(GenericPushConstants),
        DispatchGridKind::RowWise),
    make_kernel_spec(
        {0, 1, 2, 3},
        4,
        sizeof(SoftmaxPushConstants),
        DispatchGridKind::RowWise),
};

size_t kernel_spec_index(KernelSpecId id) {
  return static_cast<size_t>(id);
}

const KernelSpec& get_kernel_spec(KernelSpecId id) {
  return kKernelSpecs[kernel_spec_index(id)];
}

KernelSpecId kernel_spec_id_for_binary_variant(BinaryDispatchVariant variant) {
  switch (variant) {
    case BinaryDispatchVariant::Standard:
      return KernelSpecId::Binary;
    case BinaryDispatchVariant::AddWithPartials:
      return KernelSpecId::BinaryAddWithPartials;
  }

  throw std::runtime_error(
      "[vulkan::kernels] Unsupported binary dispatch variant.");
}

std::vector<VkDescriptorSetLayoutBinding> make_layout_bindings(
    const KernelSpec& spec) {
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  bindings.reserve(spec.binding_count);
  for (size_t i = 0; i < spec.binding_count; ++i) {
    bindings.push_back(make_storage_buffer_binding(spec.bindings[i]));
  }
  return bindings;
}

std::tuple<uint32_t, uint32_t, uint32_t> get_dispatch_grid_dims(
    DispatchGridKind grid_kind,
    uint32_t num_elements) {
  switch (grid_kind) {
    case DispatchGridKind::ElementWise:
      return get_element_wise_grid_dims(num_elements, VULKAN_INDEX_TILE_SIZE);
    case DispatchGridKind::Linear1D:
      return {
          (num_elements + VULKAN_INDEX_TILE_SIZE - 1) / VULKAN_INDEX_TILE_SIZE,
          1,
          1};
    case DispatchGridKind::RowWise:
      return get_element_wise_grid_dims(num_elements, 1);
  }

  throw std::runtime_error("[vulkan::kernels] Unsupported dispatch grid kind.");
}

BinaryPushConstants make_binary_push_constants(
    const array& a,
    const array& b,
    const array& out,
    float param1 = 0.0f,
    float param2 = 0.0f,
    int32_t param3 = 0) {
  const auto a_layout = make_tensor_layout_4d(a, "src0");
  const auto b_layout = make_tensor_layout_4d(b, "src1");
  const auto d_layout = make_tensor_layout_4d(out, "dst");

  BinaryPushConstants push_constants{};
  push_constants.ne = checked_u32(out.size(), "binary element count");
  push_constants.ne00 = a_layout.ne00;
  push_constants.ne01 = a_layout.ne01;
  push_constants.ne02 = a_layout.ne02;
  push_constants.ne03 = a_layout.ne03;
  push_constants.nb00 = a_layout.nb00;
  push_constants.nb01 = a_layout.nb01;
  push_constants.nb02 = a_layout.nb02;
  push_constants.nb03 = a_layout.nb03;
  push_constants.ne10 = b_layout.ne00;
  push_constants.ne11 = b_layout.ne01;
  push_constants.ne12 = b_layout.ne02;
  push_constants.ne13 = b_layout.ne03;
  push_constants.nb10 = b_layout.nb00;
  push_constants.nb11 = b_layout.nb01;
  push_constants.nb12 = b_layout.nb02;
  push_constants.nb13 = b_layout.nb03;
  push_constants.ne20 = d_layout.ne00;
  push_constants.ne21 = d_layout.ne01;
  push_constants.ne22 = d_layout.ne02;
  push_constants.ne23 = d_layout.ne03;
  push_constants.nb20 = d_layout.nb00;
  push_constants.nb21 = d_layout.nb01;
  push_constants.nb22 = d_layout.nb02;
  push_constants.nb23 = d_layout.nb03;
  const uint32_t a_offset = checked_offset(a, "src0", 0xFFFFu);
  const uint32_t b_offset = checked_offset(b, "src1", 0xFFu);
  const uint32_t d_offset = checked_offset(out, "dst", 0xFFu);
  push_constants.misalign_offsets =
      (a_offset << 16) | (b_offset << 8) | d_offset;
  push_constants.param1 = param1;
  push_constants.param2 = param2;
  push_constants.param3 = param3;

  return push_constants;
}

UnaryPushConstants make_unary_push_constants(
    const array& in,
    const array& out,
    float param1,
    float param2) {
  const auto in_layout = make_tensor_layout_4d(in, "src0");
  const auto out_layout = make_tensor_layout_4d(out, "dst");

  UnaryPushConstants push_constants{};
  push_constants.ne = checked_u32(out.size(), "unary element count");
  push_constants.ne00 = in_layout.ne00;
  push_constants.ne01 = in_layout.ne01;
  push_constants.ne02 = in_layout.ne02;
  push_constants.ne03 = in_layout.ne03;
  push_constants.nb00 = in_layout.nb00;
  push_constants.nb01 = in_layout.nb01;
  push_constants.nb02 = in_layout.nb02;
  push_constants.nb03 = in_layout.nb03;
  push_constants.ne10 = out_layout.ne00;
  push_constants.ne11 = out_layout.ne01;
  push_constants.ne12 = out_layout.ne02;
  push_constants.ne13 = out_layout.ne03;
  push_constants.nb10 = out_layout.nb00;
  push_constants.nb11 = out_layout.nb01;
  push_constants.nb12 = out_layout.nb02;
  push_constants.nb13 = out_layout.nb03;
  const uint32_t a_offset = checked_offset(in, "src0", 0xFFFFu);
  const uint32_t d_offset = checked_offset(out, "dst", 0xFFFFu);
  push_constants.misalign_offsets = (a_offset << 16) | d_offset;
  push_constants.param1 = param1;
  push_constants.param2 = param2;
  init_unary_fastdiv(push_constants);

  return push_constants;
}

GenericPushConstants make_generic_push_constants(
    uint32_t kx,
    float param1,
    float param2,
    float param3,
    float param4) {
  GenericPushConstants push_constants{};
  push_constants.KX = kx;
  push_constants.KY = 1;
  push_constants.param1 = param1;
  push_constants.param2 = param2;
  push_constants.param3 = param3;
  push_constants.param4 = param4;
  return push_constants;
}

SumRowsPushConstants
make_sum_rows_push_constants(const array& in, const array& out, float weight) {
  const auto in_layout = make_tensor_layout_4d(in, "src0");
  const auto out_layout = make_tensor_layout_4d(out, "dst");

  SumRowsPushConstants push_constants{};
  push_constants.n_cols = in_layout.ne00;
  push_constants.ne01 = in_layout.ne01;
  push_constants.ne02 = in_layout.ne02;
  push_constants.nb01 = in_layout.nb01;
  push_constants.nb02 = in_layout.nb02;
  push_constants.nb03 = in_layout.nb03;
  push_constants.nb11 = out_layout.nb01;
  push_constants.nb12 = out_layout.nb02;
  push_constants.nb13 = out_layout.nb03;
  push_constants.weight = weight;

  const uint32_t a_offset = checked_offset(in, "src0", 0xFFFFu);
  const uint32_t d_offset = checked_offset(out, "dst", 0xFFFFu);
  push_constants.misalign_offsets = (a_offset << 16) | d_offset;

  const uint32_t ne0_12 = checked_mul_u32(
      push_constants.ne01, push_constants.ne02, "sum_rows ne01*ne02");
  init_fastdiv_values(ne0_12, push_constants.ne0_12mp, push_constants.ne0_12L);
  init_fastdiv_values(
      push_constants.ne01, push_constants.ne0_1mp, push_constants.ne0_1L);

  return push_constants;
}

SoftmaxPushConstants make_softmax_push_constants(
    const array& in,
    uint32_t row_width,
    uint32_t row_count) {
  SoftmaxPushConstants push_constants{};
  push_constants.KX = row_width;
  push_constants.KY = 0;
  push_constants.ne00 = row_width;
  push_constants.ne01 = 1;
  push_constants.ne02 = 1;
  push_constants.ne12 = 1;
  push_constants.ne13 = 1;
  push_constants.nb11 = 0;
  push_constants.nb12 = 0;
  push_constants.nb13 = 0;
  push_constants.scale = 1.0f;
  push_constants.max_bias = 0.0f;
  push_constants.m0 = 0.0f;
  push_constants.m1 = 0.0f;
  push_constants.n_head_log2 = 0;
  push_constants.nrows_x = row_count;
  push_constants.has_sinks = 0;

  if (in.ndim() > 4) {
    throw std::runtime_error(
        "[vulkan::kernels] Softmax supports rank <= 4 for Vulkan kernels.");
  }

  return push_constants;
}

template <typename PushConstants>
void dispatch_with_spec(
    const std::string& shader_name,
    KernelSpecId spec_id,
    std::span<const BoundArray> bound_arrays,
    const PushConstants& push_constants,
    uint32_t num_elements,
    VkCommandBuffer cmd_buffer,
    const Stream& s) {
  if (num_elements == 0) {
    return;
  }

  const auto& spec = get_kernel_spec(spec_id);
  if (bound_arrays.size() != spec.binding_count) {
    throw std::runtime_error(
        "[vulkan::kernels] Kernel bindings do not match registered "
        "KernelSpec.");
  }
  if (sizeof(PushConstants) != spec.push_constant_size) {
    throw std::runtime_error(
        "[vulkan::kernels] Push constant size does not match registered "
        "KernelSpec.");
  }

  auto bindings = make_layout_bindings(spec);

  auto& manager = KernelManager::get();
  auto* pipeline = manager.get_pipeline(
      shader_name, bindings, static_cast<uint32_t>(sizeof(PushConstants)));
  VkDescriptorSet descriptor_set =
      manager.allocate_descriptor_set(pipeline->descriptor_layout);
  manager.defer_descriptor_set_free(s.index, descriptor_set);

  std::array<VkDescriptorBufferInfo, 4> descriptor_infos{};
  std::array<VkWriteDescriptorSet, 4> descriptor_writes{};

  for (size_t i = 0; i < bound_arrays.size(); ++i) {
    if (bound_arrays[i].arr == nullptr) {
      throw std::runtime_error("[vulkan::kernels] Missing bound array.");
    }

    descriptor_infos[i] =
        make_buffer_info(*bound_arrays[i].arr, bound_arrays[i].name);
    auto& write = descriptor_writes[i];
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set;
    write.dstBinding = spec.bindings[i];
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &descriptor_infos[i];
  }

  vkUpdateDescriptorSets(
      VulkanContext::get().device(),
      static_cast<uint32_t>(bound_arrays.size()),
      descriptor_writes.data(),
      0,
      nullptr);

  vkCmdBindPipeline(
      cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  vkCmdBindDescriptorSets(
      cmd_buffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline->layout,
      0,
      1,
      &descriptor_set,
      0,
      nullptr);
  vkCmdPushConstants(
      cmd_buffer,
      pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      static_cast<uint32_t>(sizeof(PushConstants)),
      &push_constants);

  auto [grid_x, grid_y, grid_z] =
      get_dispatch_grid_dims(spec.grid_kind, num_elements);
  vkCmdDispatch(cmd_buffer, grid_x, grid_y, grid_z);
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

void KernelManager::defer_descriptor_set_free(
    int stream_index,
    VkDescriptorSet set) {
  if (set == VK_NULL_HANDLE) {
    return;
  }
  std::lock_guard<std::mutex> lock(deferred_descriptor_sets_mutex_);
  deferred_descriptor_sets_[stream_index].push_back(set);
}

void KernelManager::reclaim_descriptor_sets(int stream_index) {
  std::vector<VkDescriptorSet> sets;
  {
    std::lock_guard<std::mutex> lock(deferred_descriptor_sets_mutex_);
    auto it = deferred_descriptor_sets_.find(stream_index);
    if (it == deferred_descriptor_sets_.end()) {
      return;
    }
    sets = std::move(it->second);
    deferred_descriptor_sets_.erase(it);
  }

  if (sets.empty() || descriptor_pool_ == VK_NULL_HANDLE) {
    return;
  }

  VkDevice device = VulkanContext::get().device();
  vkFreeDescriptorSets(
      device,
      descriptor_pool_,
      static_cast<uint32_t>(sets.size()),
      sets.data());
}

void KernelManager::reclaim_all_descriptor_sets() {
  std::unordered_map<int, std::vector<VkDescriptorSet>> all_sets;
  {
    std::lock_guard<std::mutex> lock(deferred_descriptor_sets_mutex_);
    all_sets = std::move(deferred_descriptor_sets_);
    deferred_descriptor_sets_.clear();
  }

  if (descriptor_pool_ == VK_NULL_HANDLE) {
    return;
  }

  VkDevice device = VulkanContext::get().device();
  for (auto& [_, sets] : all_sets) {
    if (sets.empty()) {
      continue;
    }
    vkFreeDescriptorSets(
        device,
        descriptor_pool_,
        static_cast<uint32_t>(sets.size()),
        sets.data());
  }
}

void KernelManager::cleanup() {
  reclaim_all_descriptor_sets();
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
    uint32_t tile_size) {
  if (num_elements == 0) {
    return {0, 0, 0};
  }

  const uint64_t tiles =
      (static_cast<uint64_t>(num_elements) + tile_size - 1) / tile_size;
  if (tiles >
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) * 512ULL) {
    throw std::runtime_error(
        "[vulkan::kernels] Elementwise dispatch exceeds Vulkan limits.");
  }

  const uint32_t x = 1;
  const uint32_t y = static_cast<uint32_t>(std::min<uint64_t>(tiles, 512));
  const uint32_t z = static_cast<uint32_t>((tiles + 511) / 512);
  return {x, y, z};
}

void dispatch_binary_op(
    const array& a,
    const array& b,
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s,
    BinaryDispatchVariant variant) {
  const auto push_constants = make_binary_push_constants(a, b, out);
  const auto spec_id = kernel_spec_id_for_binary_variant(variant);

  if (variant == BinaryDispatchVariant::AddWithPartials) {
    const std::array<BoundArray, 4> bound_arrays = {{
        {&a, "src0"},
        {&b, "src1"},
        {&out, "dst"},
        {&out, "partial"},
    }};
    dispatch_with_spec(
        shader_name,
        spec_id,
        bound_arrays,
        push_constants,
        push_constants.ne,
        cmd_buffer,
        s);
    return;
  }

  const std::array<BoundArray, 3> bound_arrays = {{
      {&a, "src0"},
      {&b, "src1"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_name,
      spec_id,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s);
}

void dispatch_unary_op(
    const array& in,
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s,
    float param1,
    float param2) {
  const auto push_constants =
      make_unary_push_constants(in, out, param1, param2);
  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_name,
      KernelSpecId::Unary,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s);
}

void dispatch_generic_unary_op(
    const array& in,
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s,
    float param1,
    float param2,
    float param3,
    float param4) {
  const auto element_count =
      checked_u32(out.size(), "generic unary element count");
  const auto push_constants = make_generic_push_constants(
      element_count, param1, param2, param3, param4);

  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_name,
      KernelSpecId::GenericUnary,
      bound_arrays,
      push_constants,
      push_constants.KX,
      cmd_buffer,
      s);
}

void dispatch_arange_op(
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s,
    float start,
    float step) {
  const auto num_elements = checked_u32(out.size(), "arange element count");
  const auto push_constants =
      make_generic_push_constants(num_elements, start, step, 0.0f, 0.0f);
  const std::array<BoundArray, 1> bound_arrays = {{{&out, "dst"}}};

  dispatch_with_spec(
      shader_name,
      KernelSpecId::Arange,
      bound_arrays,
      push_constants,
      push_constants.KX,
      cmd_buffer,
      s);
}

void dispatch_sum_rows_op(
    const array& in,
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s,
    float weight) {
  if (out.size() == 0) {
    return;
  }

  const auto push_constants = make_sum_rows_push_constants(in, out, weight);
  const auto row_count = checked_u32(out.size(), "sum_rows output rows");

  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_name,
      KernelSpecId::SumRows,
      bound_arrays,
      push_constants,
      row_count,
      cmd_buffer,
      s);
}

void dispatch_argmax_op(
    const array& in,
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s) {
  if (out.size() == 0) {
    return;
  }

  if (in.ndim() == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] ArgMax requires input rank >= 1.");
  }

  const uint32_t row_width =
      checked_u32(in.shape(in.ndim() - 1), "argmax reduction width");
  const uint32_t row_count = checked_u32(out.size(), "argmax row count");
  auto push_constants =
      make_generic_push_constants(row_width, 0.0f, 0.0f, 0.0f, 0.0f);
  push_constants.KY = row_count;

  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_name,
      KernelSpecId::Argmax,
      bound_arrays,
      push_constants,
      push_constants.KY,
      cmd_buffer,
      s);
}

void dispatch_softmax_op(
    const array& in,
    array& out,
    const std::string& shader_name,
    VkCommandBuffer cmd_buffer,
    const Stream& s) {
  if (out.size() == 0) {
    return;
  }

  if (in.ndim() == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Softmax requires input rank >= 1.");
  }

  const uint32_t row_width = checked_u32(in.shape(in.ndim() - 1), "softmax KX");
  if (row_width == 0) {
    throw std::runtime_error("[vulkan::kernels] Softmax requires non-zero KX.");
  }

  const uint32_t total_elements = checked_u32(out.size(), "softmax elements");
  if (total_elements % row_width != 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Softmax elements are not divisible by KX.");
  }
  const uint32_t row_count = total_elements / row_width;
  const auto push_constants =
      make_softmax_push_constants(in, row_width, row_count);

  const std::array<BoundArray, 4> bound_arrays = {{
      {&in, "src0"},
      {&in, "src1"},
      {&in, "src2"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_name,
      KernelSpecId::Softmax,
      bound_arrays,
      push_constants,
      push_constants.nrows_x,
      cmd_buffer,
      s);
}

} // namespace mlx::core::vulkan
