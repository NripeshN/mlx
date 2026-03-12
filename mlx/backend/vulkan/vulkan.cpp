// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/vulkan.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core::vulkan {

namespace {

void throw_if_vk_error(VkResult result, const std::string& context) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        context + " (VkResult=" + std::to_string(result) + ").");
  }
}

bool find_compute_queue_family(
    VkPhysicalDevice physical_device,
    uint32_t& queue_family_index) {
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &queue_family_count, nullptr);
  if (queue_family_count == 0) {
    return false;
  }

  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &queue_family_count, queue_families.data());

  for (uint32_t i = 0; i < queue_family_count; ++i) {
    if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
      queue_family_index = i;
      return true;
    }
  }
  return false;
}

bool has_device_extension(
    const std::vector<VkExtensionProperties>& extensions,
    const char* name) {
  return std::any_of(
      extensions.begin(),
      extensions.end(),
      [name](const VkExtensionProperties& ext) {
        return std::string(ext.extensionName) == name;
      });
}

} // namespace

bool is_available() {
  try {
    VulkanContext::get();
    return true;
  } catch (...) {
    return false;
  }
}

bool is_unified_memory() {
  return VulkanContext::get().is_unified_memory();
}

int device_count() {
  return is_available() ? 1 : 0;
}

VulkanContext& VulkanContext::get() {
  static VulkanContext context;
  static std::once_flag init_once;
  auto* context_ptr = &context;
  std::call_once(init_once, [context_ptr]() { context_ptr->init(); });
  return context;
}

VulkanContext::VulkanContext() = default;

VulkanContext::~VulkanContext() {
  cleanup();
}

void VulkanContext::init() {
  if (initialized_) {
    return;
  }

  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue compute_queue = VK_NULL_HANDLE;
  uint32_t compute_queue_family_index = 0;
  bool is_unified_memory = false;
  bool shader_float16_supported = false;
  bool subgroup_size_control_supported = false;
  bool subgroup_require_full_support = false;
  uint32_t subgroup_min_size = 0;
  uint32_t subgroup_max_size = 0;
  bool pipeline_robustness_supported = false;
  bool coopmat_flash_attention_f32acc_supported = false;
  VkPhysicalDeviceMemoryProperties mem_properties{};

  try {
    // 1. Create instance
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "MLX Vulkan Backend";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "MLX";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    throw_if_vk_error(
        vkCreateInstance(&create_info, nullptr, &instance),
        "[vulkan::init] Failed to create Vulkan instance");

    // 2. Pick physical device with compute support
    uint32_t available_device_count = 0;
    throw_if_vk_error(
        vkEnumeratePhysicalDevices(instance, &available_device_count, nullptr),
        "[vulkan::init] Failed to enumerate physical devices");

    if (available_device_count == 0) {
      throw std::runtime_error(
          "[vulkan::init] Failed to find GPUs with Vulkan support.");
    }

    std::vector<VkPhysicalDevice> devices(available_device_count);
    throw_if_vk_error(
        vkEnumeratePhysicalDevices(
            instance, &available_device_count, devices.data()),
        "[vulkan::init] Failed to query physical devices");

    bool found_compute_device = false;
    for (auto candidate : devices) {
      uint32_t queue_family = 0;
      if (find_compute_queue_family(candidate, queue_family)) {
        physical_device = candidate;
        compute_queue_family_index = queue_family;
        found_compute_device = true;
        break;
      }
    }

    if (!found_compute_device) {
      throw std::runtime_error(
          "[vulkan::init] Failed to find a compute-capable physical device.");
    }

    // 3. Query memory properties and unified-memory characteristics
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    uint32_t extension_count = 0;
    throw_if_vk_error(
        vkEnumerateDeviceExtensionProperties(
            physical_device, nullptr, &extension_count, nullptr),
        "[vulkan::init] Failed to enumerate device extensions");
    std::vector<VkExtensionProperties> extensions(extension_count);
    throw_if_vk_error(
        vkEnumerateDeviceExtensionProperties(
            physical_device, nullptr, &extension_count, extensions.data()),
        "[vulkan::init] Failed to query device extensions");

    const bool has_subgroup_size_control_ext = has_device_extension(
        extensions, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    const bool has_pipeline_robustness_ext = has_device_extension(
        extensions, VK_EXT_PIPELINE_ROBUSTNESS_EXTENSION_NAME);
    const bool has_cooperative_matrix_ext = has_device_extension(
        extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(physical_device, &device_properties);

    is_unified_memory =
        (device_properties.deviceType ==
         VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
      const auto flags = mem_properties.memoryTypes[i].propertyFlags;
      if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
          (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
          (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        is_unified_memory = true;
        break;
      }
    }

    // 4. Create logical device
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = compute_queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures2 supported_features{};
    supported_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDeviceVulkan11Features supported_vulkan11_features{};
    supported_vulkan11_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceShaderFloat16Int8Features supported_shader_float16_int8{};
    supported_shader_float16_int8.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    supported_features.pNext = &supported_vulkan11_features;
    supported_vulkan11_features.pNext = &supported_shader_float16_int8;
    VkPhysicalDeviceSubgroupSizeControlFeatures
        supported_subgroup_size_control{};
    supported_subgroup_size_control.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    VkPhysicalDevicePipelineRobustnessFeaturesEXT
        supported_pipeline_robustness{};
    supported_pipeline_robustness.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR supported_cooperative_matrix{};
    supported_cooperative_matrix.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    if (has_subgroup_size_control_ext) {
      supported_shader_float16_int8.pNext = &supported_subgroup_size_control;
      if (has_pipeline_robustness_ext) {
        supported_subgroup_size_control.pNext = &supported_pipeline_robustness;
        if (has_cooperative_matrix_ext) {
          supported_pipeline_robustness.pNext = &supported_cooperative_matrix;
        }
      } else if (has_cooperative_matrix_ext) {
        supported_subgroup_size_control.pNext = &supported_cooperative_matrix;
      }
    } else if (has_pipeline_robustness_ext) {
      supported_shader_float16_int8.pNext = &supported_pipeline_robustness;
      if (has_cooperative_matrix_ext) {
        supported_pipeline_robustness.pNext = &supported_cooperative_matrix;
      }
    } else if (has_cooperative_matrix_ext) {
      supported_shader_float16_int8.pNext = &supported_cooperative_matrix;
    }
    vkGetPhysicalDeviceFeatures2(physical_device, &supported_features);

    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_size_control_props{};
    subgroup_size_control_props.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    if (has_subgroup_size_control_ext) {
      props2.pNext = &subgroup_size_control_props;
    }
    vkGetPhysicalDeviceProperties2(physical_device, &props2);

    VkPhysicalDeviceFeatures2 enabled_features{};
    enabled_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDeviceVulkan11Features enabled_vulkan11_features{};
    enabled_vulkan11_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceShaderFloat16Int8Features enabled_shader_float16_int8{};
    enabled_shader_float16_int8.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    enabled_features.pNext = &enabled_vulkan11_features;
    enabled_vulkan11_features.pNext = &enabled_shader_float16_int8;
    VkPhysicalDeviceSubgroupSizeControlFeatures enabled_subgroup_size_control{};
    enabled_subgroup_size_control.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    VkPhysicalDevicePipelineRobustnessFeaturesEXT enabled_pipeline_robustness{};
    enabled_pipeline_robustness.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR enabled_cooperative_matrix{};
    enabled_cooperative_matrix.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    if (has_subgroup_size_control_ext) {
      enabled_shader_float16_int8.pNext = &enabled_subgroup_size_control;
      if (has_pipeline_robustness_ext) {
        enabled_subgroup_size_control.pNext = &enabled_pipeline_robustness;
        if (has_cooperative_matrix_ext) {
          enabled_pipeline_robustness.pNext = &enabled_cooperative_matrix;
        }
      } else if (has_cooperative_matrix_ext) {
        enabled_subgroup_size_control.pNext = &enabled_cooperative_matrix;
      }
    } else if (has_pipeline_robustness_ext) {
      enabled_shader_float16_int8.pNext = &enabled_pipeline_robustness;
      if (has_cooperative_matrix_ext) {
        enabled_pipeline_robustness.pNext = &enabled_cooperative_matrix;
      }
    } else if (has_cooperative_matrix_ext) {
      enabled_shader_float16_int8.pNext = &enabled_cooperative_matrix;
    }

    if (supported_vulkan11_features.storageBuffer16BitAccess) {
      enabled_vulkan11_features.storageBuffer16BitAccess = VK_TRUE;
    }
    if (supported_features.features.shaderInt16) {
      enabled_features.features.shaderInt16 = VK_TRUE;
    }
    if (supported_shader_float16_int8.shaderFloat16) {
      enabled_shader_float16_int8.shaderFloat16 = VK_TRUE;
      shader_float16_supported = true;
    }
    if (has_subgroup_size_control_ext &&
        supported_subgroup_size_control.subgroupSizeControl &&
        (subgroup_size_control_props.requiredSubgroupSizeStages &
         VK_SHADER_STAGE_COMPUTE_BIT)) {
      enabled_subgroup_size_control.subgroupSizeControl = VK_TRUE;
      subgroup_size_control_supported = true;
      subgroup_min_size = subgroup_size_control_props.minSubgroupSize;
      subgroup_max_size = subgroup_size_control_props.maxSubgroupSize;
    }
    if (has_subgroup_size_control_ext &&
        supported_subgroup_size_control.computeFullSubgroups) {
      enabled_subgroup_size_control.computeFullSubgroups = VK_TRUE;
      subgroup_require_full_support = true;
    }
    if (has_pipeline_robustness_ext &&
        supported_pipeline_robustness.pipelineRobustness) {
      enabled_pipeline_robustness.pipelineRobustness = VK_TRUE;
      pipeline_robustness_supported = true;
    }
    if (has_cooperative_matrix_ext &&
        supported_cooperative_matrix.cooperativeMatrix) {
      enabled_cooperative_matrix.cooperativeMatrix = VK_TRUE;
      auto get_coopmat_props = reinterpret_cast<
          PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
          vkGetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
      if (get_coopmat_props != nullptr) {
        uint32_t coopmat_prop_count = 0;
        if (get_coopmat_props(physical_device, &coopmat_prop_count, nullptr) ==
                VK_SUCCESS &&
            coopmat_prop_count > 0) {
          std::vector<VkCooperativeMatrixPropertiesKHR> coopmat_props(
              coopmat_prop_count);
          for (auto& prop : coopmat_props) {
            prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
          }
          if (get_coopmat_props(
                  physical_device, &coopmat_prop_count, coopmat_props.data()) ==
              VK_SUCCESS) {
            for (const auto& prop : coopmat_props) {
              if (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16 &&
                  prop.scope == VK_SCOPE_SUBGROUP_KHR &&
                  prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                  prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                  prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                  prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                coopmat_flash_attention_f32acc_supported = true;
                break;
              }
            }
          }
        }
      }
    }

    std::vector<const char*> device_extensions;
    if (subgroup_size_control_supported || subgroup_require_full_support) {
      device_extensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    }
    if (pipeline_robustness_supported) {
      device_extensions.push_back(VK_EXT_PIPELINE_ROBUSTNESS_EXTENSION_NAME);
    }
    if (coopmat_flash_attention_f32acc_supported) {
      device_extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = &enabled_features;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pEnabledFeatures = nullptr;
    device_create_info.enabledExtensionCount =
        static_cast<uint32_t>(device_extensions.size());
    device_create_info.ppEnabledExtensionNames = device_extensions.data();

    throw_if_vk_error(
        vkCreateDevice(physical_device, &device_create_info, nullptr, &device),
        "[vulkan::init] Failed to create logical device");

    // 5. Get compute queue
    vkGetDeviceQueue(device, compute_queue_family_index, 0, &compute_queue);

    instance_ = instance;
    physical_device_ = physical_device;
    device_ = device;
    compute_queue_ = compute_queue;
    compute_queue_family_index_ = compute_queue_family_index;
    mem_properties_ = mem_properties;
    is_unified_memory_ = is_unified_memory;
    this->shader_float16_supported_ = shader_float16_supported;
    this->subgroup_size_control_supported_ = subgroup_size_control_supported;
    this->subgroup_require_full_support_ = subgroup_require_full_support;
    this->subgroup_min_size_ = subgroup_min_size;
    this->subgroup_max_size_ = subgroup_max_size;
    this->pipeline_robustness_supported_ = pipeline_robustness_supported;
    this->coopmat_flash_attention_f32acc_supported_ =
        coopmat_flash_attention_f32acc_supported &&
        subgroup_require_full_support;
    initialized_ = true;
  } catch (...) {
    if (device != VK_NULL_HANDLE) {
      vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE) {
      vkDestroyInstance(instance, nullptr);
    }
    throw;
  }
}

void VulkanContext::cleanup() {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
  }
  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
  }
  physical_device_ = VK_NULL_HANDLE;
  compute_queue_ = VK_NULL_HANDLE;
  compute_queue_family_index_ = 0;
  is_unified_memory_ = false;
  this->shader_float16_supported_ = false;
  this->subgroup_size_control_supported_ = false;
  this->subgroup_require_full_support_ = false;
  this->subgroup_min_size_ = 0;
  this->subgroup_max_size_ = 0;
  this->pipeline_robustness_supported_ = false;
  this->coopmat_flash_attention_f32acc_supported_ = false;
  mem_properties_ = {};
  initialized_ = false;
}

uint32_t VulkanContext::find_memory_type(
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties) const {
  for (uint32_t i = 0; i < mem_properties_.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) &&
        (mem_properties_.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type");
}

} // namespace mlx::core::vulkan
