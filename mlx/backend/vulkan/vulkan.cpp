// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/vulkan.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core::vulkan {

namespace {

struct QueueFamilyIndices {
  uint32_t compute_family{0};
  uint32_t transfer_family{0};
  bool has_separate_transfer{false};
};

uint32_t find_queue_family(
    const std::vector<vk::QueueFamilyProperties>& queue_families,
    const vk::QueueFlags& required,
    const vk::QueueFlags& avoid,
    int32_t compute_index,
    uint32_t min_num_queues) {
  const uint32_t qfsize = queue_families.size();

  for (uint32_t i = 0; i < qfsize; ++i) {
    if (queue_families[i].queueCount >= min_num_queues &&
        (compute_index < 0 || i != static_cast<uint32_t>(compute_index)) &&
        (queue_families[i].queueFlags & required) &&
        !(queue_families[i].queueFlags & avoid)) {
      return i;
    }
  }

  for (uint32_t i = 0; i < qfsize; ++i) {
    if (queue_families[i].queueCount >= min_num_queues &&
        (compute_index < 0 || i != static_cast<uint32_t>(compute_index)) &&
        (queue_families[i].queueFlags & required)) {
      return i;
    }
  }

  for (uint32_t i = 0; i < qfsize; ++i) {
    if (queue_families[i].queueCount >= min_num_queues &&
        (queue_families[i].queueFlags & required)) {
      return i;
    }
  }

  for (uint32_t i = 0; i < qfsize; ++i) {
    if (queue_families[i].queueFlags & required) {
      return i;
    }
  }

  return 0;
}

QueueFamilyIndices find_queue_families(vk::PhysicalDevice physical_device) {
  auto queue_families = physical_device.getQueueFamilyProperties();

  const uint32_t compute_family = find_queue_family(
      queue_families,
      vk::QueueFlagBits::eCompute,
      vk::QueueFlagBits::eGraphics,
      -1,
      1);

  const uint32_t transfer_family = find_queue_family(
      queue_families,
      vk::QueueFlagBits::eTransfer,
      vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics,
      compute_family,
      1);

  QueueFamilyIndices indices;
  indices.compute_family = compute_family;
  indices.transfer_family = transfer_family;
  indices.has_separate_transfer = (compute_family != transfer_family) ||
      (queue_families[compute_family].queueCount == 1 &&
       (queue_families[transfer_family].queueFlags &
        vk::QueueFlagBits::eTransfer) != vk::QueueFlagBits{});

  return indices;
}

bool has_device_extension(
    const std::vector<vk::ExtensionProperties>& extensions,
    const char* name) {
  return std::any_of(
      extensions.begin(),
      extensions.end(),
      [name](const vk::ExtensionProperties& ext) {
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

  // C++ Vulkan API objects (will be automatically destroyed)
  vk::Instance instance;
  vk::PhysicalDevice physical_device;
  vk::Device device;
  vk::Queue compute_queue;
  vk::Queue transfer_queue;
  uint32_t compute_queue_family_index = 0;
  uint32_t transfer_queue_family_index = 0;
  bool has_separate_transfer_queue = false;
  bool is_unified_memory = false;
  bool shader_float16_supported = false;
  bool subgroup_size_control_supported = false;
  bool subgroup_require_full_support = false;
  uint32_t subgroup_min_size = 0;
  uint32_t subgroup_max_size = 0;
  bool pipeline_robustness_supported = false;
  bool coopmat_flash_attention_f32acc_supported = false;
  vk::PhysicalDeviceMemoryProperties mem_properties;
  vk::Semaphore timeline_semaphore;
  uint64_t timeline_value = 0;

  try {
    // 1. Create instance using C++ API
    vk::ApplicationInfo app_info(
        "MLX Vulkan Backend",
        vk::makeVersion(1, 0, 0),
        "MLX",
        vk::makeVersion(1, 0, 0),
        VK_API_VERSION_1_2);

    vk::InstanceCreateInfo create_info({}, &app_info);

    instance = vk::createInstance(create_info);

    // 2. Pick physical device with compute support and find queue families
    auto available_devices = instance.enumeratePhysicalDevices();
    if (available_devices.empty()) {
      throw std::runtime_error(
          "[vulkan::init] Failed to find GPUs with Vulkan support.");
    }

    bool found_compute_device = false;
    for (auto candidate : available_devices) {
      auto queue_families = candidate.getQueueFamilyProperties();
      bool has_compute = false;
      for (const auto& qf : queue_families) {
        if ((qf.queueFlags & vk::QueueFlagBits::eCompute) !=
            vk::QueueFlagBits{}) {
          has_compute = true;
          break;
        }
      }
      if (has_compute) {
        auto indices = find_queue_families(candidate);
        physical_device = candidate;
        compute_queue_family_index = indices.compute_family;
        transfer_queue_family_index = indices.transfer_family;
        has_separate_transfer_queue = indices.has_separate_transfer;
        found_compute_device = true;
        break;
      }
    }

    if (!found_compute_device) {
      throw std::runtime_error(
          "[vulkan::init] Failed to find a compute-capable physical device.");
    }

    // 3. Query memory properties and unified-memory characteristics
    mem_properties = physical_device.getMemoryProperties();

    auto extensions = physical_device.enumerateDeviceExtensionProperties();

    const bool has_subgroup_size_control_ext = has_device_extension(
        extensions, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    const bool has_pipeline_robustness_ext = has_device_extension(
        extensions, VK_EXT_PIPELINE_ROBUSTNESS_EXTENSION_NAME);
    const bool has_cooperative_matrix_ext = has_device_extension(
        extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    auto device_properties = physical_device.getProperties();

    is_unified_memory =
        (device_properties.deviceType ==
         vk::PhysicalDeviceType::eIntegratedGpu);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
      const auto flags = mem_properties.memoryTypes[i].propertyFlags;
      if ((flags & vk::MemoryPropertyFlagBits::eDeviceLocal) &&
          (flags & vk::MemoryPropertyFlagBits::eHostVisible) &&
          (flags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
        is_unified_memory = true;
        break;
      }
    }

    // 4. Create logical device using C++ API
    float queue_priority = 1.0f;
    float queue_priority_low = 0.5f;
    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;

    if (has_separate_transfer_queue) {
      vk::DeviceQueueCreateInfo compute_queue_info(
          vk::DeviceQueueCreateFlags(),
          compute_queue_family_index,
          1,
          &queue_priority);
      vk::DeviceQueueCreateInfo transfer_queue_info(
          vk::DeviceQueueCreateFlags(),
          transfer_queue_family_index,
          1,
          &queue_priority_low);
      queue_create_infos.push_back(compute_queue_info);
      queue_create_infos.push_back(transfer_queue_info);
    } else {
      vk::DeviceQueueCreateInfo queue_info(
          vk::DeviceQueueCreateFlags(),
          compute_queue_family_index,
          1,
          &queue_priority);
      queue_create_infos.push_back(queue_info);
    }

    // Build feature chain
    vk::PhysicalDeviceFeatures2 supported_features;
    vk::PhysicalDeviceVulkan11Features supported_vulkan11_features;
    vk::PhysicalDeviceShaderFloat16Int8Features supported_shader_float16_int8;
    supported_features.pNext = &supported_vulkan11_features;
    supported_vulkan11_features.pNext = &supported_shader_float16_int8;

    vk::PhysicalDeviceSubgroupSizeControlFeatures
        supported_subgroup_size_control{};
    vk::PhysicalDevicePipelineRobustnessFeaturesEXT
        supported_pipeline_robustness{};
    vk::PhysicalDeviceCooperativeMatrixFeaturesKHR
        supported_cooperative_matrix{};

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

    physical_device.getFeatures2(&supported_features);

    // Query subgroup size control properties
    vk::PhysicalDeviceSubgroupSizeControlProperties subgroup_size_control_props;
    vk::PhysicalDeviceProperties2 props2;
    props2.pNext = &subgroup_size_control_props;
    physical_device.getProperties2(&props2);

    // Build enabled features
    vk::PhysicalDeviceFeatures2 enabled_features;
    vk::PhysicalDeviceVulkan11Features enabled_vulkan11_features;
    vk::PhysicalDeviceShaderFloat16Int8Features enabled_shader_float16_int8;
    enabled_features.pNext = &enabled_vulkan11_features;
    enabled_vulkan11_features.pNext = &enabled_shader_float16_int8;

    vk::PhysicalDeviceSubgroupSizeControlFeatures
        enabled_subgroup_size_control{};
    vk::PhysicalDevicePipelineRobustnessFeaturesEXT
        enabled_pipeline_robustness{};
    vk::PhysicalDeviceCooperativeMatrixFeaturesKHR enabled_cooperative_matrix{};

    // Link enabled feature chain (same pattern as supported features)
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
         vk::ShaderStageFlagBits::eCompute)) {
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
      coopmat_flash_attention_f32acc_supported = false;

      // Check for flash attention support
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
    device_extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    if (subgroup_size_control_supported || subgroup_require_full_support) {
      device_extensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    }
    if (pipeline_robustness_supported) {
      device_extensions.push_back(VK_EXT_PIPELINE_ROBUSTNESS_EXTENSION_NAME);
    }
    if (coopmat_flash_attention_f32acc_supported) {
      device_extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }

    vk::DeviceCreateInfo device_create_info;
    device_create_info.flags = vk::DeviceCreateFlags();
    device_create_info.queueCreateInfoCount =
        static_cast<uint32_t>(queue_create_infos.size());
    device_create_info.pQueueCreateInfos = queue_create_infos.data();
    device_create_info.enabledLayerCount = 0;
    device_create_info.ppEnabledLayerNames = nullptr;
    device_create_info.enabledExtensionCount =
        static_cast<uint32_t>(device_extensions.size());
    device_create_info.ppEnabledExtensionNames = device_extensions.data();
    device_create_info.pEnabledFeatures = nullptr;

    // Set the feature chain using pNext
    device_create_info.pNext = &enabled_features;

    device = physical_device.createDevice(device_create_info);

    // 5. Get queues and create timeline semaphore
    compute_queue = device.getQueue(compute_queue_family_index, 0);
    if (has_separate_transfer_queue) {
      transfer_queue = device.getQueue(transfer_queue_family_index, 0);
    } else {
      transfer_queue = compute_queue;
    }

    vk::SemaphoreTypeCreateInfo timeline_ci;
    timeline_ci.sType = vk::StructureType::eSemaphoreTypeCreateInfo;
    timeline_ci.semaphoreType = vk::SemaphoreType::eTimeline;
    timeline_ci.pNext = nullptr;
    vk::SemaphoreCreateInfo ci;
    ci.pNext = &timeline_ci;
    timeline_semaphore = device.createSemaphore(ci);

    // Store in member variables (C++ API objects manage their own cleanup)
    instance_ = instance;
    physical_device_ = physical_device;
    device_ = device;
    compute_queue_ = compute_queue;
    transfer_queue_ = transfer_queue;
    compute_queue_family_index_ = compute_queue_family_index;
    transfer_queue_family_index_ = transfer_queue_family_index;
    has_separate_transfer_queue_ = has_separate_transfer_queue;
    timeline_semaphore_ = timeline_semaphore;
    timeline_value_ = timeline_value;
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
    // Clean up partially initialized resources on failure
    // Note: We don't call waitIdle() here to avoid hangs during failed init
    if (device_) {
      device_.destroy();
      device_ = nullptr;
    }
    if (instance_) {
      instance_.destroy();
      instance_ = nullptr;
    }
    throw;
  }
}

void VulkanContext::cleanup() {
  // Note: vk::Instance and vk::Device are non-owning handles.
  // We must manually destroy them to avoid resource leaks.
  if (device_) {
    // Only wait for idle if we were fully initialized
    // to avoid hanging on partially initialized devices
    if (initialized_) {
      device_.waitIdle();
    }
    if (timeline_semaphore_) {
      device_.destroySemaphore(timeline_semaphore_);
      timeline_semaphore_ = nullptr;
    }
    device_.destroy();
    device_ = nullptr;
  }
  if (instance_) {
    instance_.destroy();
    instance_ = nullptr;
  }
  physical_device_ = nullptr;
  compute_queue_ = nullptr;
  transfer_queue_ = nullptr;
  compute_queue_family_index_ = 0;
  transfer_queue_family_index_ = 0;
  has_separate_transfer_queue_ = false;
  timeline_semaphore_ = nullptr;
  timeline_value_ = 0;
  is_unified_memory_ = false;
  shader_float16_supported_ = false;
  subgroup_size_control_supported_ = false;
  subgroup_require_full_support_ = false;
  subgroup_min_size_ = 0;
  subgroup_max_size_ = 0;
  pipeline_robustness_supported_ = false;
  coopmat_flash_attention_f32acc_supported_ = false;

  // Clear memory properties by creating a default-constructed one
  vk::PhysicalDeviceMemoryProperties empty_props;
  mem_properties_ = empty_props;

  initialized_ = false;
}

uint64_t VulkanContext::increment_timeline() {
  return timeline_value_.fetch_add(1) + 1;
}

uint32_t VulkanContext::find_memory_type(
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties) const {
  // Convert VkMemoryPropertyFlags to vk::MemoryPropertyFlags for comparison
  auto vk_properties = static_cast<vk::MemoryPropertyFlags>(properties);

  for (uint32_t i = 0; i < mem_properties_.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) &&
        (mem_properties_.memoryTypes[i].propertyFlags & vk_properties) ==
            vk_properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type");
}

} // namespace mlx::core::vulkan
