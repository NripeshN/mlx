// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/vulkan.h"

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
    vkGetPhysicalDeviceFeatures2(physical_device, &supported_features);

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

    if (supported_vulkan11_features.storageBuffer16BitAccess) {
      enabled_vulkan11_features.storageBuffer16BitAccess = VK_TRUE;
    }
    if (supported_features.features.shaderInt16) {
      enabled_features.features.shaderInt16 = VK_TRUE;
    }
    if (supported_shader_float16_int8.shaderFloat16) {
      enabled_shader_float16_int8.shaderFloat16 = VK_TRUE;
    }

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = &enabled_features;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pEnabledFeatures = nullptr;

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
