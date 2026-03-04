// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/vulkan.h"
#include <iostream>
#include <stdexcept>

namespace mlx::core::vulkan {

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
  return 1; // For now, support single device
}

VulkanContext& VulkanContext::get() {
  static VulkanContext context;
  if (!context.initialized_) {
    context.init();
  }
  return context;
}

VulkanContext::VulkanContext() {}

VulkanContext::~VulkanContext() {
  cleanup();
}

void VulkanContext::init() {
  if (initialized_)
    return;

  // 1. Create Instance
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "MLX Vulkan Backend";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "MLX";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create Vulkan instance");
  }

  // 2. Pick Physical Device
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
  if (deviceCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

  // Just pick the first one for now
  physical_device_ = devices[0];

  // 3. Get memory properties and check for unified memory
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties_);

  // Check if device is integrated (unified memory)
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(physical_device_, &deviceProperties);

  // Integrated GPUs typically have unified memory architecture
  is_unified_memory_ =
      (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);

  // Alternatively, check if there's a memory type that is both device local and
  // host visible This is the Vulkan equivalent of unified memory
  for (uint32_t i = 0; i < mem_properties_.memoryTypeCount; i++) {
    VkMemoryPropertyFlags flags = mem_properties_.memoryTypes[i].propertyFlags;
    if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      is_unified_memory_ = true;
      break;
    }
  }

  // 4. Find Compute Queue Family
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device_, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device_, &queueFamilyCount, queueFamilies.data());

  int i = 0;
  bool found = false;
  for (const auto& queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      compute_queue_family_index_ = i;
      found = true;
      break;
    }
    i++;
  }

  if (!found) {
    throw std::runtime_error("failed to find a compute queue family");
  }

  // 5. Create Logical Device
  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = compute_queue_family_index_;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkPhysicalDeviceFeatures deviceFeatures{};

  VkDeviceCreateInfo deviceCreateInfo{};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

  if (vkCreateDevice(physical_device_, &deviceCreateInfo, nullptr, &device_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device");
  }

  // 6. Get Compute Queue
  vkGetDeviceQueue(device_, compute_queue_family_index_, 0, &compute_queue_);

  initialized_ = true;
}

void VulkanContext::cleanup() {
  if (device_ != VK_NULL_HANDLE) {
    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
  }
  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
  }
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