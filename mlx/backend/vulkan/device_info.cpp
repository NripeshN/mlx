// Copyright © 2024 Apple Inc.

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/vulkan/vulkan.h"

namespace mlx::core::vulkan {

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  auto init_device_info = []()
      -> std::unordered_map<std::string, std::variant<std::string, size_t>> {
    const VulkanContext& ctx = VulkanContext::get();
    VkPhysicalDevice physical_device = ctx.physical_device();

    // Get device properties
    VkPhysicalDeviceProperties device_props;
    vkGetPhysicalDeviceProperties(physical_device, &device_props);

    // Get memory properties
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

    // Calculate total device memory (device-local heaps)
    size_t device_memory = 0;
    size_t host_memory = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
      if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        device_memory += mem_props.memoryHeaps[i].size;
      } else {
        host_memory += mem_props.memoryHeaps[i].size;
      }
    }

    // For integrated GPUs with unified memory, device_memory + host_memory
    // should roughly equal total system memory available to GPU

    // Get device name
    std::string device_name(device_props.deviceName);

    // Get device type as string
    std::string device_type;
    switch (device_props.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        device_type = "Integrated GPU";
        break;
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        device_type = "Discrete GPU";
        break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        device_type = "Virtual GPU";
        break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:
        device_type = "CPU";
        break;
      default:
        device_type = "Other";
        break;
    }

    // Get vendor name
    std::string vendor_name;
    switch (device_props.vendorID) {
      case 0x10DE:
        vendor_name = "NVIDIA";
        break;
      case 0x1002:
        vendor_name = "AMD";
        break;
      case 0x8086:
        vendor_name = "Intel";
        break;
      case 0x106B:
        vendor_name = "Apple";
        break;
      case 0x5143:
        vendor_name = "Qualcomm";
        break;
      default:
        vendor_name =
            "Unknown (0x" + std::to_string(device_props.vendorID) + ")";
        break;
    }

    // Get driver version
    uint32_t driver_version = device_props.driverVersion;

    // Get API version
    uint32_t api_version = device_props.apiVersion;

    // Calculate max work group size
    VkPhysicalDeviceLimits limits = device_props.limits;
    size_t max_work_group_size = limits.maxComputeWorkGroupSize[0] *
        limits.maxComputeWorkGroupSize[1] * limits.maxComputeWorkGroupSize[2];

    // Get unified memory status
    bool is_unified = ctx.is_unified_memory();

    // Vulkan doesn't have a direct free memory query like CUDA,
    // so free_memory reports the total (we could track this in allocator later)
    size_t free_memory = device_memory;

    return {
        {"device_name", device_name},
        {"device_type", device_type},
        {"vendor_name", vendor_name},
        {"architecture", "Vulkan"},
        {"driver_version", driver_version},
        {"api_version", api_version},
        {"memory_size", device_memory},
        {"host_memory_size", host_memory},
        {"total_memory_size", device_memory + host_memory},
        {"total_memory", device_memory + host_memory},
        {"free_memory", free_memory},
        {"max_buffer_length", limits.maxStorageBufferRange},
        {"max_recommended_working_set_size", device_memory},
        {"unified_memory", is_unified},
        {"max_work_group_size", max_work_group_size},
        {"max_compute_work_group_count_x", limits.maxComputeWorkGroupCount[0]},
        {"max_compute_work_group_count_y", limits.maxComputeWorkGroupCount[1]},
        {"max_compute_work_group_count_z", limits.maxComputeWorkGroupCount[2]},
        {"max_compute_work_group_size_x", limits.maxComputeWorkGroupSize[0]},
        {"max_compute_work_group_size_y", limits.maxComputeWorkGroupSize[1]},
        {"max_compute_work_group_size_z", limits.maxComputeWorkGroupSize[2]},
        {"max_memory_allocation_count", limits.maxMemoryAllocationCount},
        {"resource_limit", limits.maxMemoryAllocationCount}};
  };

  static auto device_info_ = init_device_info();
  static std::unordered_map<std::string, std::variant<std::string, size_t>>
      empty;

  if (device_index == 0) {
    return device_info_;
  } else {
    return empty;
  }
}

} // namespace mlx::core::vulkan

namespace mlx::core::gpu {

bool is_available() {
  return mlx::core::vulkan::is_available();
}

int device_count() {
  return 1;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  return mlx::core::vulkan::device_info(device_index);
}

} // namespace mlx::core::gpu