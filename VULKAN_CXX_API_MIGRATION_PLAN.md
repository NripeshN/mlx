# Vulkan C API to C++ API Migration Plan

## Overview

This document outlines the plan to migrate the MLX Vulkan backend from using the C Vulkan API (`vulkan.h`) to the C++ Vulkan API (`vulkan.hpp`). 

## Key Differences: C vs C++ Vulkan API

### C API (Current)
```cpp
#include <vulkan/vulkan.h>

VkInstance instance;
VkDevice device;
VkBuffer buffer;
VkDeviceMemory memory;

vkCreateInstance(&createInfo, nullptr, &instance);
vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);
vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
vkAllocateMemory(device, &allocInfo, nullptr, &memory);
vkCmdCopyBuffer(cmdBuffer, srcBuffer, dstBuffer, 1, &region);
vkDestroyBuffer(device, buffer, nullptr);
vkFreeMemory(device, memory, nullptr);
vkDestroyDevice(device, nullptr);
vkDestroyInstance(instance, nullptr);
```

### C++ API (Target)
```cpp
#include <vulkan/vulkan.hpp>

vk::Instance instance;
vk::Device device;
vk::Buffer buffer;
vk::DeviceMemory memory;

instance = vk::Instance::create(createInfo);
device = physicalDevice.createDevice(createInfo);
buffer = device.createBuffer(bufferInfo);
memory = device.allocateMemory(allocInfo);
cmdBuffer.copyBuffer(srcBuffer, dstBuffer, {region});
// No manual destruction - RAII
```

## Key Benefits of C++ API

1. **RAII (Resource Acquisition Is Initialization)**: Objects automatically destroy themselves when they go out of scope
2. **Method Chaining**: Fluent API for setting up objects
3. **Type Safety**: Compiler-enforced correctness
4. **Exceptions**: Automatic error handling via exceptions
5. **No Manual Cleanup**: Eliminates entire classes of bugs (memory leaks, use-after-free)
6. **Operator Overloads**: More intuitive API (e.g., `buffer == nullptr` vs `buffer == VK_NULL_HANDLE`)

## Current State Analysis

The MLX Vulkan backend currently uses:
- `vulkan.h` (C API) in `mlx/backend/vulkan/`
- Raw `Vk*` handles throughout
- Manual resource management with explicit create/destroy calls
- No RAII patterns

Reference implementation in `llama.cpp/ggml/src/ggml-vulkan/` uses:
- `vulkan.hpp` (C++ API)
- RAII patterns
- Modern C++ features

## Migration Strategy

### Phase 1: Update Includes and Basic Types

**Files to modify:**
- `mlx/backend/vulkan/vulkan.h` → Include `vulkan/vulkan.hpp`
- Update function signatures to use C++ types
- Keep backward compatibility for now

**Changes:**
```cpp
// Before:
#include <vulkan/vulkan.h>
using VkInstance = VkInstance;

// After:
#include <vulkan/vulkan.hpp>
using VkInstance = vk::Instance;
```

### Phase 2: Update VulkanContext Class

**File:** `mlx/backend/vulkan/vulkan.h` / `vulkan.cpp`

Convert from C handles to C++ objects:
```cpp
// Before:
VkInstance instance_;
VkPhysicalDevice physical_device_;
VkDevice device_;
VkQueue compute_queue_;

// After:
vk::Instance instance_;
vk::PhysicalDevice physical_device_;
vk::Device device_;
vk::Queue compute_queue_;
```

Update creation methods:
```cpp
// Before:
throw_if_vk_error(vkCreateInstance(&create_info, nullptr, &instance_), ...);
throw_if_vk_error(vkCreateDevice(physical_device_, &device_create_info, nullptr, &device_), ...);

// After:
instance_ = vk::createInstance(create_info, nullptr);
device_ = physical_device_.createDevice(device_create_info, nullptr);
```

### Phase 3: Update Allocator

**File:** `mlx/backend/vulkan/allocator.h` / `allocator.cpp`

Convert buffer and memory management:
```cpp
// Before:
struct VulkanBuffer {
  void* mapped_ptr;
  VkBuffer buffer;
  VkDeviceMemory memory;
  // ...
};

// After:
struct VulkanBuffer {
  void* mapped_ptr;
  vk::Buffer buffer;
  vk::DeviceMemory memory;
  // ...
};
```

Update allocation code:
```cpp
// Before:
vkCreateBuffer(device, &buffer_info, nullptr, &vk_buffer);
vkAllocateMemory(device, &alloc_info, nullptr, &vk_memory);
vkBindBufferMemory(device, vk_buffer, vk_memory, 0);

// After:
buffer = device.createBuffer(buffer_info);
memory = device.allocateMemory(alloc_info);
buffer.bindMemory(memory, 0);
```

### Phase 4: Update Command Buffer Management

**File:** `mlx/backend/vulkan/device.h` / `device.cpp`

Convert command buffer operations:
```cpp
// Before:
VkCommandBuffer cmd = begin_command_recording(index);
vkCmdCopyBuffer(cmd, src, dst, 1, &region);
end_command_recording(index);

// After:
auto cmd = begin_command_recording(index);
cmd.copyBuffer(src, dst, {region});
end_command_recording(index);
```

### Phase 5: Update Compute Operations

**File:** `mlx/backend/vulkan/copy.cpp`, `matmul.cpp`, `conv.cpp`, etc.

Convert compute shaders and pipeline operations:
```cpp
// Before:
vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
vkCmdDispatch(cmd, x, y, z);

// After:
cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
cmd.dispatch(x, y, z);
```

### Phase 6: Update Synchronization Primitives

**File:** `mlx/backend/vulkan/fence.cpp`, `event.cpp`

Convert fence and event handling:
```cpp
// Before:
VkFence fence = VK_NULL_HANDLE;
vkCreateFence(device, &fence_info, nullptr, &fence);
vkWaitForFences(device, 1, &fence, VK_TRUE, timeout);
vkDestroyFence(device, fence, nullptr);

// After:
vk::Fence fence = device.createFence(fence_info);
vk::Result result = fence.wait(timeout);
// No destroy needed - RAII
```

### Phase 7: Update Shader Module Creation

**File:** `mlx/backend/vulkan/compiled.cpp`

Convert shader module creation:
```cpp
// Before:
VkShaderModule shader_module;
vkCreateShaderModule(device, &shader_info, nullptr, &shader_module);

// After:
vk::ShaderModule shader_module = device.createShaderModule(shader_info);
```

## Implementation Approach

### Step 1: Update Header Includes
- Change `#include <vulkan/vulkan.h>` to `#include <vulkan/vulkan.hpp>`
- Update type aliases
- Keep backward compatibility layer

### Step 2: Convert Object Handles
- Replace `VkInstance` → `vk::Instance`
- Replace `VkDevice` → `vk::Device`
- Replace `VkBuffer` → `vk::Buffer`
- Replace `VkDeviceMemory` → `vk::DeviceMemory`
- Replace `VkCommandBuffer` → `vk::CommandBuffer`
- Replace `VkFence` → `vk::Fence`
- Replace `VkSemaphore` → `vk::Semaphore`
- Replace `VkPipeline` → `vk::Pipeline`
- Replace `VkShaderModule` → `vk::ShaderModule`

### Step 3: Convert Function Calls
Convert C-style function calls to C++ member functions:
- `vkCreate* → .create()`
- `vkDestroy* → automatic (RAII)`
- `vkAllocate* → .allocate()`
- `vkFree* → automatic (RAII)`
- `vkCmd* → cmd.*()`
- `vkWait* → .wait()`
- `vkReset* → .reset()`

### Step 4: Remove Manual Cleanup
Eliminate manual destroy calls and replace with RAII:
```cpp
// Before:
vkDestroyBuffer(device, buffer, nullptr);
vkFreeMemory(device, memory, nullptr);

// After:
// Automatic on scope exit
```

### Step 5: Update Error Handling
The C++ API uses exceptions by default:
```cpp
// Before:
throw_if_vk_error(result, context);

// After:
// C++ API throws vk::Error automatically or returns vk::Result
```

### Step 6: Update Structure Initialization
Use C++ style initialization:
```cpp
// Before:
VkApplicationInfo app_info{}
app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;

VkInstanceCreateInfo create_info{}
create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

// After:
vk::ApplicationInfo app_info{};
vk::InstanceCreateInfo create_info{};
```

## Testing Strategy

1. **Build Tests**: Ensure code compiles with C++ Vulkan API
2. **Unit Tests**: Run existing MLX Vulkan tests
3. **Memory Leak Tests**: Verify no resource leaks with RAII
4. **Functional Tests**: Test all Vulkan operations (copy, matmul, conv, etc.)
5. **Performance Tests**: Ensure no performance regression

## Risk Mitigation

1. **Backward Compatibility**: Keep old C-style API temporarily if needed
2. **Build System**: Update CMakeLists.txt if needed
3. **Debugging**: Provide debug output showing C++ objects
4. **Fallback**: Keep ability to use C API if C++ API not available

## Files to Modify

1. `mlx/backend/vulkan/vulkan.h` / `vulkan.cpp` - Core context
2. `mlx/backend/vulkan/allocator.h` / `allocator.cpp` - Memory management
3. `mlx/backend/vulkan/device.h` / `device.cpp` - Device and commands
4. `mlx/backend/vulkan/copy.cpp` - Buffer copies
5. `mlx/backend/vulkan/matmul.cpp` - Matrix multiplication
6. `mlx/backend/vulkan/conv.cpp` - Convolution
7. `mlx/backend/vulkan/reduce.cpp` - Reduction operations
8. `mlx/backend/vulkan/compiled.cpp` - Shader compilation
9. `mlx/backend/vulkan/event.cpp` / `fence.cpp` - Synchronization
10. `mlx/backend/vulkan/primitives.cpp` - Primitives

## Estimated Effort

- **Analysis**: 1 day (completed)
- **Phase 1-2 (Core)**: 2-3 days
- **Phase 3-5 (Operations)**: 5-7 days
- **Phase 6-7 (Sync/Shaders)**: 2-3 days
- **Testing**: 2-3 days
- **Total**: 12-17 days

## Success Criteria

1. All Vulkan tests pass
2. No memory leaks detected
3. No performance regression (>5%)
4. Code is cleaner and more maintainable
5. Builds successfully on all platforms
