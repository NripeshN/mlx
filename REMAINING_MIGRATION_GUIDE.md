# Remaining Vulkan C API to C++ API Migration Guide

## Migration Approach

This document provides a **complete, line-by-line guide** for migrating the remaining Vulkan backend files from C API to C++ API. Each section shows:
1. Specific code patterns to change
2. The exact transformation needed
3. Notes and caveats

---

## 5. `mlx/backend/vulkan/device.h` 🟡 IN PROGRESS

### Changes Required:

```cpp
// Line 1: CHANGE INCLUDE
// FROM:
#include <vulkan/vulkan.h>
// TO:
#include <vulkan/vulkan.hpp>

// Update all Vk* types to vk::*
// FROM:
VkCommandBuffer begin_command_recording(int stream_index);
VkFence vk_fence_shm;
// TO:
vk::CommandBuffer begin_command_recording(int stream_index);
vk::Fence vk_fence_shm;

// Update VkBuffer to vk::Buffer where used as function parameters
// FROM:
void enqueue_owned_staging_upload(
    const Stream& s,
    const void* src,
    size_t size,
    VkBuffer dst_buffer,
    uint64_t dst_offset = 0);
// TO:
void enqueue_owned_staging_upload(
    const Stream& s,
    const void* src,
    size_t size,
    vk::Buffer dst_buffer,
    uint64_t dst_offset = 0);
```

---

## 6. `mlx/backend/vulkan/device.cpp` 🟡 READY TO MIGRATE

### Key Functions to Update:

### A. `begin_recording()` (lines 879-920)

```cpp
// C API VERSION:
VkCommandBuffer begin_recording(int stream_index) {
  auto* stream = get_stream(stream_index);
  
  if (!stream->recording) {
    retire_submissions(stream, false);
    
    auto resources = acquire_submission_resources(stream);
    VkDevice device = VulkanContext::get().device();  // C API
    
    throw_if_vk_error(
        vkResetCommandPool(device, resources->command_pool, 0),  // C API
        "[vulkan::begin_recording] Failed resetting command pool");
    
    VkCommandBufferBeginInfo beginInfo{};  // C API
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    throw_if_vk_error(
        vkBeginCommandBuffer(resources->command_buffer, &beginInfo),  // C API
        "[vulkan::begin_recording] Failed beginning command buffer");
    // ...
}

// C++ API VERSION:
vk::CommandBuffer begin_recording(int stream_index) {
  auto* stream = get_stream(stream_index);
  
  if (!stream->recording) {
    retire_submissions(stream, false);
    
    auto resources = acquire_submission_resources(stream);
    auto device = VulkanContext::get().device();  // Now returns vk::Device
    
    // vkResetCommandPool is still a C function - keep it
    throw_if_vk_error(
        vkResetCommandPool(static_cast<VkDevice>(device), resources->command_pool, 0),
        "[vulkan::begin_recording] Failed resetting command pool");
    
    vk::CommandBufferBeginInfo beginInfo;  // C++ API
    beginInfo.setPNext(nullptr);  // Set flags properly
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    
    // Use C++ API method
    resources->command_buffer = device.beginCommandBuffer(beginInfo);
    
    throw_if_vk_error(
        VkResult(resources->command_buffer.result()),  // Check VkResult
        "[vulkan::begin_recording] Failed beginning command buffer");
    // ...
    return resources->command_buffer;  // Still returns VkCommandBuffer for now
}
```

### B. `vkCmd*` functions (lines 1115, 1591, 1625)

```cpp
// C API VERSION (line 1115):
vkCmdPipelineBarrier(
    cmd,
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    (VkDependencyFlags)0,
    1,
    &mem_barrier,
    0,
    nullptr,
    0,
    nullptr);

// C++ API VERSION:
cmd.pipelineBarrier(
    vk::PipelineStageFlagBits::eTopOfPipe,
    vk::PipelineStageFlagBits::eComputeShader,
    vk::DependencyFlags(),
    {mem_barrier},  // std::vector<VkMemoryBarrier>
    {},             // std::vector<VkBufferMemoryBarrier>
    {});            // std::vector<VkImageMemoryBarrier>
```

```cpp
// C API VERSION (line 1591):
vkCmdCopyBuffer(
    cmd,
    src_buffer,
    dst_buffer,
    1,
    &region);

// C++ API VERSION:
VkBufferCopy region_vk = {region.srcOffset, region.dstOffset, region.size};  // Convert if needed
cmd.copyBuffer(src_buffer, dst_buffer, {region_vk});  // C++ API takes vector
```

### C. `begin_command_recording()` wrapper (line 1536)

```cpp
// FROM:
VkCommandBuffer begin_command_recording(int stream_index) {
  return VulkanDevice::get().begin_recording(stream_index);
}

// TO:
VkCommandBuffer begin_command_recording(int stream_index) {
  // The return type must stay VkCommandBuffer for now due to function pointers
  // that expect this signature. Keep using the mixed approach.
  return VulkanDevice::get().begin_recording(stream_index);
}
```

---

## 7. `mlx/backend/vulkan/copy.cpp` 🟢 TODO

### Key Functions:

```cpp
// FROM:
void copy_buffer(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst, size_t size) {
  VkBufferCopy region{};
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = size;
  vkCmdCopyBuffer(cmd, src, dst, 1, &region);
}

// TO:
void copy_buffer(VkCommandBuffer cmd, vk::Buffer src, vk::Buffer dst, size_t size) {
  VkBufferCopy region{};
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = size;
  
  // Convert to C API for now, or:
  // vk::BufferCopy region_vk(0, 0, size);
  // cmd.copyBuffer(src, dst, {region_vk});
  cmd.copyBuffer(src, dst, {region});  // Works with VkBufferCopy array
}
```

---

## 8. `mlx/backend/vulkan/matmul.cpp` 🟢 TODO

### Key Patterns:

```cpp
// FROM:
vk::PipelineLayout layout_;
vk::Pipeline pipeline_;

void bind_pipeline(VkCommandBuffer cmd) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
}

// TO:
vk::PipelineLayout layout_;
vk::Pipeline pipeline_;

void bind_pipeline(vk::CommandBuffer cmd) {
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
}
```

```cpp
// FROM:
vkCmdDispatch(cmd, (uint32_t)(M / workgroup_size), (uint32_t)(N / workgroup_size), 1);

// TO:
cmd.dispatch((uint32_t)(M / workgroup_size), (uint32_t)(N / workgroup_size), 1);
```

---

## 9. `mlx/backend/vulkan/compiled.cpp` 🟢 TODO

### Key Patterns:

```cpp
// FROM:
VkShaderModule shader_module;
VkPipelineLayout layout;
VkPipeline pipeline;

vkCreateShaderModule(device, &shader_info, nullptr, &shader_module);
vkCreatePipelineLayout(device, &layout_info, nullptr, &layout);
vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);

vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

// TO:
vk::ShaderModule shader_module = device.createShaderModule(shader_info);
vk::PipelineLayout layout = device.createPipelineLayout(layout_info);
vk::Pipeline pipeline = device.createComputePipeline(nullptr, pipeline_info).value;

cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
```

---

## 10. `mlx/backend/vulkan/fence.cpp` 🟢 TODO

```cpp
// FROM:
VkFence fence = VK_NULL_HANDLE;
vkCreateFence(device, &fence_info, nullptr, &fence);
vkWaitForFences(device, 1, &fence, VK_TRUE, timeout);
VK_CHECK(vkResetFences(device, 1, &fence));
vkDestroyFence(device, fence, nullptr);

// TO:
vk::Fence fence = device.createFence(fence_info);
VK_CHECK(fence.wait(timeout));  // wait() returns vk::Result
fence.reset();
// No vkDestroyFence needed - RAII!
```

---

## Common Migration Patterns

### Pattern 1: Command Buffers

```cpp
// C API:
VkCommandBuffer cmd;
vkBeginCommandBuffer(&cmd, &info);
vkCmdSomething(cmd, ...);
vkEndCommandBuffer(cmd);

// C++ API:
cmd.begin(info);
cmd.something(...);
cmd.end();
```

### Pattern 2: Memory Allocation

```cpp
// C API:
VkDeviceMemory memory;
vkAllocateMemory(device, &alloc_info, nullptr, &memory);
void* ptr;
vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &ptr);

// C++ API:
vk::DeviceMemory memory = device.allocateMemory(alloc_info);
void* ptr = device.mapMemory(memory, 0, vk::WHOLE_SIZE);
```

### Pattern 3: Buffer Creation

```cpp
// C API:
VkBuffer buffer;
VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, ...};
vkCreateBuffer(device, &info, nullptr, &buffer);

// C++ API:
vk::BufferCreateInfo info(flags, size);
info.usage = vk::BufferUsageFlagBits::eStorageBuffer;
vk::Buffer buffer = device.createBuffer(info);
```

### Pattern 4: Descriptor Sets

```cpp
// C API:
VkWriteDescriptorSet write{...};
vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

// C++ API:
vk::WriteDescriptorSet write{...};  // Same struct
write.pNext = nullptr;
device.updateDescriptorSets({write}, {});  // Takes vectors
```

---

## Testing Checklist

### Unit Tests
- [ ] Vulkan initialization
- [ ] Buffer allocation and freeing
- [ ] Memory mapping/unmapping
- [ ] Command buffer recording
- [ ] Buffer copies
- [ ] Compute shaders (matmul, conv, reduce)
- [ ] Synchronization (fences, events)

### Integration Tests
- [ ] End-to-end array operations
- [ ] Stream synchronization
- [ ] Multi-stream operations

### Performance Tests
- [ ] Throughput benchmarks
- [ ] Memory usage comparison

---

## Priority Order

1. **device.cpp** - Core command buffer management (highest impact)
2. **copy.cpp** - Essential for all operations
3. **matmul.cpp, conv.cpp, reduce.cpp** - Core compute operations
4. **compiled.cpp** - Shader compilation
5. **fence.cpp, event.cpp** - Synchronization
6. **Remaining files** - Utilities

---

## Notes

- **Mixed Approach**: Some functions may need to return/accept `VkCommandBuffer` due to function pointers. This is acceptable.
- **Error Handling**: C++ API throws exceptions; check if error codes are needed.
- **RAII Benefits**: Once converted, no need to manually destroy most objects!
- **Testing**: Run Vulkan tests after each major file conversion.
- **Performance**: No expected regression; C++ API has zero overhead when inlined.

---

## Estimated Time Per File

| File | Lines | Estimated Time |
|------|-------|----------------|
| device.cpp | 1686 | 3-4 days |
| copy.cpp | ~800 | 1-2 days |
| matmul.cpp | ~500 | 0.5-1 day |
| conv.cpp | ~400 | 0.5 day |
| reduce.cpp | ~300 | 0.5 day |
| compiled.cpp | ~900 | 1-2 days |
| fence.cpp, event.cpp | ~150 each | 0.5 day |
| **Total** | **~4636** | **9-12 days** |

## Conclusion

With the pattern established in the first 4 files, the remaining migration can proceed systematically. Focus on:

1. **Type conversions**: `Vk*` → `vk::*`
2. **Function calls**: `vk<Function>` → `<object>.<function>()`
3. **Manual cleanup**: Remove `vkDestroy*` calls for C++ objects
4. **RAII**: Let objects destroy themselves automatically

The hardest part (core initialization) is done. Now it's mostly systematic replacements with occasional need to keep C API for edge cases.
