// Copyright © 2024 Apple Inc.

#include <fmt/format.h>
#include <shaderc/shaderc.hpp>
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

// Convert MLX dtype to GLSL type string
std::string dtype_to_glsl(Dtype d) {
  switch (d) {
    case float32:
      return "float";
    case float16:
      return "float16_t";
    case int32:
      return "int";
    case uint32:
      return "uint";
    case int64:
      return "int64_t";
    case uint64:
      return "uint64_t";
    case int16:
      return "int16_t";
    case uint16:
      return "uint16_t";
    case int8:
      return "int8_t";
    case uint8:
      return "uint8_t";
    case bool_:
      return "bool";
    default:
      throw std::runtime_error(
          fmt::format(
              "Unsupported dtype for Vulkan compiled: {}", dtype_to_string(d)));
  }
}

// Map primitive names to GLSL operators/functions
std::string get_glsl_operator(const std::string& primitive_name) {
  static const std::unordered_map<std::string, std::string> op_map = {
      {"Add", "+"},
      {"Subtract", "-"},
      {"Multiply", "*"},
      {"Divide", "/"},
      {"Maximum", "max"},
      {"Minimum", "min"},
      {"Equal", "=="},
      {"NotEqual", "!="},
      {"Greater", ">"},
      {"Less", "<"},
      {"GreaterEqual", ">="},
      {"LessEqual", "<="},
      {"LogicalAnd", "&&"},
      {"LogicalOr", "||"},
      {"BitwiseAnd", "&"},
      {"BitwiseOr", "|"},
      {"BitwiseXor", "^"},
      // GLSL built-in functions (lowercase)
      {"Exp", "exp"},
      {"Log", "log"},
      {"Sin", "sin"},
      {"Cos", "cos"},
      {"Tan", "tan"},
      {"Sqrt", "sqrt"},
      {"Abs", "abs"},
      {"Floor", "floor"},
      {"Ceil", "ceil"},
      {"Round", "round"},
      {"Sigmoid", "sigmoid"},
  };

  auto it = op_map.find(primitive_name);
  if (it != op_map.end()) {
    return it->second;
  }
  return primitive_name; // Function call style
}

// Build GLSL kernel source for the compiled tape
inline void build_glsl_kernel(
    std::string& os,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    const std::unordered_set<uintptr_t>& constant_ids,
    bool contiguous,
    int ndim,
    int work_per_thread) {
  // Maps to store array identifiers - use simple valid GLSL names
  std::unordered_map<std::uintptr_t, std::string> var_names;
  int var_counter = 0;

  // Helper to get or create identifier
  auto get_var_name = [&](const array& x) -> const std::string& {
    auto key = x.id();
    auto it = var_names.find(key);
    if (it != var_names.end()) {
      return it->second;
    }
    std::string id = fmt::format("v{}", var_counter++);
    auto [insert_it, _] = var_names.emplace(key, id);
    return insert_it->second;
  };

  // Pre-populate variable names for all arrays we'll reference
  for (const auto& x : inputs) {
    get_var_name(x);
  }
  for (const auto& x : outputs) {
    get_var_name(x);
  }
  for (const auto& x : tape) {
    get_var_name(x);
  }

  // GLSL header
  os = "#version 450\n";
  os += "#extension GL_EXT_shader_16bit_storage : require\n";
  os +=
      "\nlayout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";

  // Determine max work per thread based on output dtype size
  int max_itemsize = 1;
  for (const auto& x : outputs) {
    max_itemsize = std::max(max_itemsize, static_cast<int>(x.itemsize()));
  }
  int wpt = std::min(work_per_thread, 16 / max_itemsize);
  wpt = std::max(wpt, 1);

  // Buffer bindings for non-constant inputs
  int binding = 0;
  std::vector<std::pair<int, std::string>> input_bindings; // (index, name)

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant(i)) {
      continue;
    }
    const auto& x = inputs[i];
    const auto& xname = get_var_name(x);

    if (is_scalar(x)) {
      os += fmt::format(
          "layout(binding = {}) readonly buffer Buf{} {{ {} {}; }};\n",
          binding++,
          i,
          dtype_to_glsl(x.dtype()),
          xname);
    } else {
      os += fmt::format(
          "layout(binding = {}) readonly buffer Buf{} {{ {} {}[]; }};\n",
          binding++,
          i,
          dtype_to_glsl(x.dtype()),
          xname);
    }
    input_bindings.push_back({static_cast<int>(i), xname});
  }

  // Strides buffer for non-contiguous access
  int strides_binding = -1;
  if (!contiguous && !input_bindings.empty()) {
    strides_binding = binding++;
    os += fmt::format(
        "layout(binding = {}) readonly buffer StridesBuf {{ int in_strides[]; }};\n",
        strides_binding);
  }

  // Output buffers
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& x = outputs[i];
    const auto& xname = get_var_name(x);
    os += fmt::format(
        "layout(binding = {}) buffer OutBuf{} {{ {} {}[]; }};\n",
        binding++,
        i,
        dtype_to_glsl(x.dtype()),
        xname);
  }

  // Push constants
  os += R"(
layout(push_constant) uniform PushConstants {
  uint size;
)";
  if (!contiguous) {
    os += "  int shape[8];\n";
    os += "  int ndim;\n";
  }
  os += "} pc;\n\n";

  // Helper function for index calculation (strided)
  if (!contiguous) {
    os += R"(
uint elem_to_loc(uint idx, int start_stride_idx, int ndim) {
  uint loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    loc += (idx % uint(pc.shape[i])) * uint(in_strides[start_stride_idx + i]);
    idx /= uint(pc.shape[i]);
  }
  return loc;
}

)";
  }

  // Main kernel function
  os += "void main() {\n";
  os += fmt::format("  uint base_idx = gl_GlobalInvocationID.x * {};\n", wpt);
  os += "  if (base_idx >= pc.size) return;\n\n";

  // Work loop
  if (wpt > 1) {
    os += fmt::format(
        "  for (int w = 0; w < {} && (base_idx + w) < pc.size; ++w) {{\n", wpt);
    os += "    uint idx = base_idx + w;\n";
  } else {
    os += "  uint idx = base_idx;\n";
  }

  // Declare and load inputs into temps
  std::vector<std::string> loaded_inputs(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& x = inputs[i];
    const auto& xname = get_var_name(x);
    std::string type_str = dtype_to_glsl(x.dtype());

    if (is_constant(i)) {
      // Constants are inlined directly
      std::ostringstream ss;
      print_constant(ss, x);
      loaded_inputs[i] = fmt::format("{}({})", type_str, ss.str());
    } else if (is_scalar(x)) {
      // Scalars read from buffer[0]
      os += fmt::format("    {} t_{} = {}[0];\n", type_str, xname, xname);
      loaded_inputs[i] = fmt::format("t_{}", xname);
    } else if (contiguous) {
      // Contiguous: direct indexing
      os += fmt::format("    {} t_{} = {}[idx];\n", type_str, xname, xname);
      loaded_inputs[i] = fmt::format("t_{}", xname);
    } else {
      // Strided: compute location from strides
      // Find the binding index for this input
      int binding_idx = 0;
      for (size_t j = 0; j < input_bindings.size(); ++j) {
        if (input_bindings[j].first == static_cast<int>(i)) {
          binding_idx = static_cast<int>(j);
          break;
        }
      }
      os += fmt::format(
          "    uint loc_{} = elem_to_loc(idx, {}, {});\n",
          xname,
          binding_idx * ndim,
          ndim);
      os += fmt::format(
          "    {} t_{} = {}[loc_{}];\n", type_str, xname, xname, xname);
      loaded_inputs[i] = fmt::format("t_{}", xname);
    }
  }

  // Helper to check if an array is a constant
  auto is_constant_array = [&](const array& x) -> bool {
    // Check if this array's id is in constant_ids
    return constant_ids.find(x.id()) != constant_ids.end();
  };

  // Helper to get GLSL expression for an input to a tape operation
  auto get_input_expr = [&](const array& x) -> std::string {
    if (is_constant_array(x)) {
      // Inline constant
      std::ostringstream ss;
      print_constant(ss, x);
      return fmt::format("{}({})", dtype_to_glsl(x.dtype()), ss.str());
    } else {
      // Use temp variable
      return fmt::format("t_{}", get_var_name(x));
    }
  };

  // Replay tape operations
  for (const auto& x : tape) {
    const auto& xname = get_var_name(x);
    std::string type_str = dtype_to_glsl(x.dtype());
    const auto& prim = x.primitive();
    std::string prim_name = prim.name();

    os += fmt::format("    {} t_{} = ", type_str, xname);

    if (is_static_cast(prim)) {
      // Handle Broadcast/AsType as static casts
      os += fmt::format("{}({});\n", type_str, get_input_expr(x.inputs()[0]));
    } else {
      // Get operator or function name
      std::string op = get_glsl_operator(prim_name);

      // Check if it's a binary operator or function
      bool is_binary_op =
          (op == "+" || op == "-" || op == "*" || op == "/" || op == "==" ||
           op == "!=" || op == ">" || op == "<" || op == ">=" || op == "<=" ||
           op == "&&" || op == "||" || op == "&" || op == "|" || op == "^");

      if (is_binary_op && x.inputs().size() == 2) {
        os += fmt::format(
            "({} {} {});\n",
            get_input_expr(x.inputs()[0]),
            op,
            get_input_expr(x.inputs()[1]));
      } else if (op == "max" || op == "min") {
        // Built-in GLSL functions
        os += fmt::format(
            "{}({}, {});\n",
            op,
            get_input_expr(x.inputs()[0]),
            get_input_expr(x.inputs()[1]));
      } else {
        // Generic function call - use the mapped name (e.g., "exp" instead of
        // "Exp")
        os += fmt::format("{}(", op);
        for (size_t i = 0; i < x.inputs().size(); ++i) {
          if (i > 0)
            os += ", ";
          os += get_input_expr(x.inputs()[i]);
        }
        os += ");\n";
      }
    }
  }

  // Write outputs
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& x = outputs[i];
    const auto& xname = get_var_name(x);

    if (contiguous) {
      os += fmt::format("    {}[idx] = t_{};\n", xname, xname);
    } else {
      // For strided outputs, compute output location
      os += fmt::format(
          "    uint out_loc_{} = elem_to_loc(idx, 0, {});\n", xname, ndim);
      os += fmt::format("    {}[out_loc_{}] = t_{};\n", xname, xname, xname);
    }
  }

  if (wpt > 1) {
    os += "  }\n";
  }

  os += "}\n";
}

// Compile GLSL source to SPIR-V using shaderc
std::vector<uint32_t> compile_glsl_to_spirv(
    const std::string& glsl_source,
    const std::string& shader_name) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  // Set target environment
  options.SetTargetEnvironment(
      shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
  options.SetSourceLanguage(shaderc_source_language_glsl);
  options.SetForcedVersionProfile(450, shaderc_profile_core);

  // Optimization level
  options.SetOptimizationLevel(shaderc_optimization_level_performance);

  // Compile the shader
  auto result = compiler.CompileGlslToSpv(
      glsl_source.c_str(),
      glsl_source.size(),
      shaderc_compute_shader,
      shader_name.c_str(),
      options);

  if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
    throw std::runtime_error(
        fmt::format(
            "Failed to compile Vulkan kernel '{}': {}",
            shader_name,
            result.GetErrorMessage()));
  }

  return std::vector<uint32_t>(result.cbegin(), result.cend());
}

} // namespace

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::ScopedSyncLabel sync_label("compiled_eval_gpu");
  auto& s = stream();

  // Check for unsupported dtypes (bfloat16 requires special handling)
  bool has_unsupported_dtype = false;
  for (const auto& x : inputs_) {
    if (x.dtype() == bfloat16 || x.dtype() == complex64) {
      has_unsupported_dtype = true;
      break;
    }
  }
  if (!has_unsupported_dtype) {
    for (const auto& x : outputs_) {
      if (x.dtype() == bfloat16 || x.dtype() == complex64) {
        has_unsupported_dtype = true;
        break;
      }
    }
  }

  if (has_unsupported_dtype) {
    // Fall back to CPU for unsupported dtypes
    ::mlx::core::gpu::synchronize(s);
    auto cpu_stream = default_stream(Device::cpu);
    Compiled cpu_compiled(cpu_stream, inputs_, outputs_, tape_, constant_ids_);
    cpu_compiled.eval_cpu(inputs, outputs);
    synchronize(cpu_stream);
    return;
  }

  // Determine work per thread based on output dtype size
  int max_itemsize = 1;
  for (const auto& x : outputs_) {
    max_itemsize = std::max(max_itemsize, static_cast<int>(x.itemsize()));
  }
  int work_per_thread = 16 / max_itemsize;
  work_per_thread = std::max(work_per_thread, 1);

  // Collapse contiguous dims to route to a faster kernel if possible
  auto [contiguous, shape, strides] =
      compiled_collapse_contiguous_dims(inputs, outputs[0], is_constant_);

  // Use large index if needed
  bool large = compiled_use_large_index(inputs, outputs, contiguous);

  // Build kernel name based on configuration
  std::string kernel_name = kernel_lib_;
  if (contiguous) {
    kernel_name += "_contiguous";
  } else {
    kernel_name += fmt::format("_strided_{}", shape.size());
  }
  if (large) {
    kernel_name += "_large";
  }

  // Check if we already have this kernel compiled (simple cache check)
  auto& manager = vulkan::KernelManager::get();
  auto* existing_shader = manager.get_shader(kernel_name);

  std::vector<uint32_t> spirv;
  if (!existing_shader) {
    // Generate GLSL source
    std::string glsl_source;
    build_glsl_kernel(
        glsl_source,
        kernel_name,
        inputs_,
        outputs_,
        tape_,
        is_constant_,
        constant_ids_,
        contiguous,
        static_cast<int>(shape.size()),
        work_per_thread);

    // Compile to SPIR-V
    try {
      spirv = compile_glsl_to_spirv(glsl_source, kernel_name);
    } catch (const std::exception& e) {
      std::cerr << "=== FAILED GLSL for: " << kernel_name
                << " ===" << std::endl;
      std::cerr << glsl_source << std::endl;
      std::cerr << "=== End GLSL ===" << std::endl;
      throw;
    }

    // Register the shader
    manager.register_shader(
        kernel_name, spirv.data(), spirv.size() * sizeof(uint32_t));
  }

  // Allocate outputs with buffer donation
  compiled_allocate_outputs(
      inputs, outputs, is_constant_, contiguous, [&](size_t n) {
        return vulkan::allocator().malloc(n);
      });

  // Build dynamic descriptor set bindings
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  uint32_t binding_idx = 0;

  // Input bindings
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant_(i)) {
      continue;
    }
    bindings.push_back(
        {binding_idx++,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         1,
         VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr});
  }

  // Strides binding (for non-contiguous)
  if (!contiguous && binding_idx > 0) {
    bindings.push_back(
        {binding_idx++,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         1,
         VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr});
  }

  // Output bindings
  for (size_t i = 0; i < outputs.size(); ++i) {
    bindings.push_back(
        {binding_idx++,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         1,
         VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr});
  }

  // Get or create pipeline
  uint32_t push_constant_size =
      contiguous ? sizeof(uint32_t) : sizeof(uint32_t) + sizeof(int) * 9;

  auto* pipeline =
      manager.get_pipeline(kernel_name, bindings, push_constant_size);

  // Get command buffer
  auto cmd_buffer = vulkan::begin_command_recording(s.index);

  // Allocate descriptor set
  auto descriptor_set =
      manager.allocate_descriptor_set(pipeline->descriptor_layout);

  // Prepare descriptor writes
  std::vector<VkWriteDescriptorSet> writes;
  std::vector<VkDescriptorBufferInfo> buffer_infos;
  buffer_infos.reserve(binding_idx);

  // Helper to add buffer info
  auto add_buffer = [&](const array& arr) {
    auto* vulkan_buffer = static_cast<const vulkan::VulkanBuffer*>(
        static_cast<const void*>(arr.buffer().ptr()));
    if (!vulkan_buffer || vulkan_buffer->buffer == VK_NULL_HANDLE) {
      throw std::runtime_error("Missing Vulkan buffer for compiled kernel");
    }

    VkDescriptorBufferInfo info{};
    info.buffer = vulkan_buffer->buffer;
    info.offset = 0;
    info.range = VK_WHOLE_SIZE;
    buffer_infos.push_back(info);
  };

  // Input buffers
  uint32_t write_idx = 0;
  std::vector<int64_t> all_strides;

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant_(i)) {
      continue;
    }
    add_buffer(inputs[i]);
    writes.push_back({});
    writes[write_idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[write_idx].dstSet = descriptor_set;
    writes[write_idx].dstBinding = write_idx;
    writes[write_idx].dstArrayElement = 0;
    writes[write_idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[write_idx].descriptorCount = 1;
    writes[write_idx].pBufferInfo = &buffer_infos[write_idx];
    ++write_idx;

    // Collect strides for non-contiguous access
    if (!contiguous && !is_scalar(inputs[i])) {
      // strides[0] is output strides, find the right stride index for this
      // input
      int stride_idx = 1; // Start after output strides
      for (size_t j = 0; j < i; ++j) {
        if (!is_constant_(j) && !is_scalar(inputs[j])) {
          stride_idx++;
        }
      }
      if (stride_idx < static_cast<int>(strides.size())) {
        all_strides.insert(
            all_strides.end(),
            strides[stride_idx].begin(),
            strides[stride_idx].end());
      }
    }
  }

  // Strides buffer (for non-contiguous) - skip for now, use push constants
  if (!contiguous && !all_strides.empty()) {
    // TODO: Upload strides to a buffer
    // For now we'll skip this complex case
  }

  // Output buffers
  for (size_t i = 0; i < outputs.size(); ++i) {
    add_buffer(outputs[i]);
    writes.push_back({});
    writes[write_idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[write_idx].dstSet = descriptor_set;
    writes[write_idx].dstBinding = write_idx;
    writes[write_idx].dstArrayElement = 0;
    writes[write_idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[write_idx].descriptorCount = 1;
    writes[write_idx].pBufferInfo = &buffer_infos[write_idx];
    ++write_idx;
  }

  // Update descriptor sets
  if (!writes.empty()) {
    vkUpdateDescriptorSets(
        vulkan::VulkanContext::get().device(),
        static_cast<uint32_t>(writes.size()),
        writes.data(),
        0,
        nullptr);
  }

  // Bind pipeline and descriptor set
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

  // Set push constants
  struct PushConstants {
    uint32_t size;
    int shape[8];
    int ndim;
  } pc;

  // TODO: Handle arrays larger than 2^32 elements
  pc.size = static_cast<uint32_t>(outputs[0].data_size());
  if (!contiguous) {
    pc.ndim = static_cast<int>(shape.size());
    for (size_t i = 0; i < shape.size() && i < 8; ++i) {
      pc.shape[i] = static_cast<int>(shape[i]);
    }
  }

  vkCmdPushConstants(
      cmd_buffer,
      pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      contiguous ? sizeof(uint32_t) : sizeof(pc),
      &pc);

  // Dispatch
  uint64_t num_elements = outputs[0].data_size();
  uint32_t workgroups = static_cast<uint32_t>(
      (num_elements + 256ULL * static_cast<uint64_t>(work_per_thread) - 1) /
      (256ULL * static_cast<uint64_t>(work_per_thread)));
  workgroups = std::max(workgroups, 1u);

  vkCmdDispatch(cmd_buffer, workgroups, 1, 1);

  // Defer descriptor set cleanup
  manager.defer_descriptor_set_free(s.index, descriptor_set);
}

} // namespace mlx::core
