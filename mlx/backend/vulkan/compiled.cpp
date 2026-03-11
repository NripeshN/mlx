// Copyright © 2024 Apple Inc.

#include <fmt/format.h>
#include <shaderc/shaderc.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <sstream>
#include <unordered_set>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/dtype_utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

bool trace_compiled_glsl_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_TRACE_COMPILED_GLSL");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

bool trace_compiled_runtime_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_TRACE_COMPILED_RUNTIME");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

std::string join_primitive_names(const std::vector<array>& arrays) {
  std::ostringstream os;
  bool first = true;
  for (const auto& x : arrays) {
    if (!first) {
      os << ",";
    }
    first = false;
    os << x.primitive().name();
  }
  return os.str();
}

std::string dtype_to_glsl_storage(Dtype d) {
  switch (d) {
    case float32:
      return "float";
    case float16:
      return "float16_t";
    case bfloat16:
      return "uint16_t";
    case complex64:
      return "vec2";
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
      return "uint8_t";
    default:
      throw std::runtime_error(
          fmt::format(
              "Unsupported dtype for Vulkan compiled: {}", dtype_to_string(d)));
  }
}

std::string dtype_to_glsl_compute(Dtype d) {
  switch (d) {
    case bfloat16:
      return "float";
    case complex64:
      return "vec2";
    default:
      return dtype_to_glsl_storage(d);
  }
}

bool has_dtype(const std::vector<array>& arrays, Dtype dtype) {
  for (const auto& x : arrays) {
    if (x.dtype() == dtype) {
      return true;
    }
  }
  return false;
}

bool has_any_dtype(
    const std::vector<array>& arrays,
    std::initializer_list<Dtype> dtypes) {
  for (const auto& x : arrays) {
    for (auto dtype : dtypes) {
      if (x.dtype() == dtype) {
        return true;
      }
    }
  }
  return false;
}

std::string glsl_float_constant(float v) {
  if (std::isnan(v)) {
    return "uintBitsToFloat(0x7fc00000u)";
  }
  if (std::isinf(v)) {
    return v > 0 ? "uintBitsToFloat(0x7f800000u)"
                 : "uintBitsToFloat(0xff800000u)";
  }
  return fmt::format("{:.9g}", v);
}

std::string glsl_constant(const array& x) {
  switch (x.dtype()) {
    case bool_:
      return x.item<bool>() ? "true" : "false";
    case float32:
      return fmt::format("float({})", glsl_float_constant(x.item<float>()));
    case complex64: {
      auto v = x.item<complex64_t>();
      return fmt::format(
          "vec2({}, {})",
          glsl_float_constant(v.real()),
          glsl_float_constant(v.imag()));
    }
    case bfloat16: {
      auto v = x.item<bfloat16_t>();
      return fmt::format(
          "float({})", glsl_float_constant(static_cast<float>(v)));
    }
    case float16: {
      auto v = x.item<float16_t>();
      return fmt::format(
          "float16_t({})", glsl_float_constant(static_cast<float>(v)));
    }
    default: {
      std::ostringstream ss;
      print_constant(ss, x);
      return fmt::format("{}({})", dtype_to_glsl_compute(x.dtype()), ss.str());
    }
  }
}

std::string glsl_cast_expr(Dtype dst, Dtype src, const std::string& expr) {
  if (dst == bool_ && src != bool_) {
    return fmt::format("bool({})", expr);
  }
  if (dst != bool_ && src == bool_) {
    return fmt::format("{}(({}) ? 1 : 0)", dtype_to_glsl_compute(dst), expr);
  }
  if (dst == complex64 && src != complex64) {
    return fmt::format("vec2({}, 0.0)", expr);
  }
  if (dst != complex64 && src == complex64) {
    return fmt::format("{}(({}).x)", dtype_to_glsl_compute(dst), expr);
  }
  return fmt::format("{}({})", dtype_to_glsl_compute(dst), expr);
}

bool supports_primitive_name(const std::string& prim_name) {
  static const std::unordered_set<std::string> supported = {
      "Abs",       "Add",     "AsType",  "Broadcast", "Ceil",     "Conjugate",
      "Cos",       "Divide",  "Exp",     "Floor",     "Imag",     "Log",
      "LogAddExp", "Maximum", "Minimum", "Multiply",  "Negative", "Real",
      "Round",     "Sigmoid", "Sin",     "Sqrt",      "Subtract", "Tan"};
  return supported.contains(prim_name);
}

std::string emit_glsl_preamble(
    bool uses_bfloat16,
    bool uses_complex64,
    bool uses_float16_types,
    bool uses_int16_types,
    bool uses_int8_types) {
  std::ostringstream os;
  os << "#version 450\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n";
  if (uses_float16_types) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
  }
  if (uses_float16_types || uses_int16_types || uses_bfloat16) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    os << "#extension GL_EXT_shader_16bit_storage : require\n";
  }
  if (uses_int8_types) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
  }
  os << "\n";

  if (uses_bfloat16) {
    os << R"(
uint fp32_to_bf16(float f) {
  uint u = floatBitsToUint(f);
  u = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;
  return u;
}

float bf16_to_fp32(uint u) {
  return uintBitsToFloat(u << 16);
}

)";
  }

  if (uses_complex64) {
    os << R"(
vec2 complex_mul(vec2 a, vec2 b) {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 complex_div(vec2 a, vec2 b) {
  float denom = dot(b, b);
  return vec2(
      (a.x * b.x + a.y * b.y) / denom,
      (a.y * b.x - a.x * b.y) / denom);
}

vec2 complex_exp(vec2 z) {
  float e = exp(z.x);
  return vec2(e * cos(z.y), e * sin(z.y));
}

vec2 complex_log(vec2 z) {
  return vec2(log(length(z)), atan(z.y, z.x));
}

vec2 complex_sin(vec2 z) {
  return vec2(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}

vec2 complex_cos(vec2 z) {
  return vec2(cos(z.x) * cosh(z.y), -sin(z.x) * sinh(z.y));
}

vec2 complex_tan(vec2 z) {
  return complex_div(complex_sin(z), complex_cos(z));
}

vec2 complex_sigmoid(vec2 z) {
  return complex_div(vec2(1.0, 0.0), vec2(1.0, 0.0) + complex_exp(-z));
}

float complex_abs(vec2 z) {
  return length(z);
}

vec2 complex_conjugate(vec2 z) {
  return vec2(z.x, -z.y);
}

)";
  }

  os << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  return os.str();
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
    int work_per_thread,
    const Shape* strided_shape = nullptr,
    const std::vector<Strides>* strided_strides = nullptr) {
  if (!contiguous && (strided_shape == nullptr || strided_strides == nullptr)) {
    throw std::runtime_error(
        "Missing runtime shape/strides for Vulkan compiled strided kernel");
  }

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

  bool uses_bfloat16 = has_dtype(inputs, bfloat16) ||
      has_dtype(outputs, bfloat16) || has_dtype(tape, bfloat16);
  bool uses_complex64 = has_dtype(inputs, complex64) ||
      has_dtype(outputs, complex64) || has_dtype(tape, complex64);
  bool uses_float16_types = has_dtype(inputs, float16) ||
      has_dtype(outputs, float16) || has_dtype(tape, float16);
  bool uses_int16_types = has_any_dtype(inputs, {int16, uint16}) ||
      has_any_dtype(outputs, {int16, uint16}) ||
      has_any_dtype(tape, {int16, uint16});
  bool uses_int8_types = has_any_dtype(inputs, {int8, uint8, bool_}) ||
      has_any_dtype(outputs, {int8, uint8, bool_}) ||
      has_any_dtype(tape, {int8, uint8, bool_});

  // GLSL header
  os = emit_glsl_preamble(
      uses_bfloat16,
      uses_complex64,
      uses_float16_types,
      uses_int16_types,
      uses_int8_types);

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

    os += fmt::format(
        "layout(binding = {}) readonly buffer Buf{} {{ {} {}[]; }};\n",
        binding++,
        i,
        dtype_to_glsl_storage(x.dtype()),
        xname);
    input_bindings.push_back({static_cast<int>(i), xname});
  }

  // Strides buffer for non-contiguous access
  // Output buffers
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& x = outputs[i];
    const auto& xname = get_var_name(x);
    os += fmt::format(
        "layout(binding = {}) buffer OutBuf{} {{ {} {}[]; }};\n",
        binding++,
        i,
        dtype_to_glsl_storage(x.dtype()),
        xname);
  }

  // Push constants
  os += R"(
layout(push_constant) uniform PushConstants {
  uint size;
} pc;
 
)";

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
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& x = inputs[i];
    const auto& xname = get_var_name(x);
    std::string type_str = dtype_to_glsl_compute(x.dtype());

    if (is_constant(i)) {
      continue;
    } else if (is_scalar(x)) {
      if (x.dtype() == bool_) {
        os += fmt::format(
            "    {} t_{} = ({}[0] != uint8_t(0));\n", type_str, xname, xname);
      } else if (x.dtype() == bfloat16) {
        os += fmt::format(
            "    {} t_{} = bf16_to_fp32(uint({}[0]));\n",
            type_str,
            xname,
            xname);
      } else {
        os += fmt::format("    {} t_{} = {}[0];\n", type_str, xname, xname);
      }
    } else if (contiguous) {
      if (x.dtype() == bool_) {
        os += fmt::format(
            "    {} t_{} = ({}[idx] != uint8_t(0));\n", type_str, xname, xname);
      } else if (x.dtype() == bfloat16) {
        os += fmt::format(
            "    {} t_{} = bf16_to_fp32(uint({}[idx]));\n",
            type_str,
            xname,
            xname);
      } else {
        os += fmt::format("    {} t_{} = {}[idx];\n", type_str, xname, xname);
      }
    } else {
      // Strided: compute location from strides
      int stride_idx = 1;
      for (size_t j = 0; j < i; ++j) {
        if (!is_constant(j) && !is_scalar(inputs[j])) {
          stride_idx++;
        }
      }
      const auto& runtime_shape = *strided_shape;
      const auto& runtime_input_strides = (*strided_strides)[stride_idx];
      os += fmt::format("    uint loc_{} = 0u;\n", xname);
      os += fmt::format("    uint rem_{} = idx;\n", xname);
      for (int axis = ndim - 1; axis >= 0; --axis) {
        os += fmt::format(
            "    uint coord_{0}_{1} = rem_{0} % uint({2});\n",
            xname,
            axis,
            runtime_shape[axis]);
        os += fmt::format(
            "    rem_{0} /= uint({1});\n", xname, runtime_shape[axis]);
        os += fmt::format(
            "    loc_{0} += coord_{0}_{1} * uint({2});\n",
            xname,
            axis,
            runtime_input_strides[axis]);
      }
      if (x.dtype() == bool_) {
        os += fmt::format(
            "    {} t_{} = ({}[loc_{}] != uint8_t(0));\n",
            type_str,
            xname,
            xname,
            xname);
      } else if (x.dtype() == bfloat16) {
        os += fmt::format(
            "    {} t_{} = bf16_to_fp32(uint({}[loc_{}]));\n",
            type_str,
            xname,
            xname,
            xname);
      } else {
        os += fmt::format(
            "    {} t_{} = {}[loc_{}];\n", type_str, xname, xname, xname);
      }
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
      return glsl_constant(x);
    } else {
      // Use temp variable
      return fmt::format("t_{}", get_var_name(x));
    }
  };

  // Replay tape operations
  for (const auto& x : tape) {
    const auto& xname = get_var_name(x);
    std::string type_str = dtype_to_glsl_compute(x.dtype());
    const auto& prim = x.primitive();
    std::string prim_name = prim.name();

    os += fmt::format("    {} t_{} = ", type_str, xname);

    if (is_static_cast(prim)) {
      // Handle Broadcast/AsType as static casts
      os += fmt::format(
          "{};\n",
          glsl_cast_expr(
              x.dtype(), x.inputs()[0].dtype(), get_input_expr(x.inputs()[0])));
    } else {
      if (!supports_primitive_name(prim_name)) {
        throw std::runtime_error(
            fmt::format(
                "Unsupported primitive '{}' in Vulkan compiled kernel",
                prim_name));
      }

      // Get operator or function name
      std::string op = get_glsl_operator(prim_name);

      // Check if it's a binary operator or function
      bool is_binary_op =
          (op == "+" || op == "-" || op == "*" || op == "/" || op == "==" ||
           op == "!=" || op == ">" || op == "<" || op == ">=" || op == "<=" ||
           op == "&&" || op == "||" || op == "&" || op == "|" || op == "^");

      bool is_complex = x.dtype() == complex64;

      if (prim_name == "Negative" && x.inputs().size() == 1) {
        os += fmt::format("(-{});\n", get_input_expr(x.inputs()[0]));
      } else if (prim_name == "LogAddExp" && x.inputs().size() == 2) {
        auto lhs = get_input_expr(x.inputs()[0]);
        auto rhs = get_input_expr(x.inputs()[1]);
        os += fmt::format(
            "((min({0}, {1}) == -1.0 / 0.0 || max({0}, {1}) == 1.0 / 0.0) ? max({0}, {1}) : (max({0}, {1}) + log(1.0 + exp(min({0}, {1}) - max({0}, {1})))));\n",
            lhs,
            rhs);
      } else if (is_binary_op && x.inputs().size() == 2) {
        if (is_complex && op == "*") {
          os += fmt::format(
              "complex_mul({}, {});\n",
              get_input_expr(x.inputs()[0]),
              get_input_expr(x.inputs()[1]));
        } else if (is_complex && op == "/") {
          os += fmt::format(
              "complex_div({}, {});\n",
              get_input_expr(x.inputs()[0]),
              get_input_expr(x.inputs()[1]));
        } else {
          os += fmt::format(
              "({} {} {});\n",
              get_input_expr(x.inputs()[0]),
              op,
              get_input_expr(x.inputs()[1]));
        }
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
        std::string fn = op;
        if (prim_name == "Real" && !x.inputs().empty() &&
            x.inputs()[0].dtype() == complex64) {
          os += fmt::format("({}).x;\n", get_input_expr(x.inputs()[0]));
        } else if (
            prim_name == "Imag" && !x.inputs().empty() &&
            x.inputs()[0].dtype() == complex64) {
          os += fmt::format("({}).y;\n", get_input_expr(x.inputs()[0]));
        } else if (x.dtype() == complex64) {
          if (op == "exp") {
            fn = "complex_exp";
          } else if (op == "log") {
            fn = "complex_log";
          } else if (op == "sin") {
            fn = "complex_sin";
          } else if (op == "cos") {
            fn = "complex_cos";
          } else if (op == "tan") {
            fn = "complex_tan";
          } else if (op == "sigmoid") {
            fn = "complex_sigmoid";
          } else if (op == "abs") {
            fn = "complex_abs";
          } else if (prim_name == "Conjugate") {
            fn = "complex_conjugate";
          }
        } else if (
            prim_name == "Abs" && !x.inputs().empty() &&
            x.inputs()[0].dtype() == complex64) {
          fn = "complex_abs";
        } else if (
            prim_name == "Conjugate" && !x.inputs().empty() &&
            x.inputs()[0].dtype() == complex64) {
          fn = "complex_conjugate";
        } else if (op == "sigmoid") {
          fn = "(1.0 / (1.0 + exp(-";
        }

        if ((prim_name == "Real" || prim_name == "Imag") &&
            !x.inputs().empty() && x.inputs()[0].dtype() == complex64) {
          // Already emitted.
        } else if (op == "sigmoid" && x.dtype() != complex64) {
          os += fn;
          os += get_input_expr(x.inputs()[0]);
          os += ")));\n";
        } else {
          os += fmt::format("{}(", fn);
          for (size_t i = 0; i < x.inputs().size(); ++i) {
            if (i > 0)
              os += ", ";
            os += get_input_expr(x.inputs()[i]);
          }
          os += ");\n";
        }
      }
    }
  }

  // Write outputs
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& x = outputs[i];
    const auto& xname = get_var_name(x);

    if (contiguous) {
      if (x.dtype() == bool_) {
        os +=
            fmt::format("    {}[idx] = uint8_t(t_{} ? 1 : 0);\n", xname, xname);
      } else if (x.dtype() == bfloat16) {
        os += fmt::format(
            "    {}[idx] = uint16_t(fp32_to_bf16(t_{}));\n", xname, xname);
      } else {
        os += fmt::format("    {}[idx] = t_{};\n", xname, xname);
      }
    } else {
      if (x.dtype() == bool_) {
        os +=
            fmt::format("    {}[idx] = uint8_t(t_{} ? 1 : 0);\n", xname, xname);
      } else if (x.dtype() == bfloat16) {
        os += fmt::format(
            "    {}[idx] = uint16_t(fp32_to_bf16(t_{}));\n", xname, xname);
      } else {
        os += fmt::format("    {}[idx] = t_{};\n", xname, xname);
      }
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

  // Check for unsupported dtypes.
  bool has_unsupported_dtype = false;
  for (const auto& x : inputs_) {
    if (x.dtype() == float64) {
      has_unsupported_dtype = true;
      break;
    }
  }
  if (!has_unsupported_dtype) {
    for (const auto& x : outputs_) {
      if (x.dtype() == float64) {
        has_unsupported_dtype = true;
        break;
      }
    }
  }

  if (has_unsupported_dtype) {
    throw std::runtime_error(
        "Compiled kernel failed on Vulkan (unsupported dtype).");
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
  auto dispatch_inputs = inputs;

  if (!contiguous) {
    work_per_thread = 1;
  }

  const auto requires_cpu_fallback = [&]() {
    auto has_nonzero_offset = [](const std::vector<array>& arrays) {
      return std::any_of(arrays.begin(), arrays.end(), [](const array& x) {
        return x.offset() != 0;
      });
    };

    if (has_nonzero_offset(dispatch_inputs) || has_nonzero_offset(outputs) ||
        has_nonzero_offset(tape_)) {
      return true;
    }

    auto has_bool_dtype = [](const std::vector<array>& arrays) {
      return std::any_of(arrays.begin(), arrays.end(), [](const array& x) {
        return x.dtype() == bool_;
      });
    };

    if (has_bool_dtype(inputs_) || has_bool_dtype(outputs_) ||
        has_bool_dtype(tape_)) {
      return true;
    }

    return std::any_of(tape_.begin(), tape_.end(), [](const array& x) {
      return !is_static_cast(x.primitive()) &&
          !supports_primitive_name(x.primitive().name());
    });
  }();

  if (requires_cpu_fallback) {
    std::ostringstream msg;
    msg << "Compiled kernel failed on Vulkan (complex tape operations or unsupported layout)."
        << " contiguous=" << contiguous << " inputs_offset="
        << std::any_of(
               inputs.begin(),
               inputs.end(),
               [](const array& x) { return x.offset() != 0; })
        << " outputs_offset="
        << std::any_of(
               outputs.begin(),
               outputs.end(),
               [](const array& x) { return x.offset() != 0; })
        << " tape_offset="
        << std::any_of(
               tape_.begin(),
               tape_.end(),
               [](const array& x) { return x.offset() != 0; })
        << " tape_primitives=" << join_primitive_names(tape_);
    throw std::runtime_error(msg.str());
  }

  // Use large index if needed
  bool large = compiled_use_large_index(dispatch_inputs, outputs, contiguous);

  // Build kernel name based on configuration
  std::string kernel_name = kernel_lib_;
  if (contiguous) {
    kernel_name += "_contiguous";
  } else {
    kernel_name += fmt::format("_strided_{}", shape.size());
    std::ostringstream layout_key;
    layout_key << "shape=";
    for (auto dim : shape) {
      layout_key << dim << ",";
    }
    for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
      if (is_constant_(i) || is_scalar(dispatch_inputs[i])) {
        continue;
      }
      int stride_idx = 1;
      for (size_t j = 0; j < i; ++j) {
        if (!is_constant_(j) && !is_scalar(dispatch_inputs[j])) {
          stride_idx++;
        }
      }
      layout_key << "s" << i << "=";
      for (auto stride : strides[stride_idx]) {
        layout_key << stride << ",";
      }
    }
    kernel_name +=
        fmt::format("_layout_{}", std::hash<std::string>{}(layout_key.str()));
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
        work_per_thread,
        contiguous ? nullptr : &shape,
        contiguous ? nullptr : &strides);

    if (trace_compiled_glsl_enabled()) {
      std::cerr << "=== Vulkan compiled GLSL: " << kernel_name << " ===\n"
                << glsl_source << "\n=== End GLSL ===\n";
    }

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
  std::vector<array> allocation_inputs;
  if (contiguous) {
    allocation_inputs = dispatch_inputs;
  }
  compiled_allocate_outputs(
      allocation_inputs, outputs, is_constant_, contiguous, [&](size_t n) {
        return vulkan::allocator().malloc(n);
      });

  // Build dynamic descriptor set bindings
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  uint32_t binding_idx = 0;

  // Input bindings
  for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
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
  uint32_t push_constant_size = sizeof(uint32_t);

  auto* pipeline =
      manager.get_pipeline(kernel_name, bindings, push_constant_size);

  // Get command buffer
  auto cmd_buffer = vulkan::begin_command_recording(s.index);
  const uint64_t descriptor_epoch = vulkan::descriptor_epoch_for_stream(s);

  // Allocate descriptor set
  auto descriptor_set =
      manager.allocate_descriptor_set(pipeline->descriptor_layout);

  // Prepare descriptor writes
  std::vector<VkWriteDescriptorSet> writes;
  std::vector<VkDescriptorBufferInfo> buffer_infos;
  buffer_infos.reserve(binding_idx);

  // Helper to add buffer info
  auto add_buffer = [&](const array& arr) {
    vulkan::retain_array_for_stream(s, arr);
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
  std::vector<int32_t> all_strides;

  for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
    if (is_constant_(i)) {
      continue;
    }
    add_buffer(dispatch_inputs[i]);
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

  if (trace_compiled_runtime_enabled()) {
    std::cerr << "[vulkan-compiled] kernel=" << kernel_name
              << " contiguous=" << contiguous << " shape=" << shape
              << " all_strides=" << all_strides
              << " out_data_size=" << outputs[0].data_size()
              << " out_size=" << outputs[0].size() << "\n";
    for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
      if (dispatch_inputs[i].dtype() == float32 &&
          dispatch_inputs[i].size() > 0) {
        auto* ptr =
            static_cast<const float*>(dispatch_inputs[i].buffer().raw_ptr());
        std::cerr << "  input[" << i << "] first=" << ptr[0]
                  << " size=" << dispatch_inputs[i].size() << "\n";
      }
    }
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
  } pc;

  // TODO: Handle arrays larger than 2^32 elements
  pc.size = static_cast<uint32_t>(outputs[0].data_size());

  vkCmdPushConstants(
      cmd_buffer,
      pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(pc),
      &pc);

  // Dispatch
  uint64_t num_elements = outputs[0].data_size();
  uint32_t workgroups = static_cast<uint32_t>(
      (num_elements + 256ULL * static_cast<uint64_t>(work_per_thread) - 1) /
      (256ULL * static_cast<uint64_t>(work_per_thread)));
  workgroups = std::max(workgroups, 1u);

  vkCmdDispatch(cmd_buffer, workgroups, 1, 1);

  // Defer descriptor set cleanup
  manager.defer_descriptor_set_free(s.index, descriptor_epoch, descriptor_set);
}

} // namespace mlx::core
