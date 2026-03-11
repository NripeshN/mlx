// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

namespace {

template <typename Primitive>
bool try_eval_unary_op_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::string& shader_name,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f) {
  if (inputs.size() != 1) {
    return false;
  }

  array in = inputs[0];
  const bool complex_io = in.dtype() == complex64 && out.dtype() == complex64;
  if ((!is_vulkan_float_dtype(in.dtype()) && !complex_io) ||
      in.dtype() != out.dtype()) {
    return false;
  }

  const bool use_f32_staging_io =
      !complex_io && (in.dtype() == bfloat16 || out.dtype() == bfloat16);
  if (use_f32_staging_io) {
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }

  if (!is_supported_unary_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }

  const bool staged_output =
      use_f32_staging_io || !is_supported_unary_layout(out);
  array out_work = staged_output
      ? array(
            out.shape(),
            use_f32_staging_io ? float32 : out.dtype(),
            nullptr,
            {})
      : out;

  set_unary_output_data(in, out_work);
  if (!is_supported_unary_layout(in) || !is_supported_unary_layout(out_work)) {
    return false;
  }

  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_unary_op(
        in, out_work, shader_name, command_buffer, s, param1, param2);
    vulkan::end_command_recording(s.index);
    if (staged_output || use_f32_staging_io) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "unary_dispatch_failed shader=" << shader_name
          << " reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

template <typename Primitive>
void eval_unary_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::string& shader_name,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f) {
  if (!try_eval_unary_op_vulkan<Primitive>(
          inputs, out, shader_name, s, param1, param2)) {
    throw std::runtime_error(
        std::string("Unary operation ") + shader_name +
        " failed on Vulkan (unsupported dtype or layout).");
  }
}

template <typename Primitive>
bool try_eval_generic_unary_op_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::string& shader_name,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f,
    float param3 = 0.0f,
    float param4 = 0.0f) {
  if (inputs.size() != 1) {
    return false;
  }

  array in = inputs[0];
  if (!is_vulkan_float_dtype(in.dtype()) || in.dtype() != out.dtype()) {
    return false;
  }

  const bool use_f32_staging_io =
      in.dtype() == bfloat16 || out.dtype() == bfloat16;
  if (use_f32_staging_io) {
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }

  if (!is_supported_generic_unary_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }

  const bool staged_output =
      use_f32_staging_io || !is_supported_generic_unary_layout(out);
  array out_work = staged_output
      ? array(
            out.shape(),
            use_f32_staging_io ? float32 : out.dtype(),
            nullptr,
            {})
      : out;

  set_unary_output_data(in, out_work);
  if (!is_supported_generic_unary_layout(in) ||
      !is_supported_generic_unary_layout(out_work)) {
    return false;
  }

  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_generic_unary_op(
        in,
        out_work,
        shader_name,
        command_buffer,
        s,
        param1,
        param2,
        param3,
        param4);
    vulkan::end_command_recording(s.index);
    if (staged_output || use_f32_staging_io) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "generic_unary_dispatch_failed shader=" << shader_name
          << " reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

template <typename Primitive>
void eval_generic_unary_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::string& shader_name,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f,
    float param3 = 0.0f,
    float param4 = 0.0f) {
  if (!try_eval_generic_unary_op_vulkan<Primitive>(
          inputs, out, shader_name, s, param1, param2, param3, param4)) {
    throw std::runtime_error(
        std::string("Unary operation ") + shader_name +
        " failed on Vulkan (unsupported dtype or layout).");
  }
}

template <typename Primitive>
void eval_generic_unary_suffix_vulkan(
    const std::vector<array>& inputs,
    array& out,
    std::string_view op_name,
    Stream s,
    bool f16_with_rte = false) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto suffix = dtype_suffix(out.dtype());
    if (suffix.empty() && out.dtype() == bfloat16) {
      suffix = "f32";
    }
    if (!suffix.empty()) {
      std::string shader_name = std::string(op_name) + "_" + suffix;
      if (f16_with_rte && out.dtype() == float16) {
        shader_name += "_rte";
      }
      eval_generic_unary_vulkan<Primitive>(inputs, out, shader_name, s);
      return;
    }
  }
  throw std::runtime_error(
      std::string("Unary operation ") + std::string(op_name) +
      " failed on Vulkan (unsupported dtype or layout).");
}

} // namespace

#define VULKAN_GENERIC_UNARY_GPU(func, op_name)                             \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) {       \
    eval_generic_unary_suffix_vulkan<func>(inputs, out, op_name, stream()); \
  }

#define VULKAN_GENERIC_UNARY_RTE_GPU(func, op_name)                   \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    eval_generic_unary_suffix_vulkan<func>(                           \
        inputs, out, op_name, stream(), true);                        \
  }

// Generic unary ops
VULKAN_GENERIC_UNARY_GPU(Abs, "abs")
VULKAN_GENERIC_UNARY_GPU(Ceil, "ceil")
VULKAN_GENERIC_UNARY_RTE_GPU(Exp, "exp")
VULKAN_GENERIC_UNARY_GPU(Floor, "floor")
VULKAN_GENERIC_UNARY_GPU(Negative, "neg")
VULKAN_GENERIC_UNARY_GPU(Round, "round")
VULKAN_GENERIC_UNARY_GPU(Sigmoid, "sigmoid")
VULKAN_GENERIC_UNARY_GPU(Tanh, "tanh")

// Specialized unary ops
void Cos::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    if (try_eval_unary_op_vulkan<Cos>(inputs, out, "cos_f32", stream())) {
      return;
    }
  }
  throw std::runtime_error(
      "Cos operation failed on Vulkan (unsupported dtype or layout).");
}

void Erf::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    if (try_eval_unary_op_vulkan<Erf>(inputs, out, "erf_f32", stream())) {
      return;
    }
  }
  throw std::runtime_error(
      "Erf operation failed on Vulkan (unsupported dtype or layout).");
}

void ErfInv::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    if (try_eval_unary_op_vulkan<ErfInv>(inputs, out, "erfinv_f32", stream())) {
      return;
    }
  }
  throw std::runtime_error(
      "ErfInv operation failed on Vulkan (unsupported dtype or layout).");
}

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    if (state() == Log::e && out.dtype() == float32) {
      if (try_eval_unary_op_vulkan<Log>(inputs, out, "log_f32", stream())) {
        return;
      }
    }
    if (state() == Log::e && out.dtype() == float16) {
      if (try_eval_unary_op_vulkan<Log>(inputs, out, "log_f16_rte", stream())) {
        return;
      }
    }
  }
  throw std::runtime_error(
      "Log operation failed on Vulkan (unsupported dtype or layout).");
}

void Sin::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    if (try_eval_unary_op_vulkan<Sin>(inputs, out, "sin_f32", stream())) {
      return;
    }
  }
  throw std::runtime_error(
      "Sin operation failed on Vulkan (unsupported dtype or layout).");
}

void Square::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    if (out.dtype() == float32 || out.dtype() == bfloat16) {
      eval_unary_vulkan<Square>(inputs, out, "sqr_f32", stream());
      return;
    }
    if (out.dtype() == float16) {
      eval_unary_vulkan<Square>(inputs, out, "sqr_f16", stream());
      return;
    }
    if (out.dtype() == complex64) {
      eval_unary_vulkan<Square>(inputs, out, "sqr_c64", stream());
      return;
    }
  }
  throw std::runtime_error(
      "Square operation failed on Vulkan (unsupported dtype or layout).");
}

void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 &&
      (inputs[0].dtype() == float32 || inputs[0].dtype() == bfloat16 ||
       inputs[0].dtype() == float16 || inputs[0].dtype() == complex64) &&
      out.dtype() == inputs[0].dtype()) {
    const char* shader = state() ? "rsqrt_f32" : "sqrt_f32";
    if (out.dtype() == complex64) {
      shader = state() ? "rsqrt_c64" : "sqrt_c64";
    } else if (out.dtype() == float16) {
      shader = state() ? "rsqrt_f16" : "sqrt_f16";
    }
    eval_unary_vulkan<Sqrt>(inputs, out, shader, stream(), 0.0f, 0.0f);
    return;
  }
  throw std::runtime_error(
      "Sqrt operation failed on Vulkan (unsupported dtype or layout).");
}

} // namespace mlx::core
