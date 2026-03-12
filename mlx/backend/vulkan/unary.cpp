// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

namespace {

template <typename Primitive>
bool try_eval_unary_op_vulkan(
    const std::vector<array>& inputs,
    array& out,
    vulkan::StaticShaderId shader_id,
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
        in, out_work, shader_id, command_buffer, s, param1, param2);
    vulkan::end_command_recording(s.index);
    if (staged_output || use_f32_staging_io) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "unary_dispatch_failed shader="
          << vulkan::static_shader_name(shader_id) << " reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

template <typename Primitive>
void eval_unary_vulkan(
    const std::vector<array>& inputs,
    array& out,
    vulkan::StaticShaderId shader_id,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f) {
  if (!try_eval_unary_op_vulkan<Primitive>(
          inputs, out, shader_id, s, param1, param2)) {
    throw std::runtime_error(
        std::string("Unary operation ") +
        vulkan::static_shader_name(shader_id) +
        " failed on Vulkan (unsupported dtype or layout).");
  }
}

template <typename Primitive>
bool try_eval_generic_unary_op_vulkan(
    const std::vector<array>& inputs,
    array& out,
    vulkan::StaticShaderId shader_id,
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
        shader_id,
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
      oss << "generic_unary_dispatch_failed shader="
          << vulkan::static_shader_name(shader_id) << " reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

template <typename Primitive>
void eval_generic_unary_vulkan(
    const std::vector<array>& inputs,
    array& out,
    vulkan::StaticShaderId shader_id,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f,
    float param3 = 0.0f,
    float param4 = 0.0f) {
  if (!try_eval_generic_unary_op_vulkan<Primitive>(
          inputs, out, shader_id, s, param1, param2, param3, param4)) {
    throw std::runtime_error(
        std::string("Unary operation ") +
        vulkan::static_shader_name(shader_id) +
        " failed on Vulkan (unsupported dtype or layout).");
  }
}

template <typename Primitive>
void eval_generic_unary_suffix_vulkan(
    const std::vector<array>& inputs,
    array& out,
    GenericUnaryShaderOp op,
    Stream s,
    bool f16_with_rte = false) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    Dtype shader_dtype = out.dtype() == bfloat16 ? float32 : out.dtype();
    auto shader_id = generic_unary_shader_id(
        op, shader_dtype, f16_with_rte && out.dtype() == float16);
    if (shader_id.has_value()) {
      eval_generic_unary_vulkan<Primitive>(inputs, out, *shader_id, s);
      return;
    }
  }
  throw std::runtime_error(
      "Unary operation failed on Vulkan (unsupported dtype or layout).");
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
VULKAN_GENERIC_UNARY_GPU(Abs, GenericUnaryShaderOp::Abs)
VULKAN_GENERIC_UNARY_GPU(Ceil, GenericUnaryShaderOp::Ceil)
VULKAN_GENERIC_UNARY_RTE_GPU(Exp, GenericUnaryShaderOp::Exp)
VULKAN_GENERIC_UNARY_GPU(Floor, GenericUnaryShaderOp::Floor)
VULKAN_GENERIC_UNARY_GPU(Negative, GenericUnaryShaderOp::Negative)
VULKAN_GENERIC_UNARY_GPU(Round, GenericUnaryShaderOp::Round)
VULKAN_GENERIC_UNARY_GPU(Sigmoid, GenericUnaryShaderOp::Sigmoid)
VULKAN_GENERIC_UNARY_GPU(Tanh, GenericUnaryShaderOp::Tanh)

// Specialized unary ops
void Cos::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    if (try_eval_unary_op_vulkan<Cos>(
            inputs, out, vulkan::StaticShaderId::cos_f32, stream())) {
      return;
    }
  }
  throw std::runtime_error(
      "Cos operation failed on Vulkan (unsupported dtype or layout).");
}

void Erf::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    if (try_eval_unary_op_vulkan<Erf>(
            inputs, out, vulkan::StaticShaderId::erf_f32, stream())) {
      return;
    }
  }
  throw std::runtime_error(
      "Erf operation failed on Vulkan (unsupported dtype or layout).");
}

void ErfInv::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    if (try_eval_unary_op_vulkan<ErfInv>(
            inputs, out, vulkan::StaticShaderId::erfinv_f32, stream())) {
      return;
    }
  }
  throw std::runtime_error(
      "ErfInv operation failed on Vulkan (unsupported dtype or layout).");
}

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    if (state() == Log::e) {
      auto shader_id = unary_shader_id(UnaryShaderOp::Log, out.dtype());
      if (shader_id.has_value() &&
          try_eval_unary_op_vulkan<Log>(inputs, out, *shader_id, stream())) {
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
    if (try_eval_unary_op_vulkan<Sin>(
            inputs, out, vulkan::StaticShaderId::sin_f32, stream())) {
      return;
    }
  }
  throw std::runtime_error(
      "Sin operation failed on Vulkan (unsupported dtype or layout).");
}

void Square::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto shader_id = unary_shader_id(UnaryShaderOp::Square, out.dtype());
    if (shader_id.has_value()) {
      eval_unary_vulkan<Square>(inputs, out, *shader_id, stream());
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
    auto shader_id = unary_shader_id(
        state() ? UnaryShaderOp::Rsqrt : UnaryShaderOp::Sqrt, out.dtype());
    if (shader_id.has_value()) {
      eval_unary_vulkan<Sqrt>(inputs, out, *shader_id, stream(), 0.0f, 0.0f);
      return;
    }
  }
  throw std::runtime_error(
      "Sqrt operation failed on Vulkan (unsupported dtype or layout).");
}

} // namespace mlx::core
