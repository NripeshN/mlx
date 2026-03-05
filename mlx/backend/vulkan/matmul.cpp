// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/primitives.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace mlx::core {

namespace {

constexpr uint32_t kMulMmTileM = 32;
constexpr uint32_t kMulMmTileN = 32;
constexpr uint32_t kMaxGridZ = 65535;

bool is_supported_matmul_dtype(Dtype dtype) {
  return dtype == float32 || dtype == float16 || dtype == bfloat16;
}

bool matmul_debug_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_MATMUL_DEBUG");
    return env != nullptr && std::string(env) == "1";
  }();
  return enabled;
}

bool mul_mm_enabled() {
  static auto& runtime_disabled = []() -> std::atomic<bool>& {
    static std::atomic<bool> disabled{false};
    return disabled;
  }();
  if (runtime_disabled.load(std::memory_order_relaxed)) {
    return false;
  }

  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_ENABLE_MUL_MM");
    if (env == nullptr) {
      return true;
    }
    return std::string(env) != "0";
  }();
  return enabled;
}

void disable_mul_mm_runtime(const std::string& reason) {
  static auto& runtime_disabled = []() -> std::atomic<bool>& {
    static std::atomic<bool> disabled{false};
    return disabled;
  }();

  const bool was_disabled =
      runtime_disabled.exchange(true, std::memory_order_relaxed);
  if (!was_disabled && matmul_debug_enabled()) {
    std::cerr << "[vulkan::mul_mm] disabling mul_mm after failure: " << reason
              << "\n";
  }
}

void log_matmul_path(const std::vector<array>& inputs, const char* path) {
  if (!matmul_debug_enabled() || inputs.size() < 2) {
    return;
  }
  static int printed = 0;
  if (printed >= 32) {
    return;
  }
  ++printed;
  std::cerr << "[vulkan::matmul] path=" << path
            << " a_shape=" << inputs[0].shape()
            << " b_shape=" << inputs[1].shape() << "\n";
}

std::string matvec_shader_name(Dtype matrix_dtype, Dtype vec_dtype) {
  auto matrix_suffix = [&]() -> std::string {
    switch (matrix_dtype) {
      case float32:
        return "f32";
      case float16:
        return "f16";
      case bfloat16:
        return "bf16";
      default:
        return {};
    }
  }();

  auto vec_suffix = [&]() -> std::string {
    switch (vec_dtype) {
      case float32:
        return "f32";
      case float16:
        return "f16";
      default:
        return {};
    }
  }();

  if (matrix_suffix.empty() || vec_suffix.empty()) {
    return {};
  }
  return "mul_mat_vec_" + matrix_suffix + "_" + vec_suffix + "_f32";
}

std::vector<std::string> matvec_shader_candidates(
    Dtype matrix_dtype,
    Dtype vec_dtype) {
  auto base = matvec_shader_name(matrix_dtype, vec_dtype);
  if (base.empty()) {
    return {};
  }
  return {
      base + "_subgroup_no_shmem",
      base + "_subgroup",
      base,
  };
}

std::vector<std::string> mul_mm_shader_candidates(Dtype dtype) {
  std::string base;
  switch (dtype) {
    case float16:
      base = "matmul_f16";
      break;
    case bfloat16:
      base = "matmul_bf16";
      break;
    case float32:
      base = "matmul_f32_f32";
      break;
    default:
      return {};
  }
  return {
      base,
      base + "_fp32",
  };
}

bool is_row_contiguous_zero_offset(const array& arr) {
  return arr.flags().row_contiguous && arr.offset() == 0 &&
      arr.strides(-1) == 1;
}

bool try_eval_matvec_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 2) {
    return false;
  }

  array vec = inputs[0];
  array matrix_t = inputs[1];
  if (vec.ndim() != 2 || matrix_t.ndim() != 2 || out.ndim() != 2) {
    return false;
  }
  if (vec.dtype() != matrix_t.dtype() || out.dtype() != vec.dtype() ||
      !is_supported_matmul_dtype(vec.dtype())) {
    return false;
  }
  if (vec.shape(0) != 1 || out.shape(0) != 1 ||
      vec.shape(1) != matrix_t.shape(0) || out.shape(1) != matrix_t.shape(1)) {
    return false;
  }

  if (!matrix_t.flags().col_contiguous || matrix_t.offset() != 0 ||
      matrix_t.strides(0) != 1) {
    return false;
  }

  if (!is_row_contiguous_zero_offset(vec)) {
    vec = contiguous_copy_gpu(vec, s);
  }
  if (!is_row_contiguous_zero_offset(vec)) {
    return false;
  }

  if (vec.dtype() == bfloat16) {
    array vec_f16(vec.shape(), float16, nullptr, {});
    copy_gpu(vec, vec_f16, CopyType::General, s);
    vec = vec_f16;
  }

  auto shader_candidates =
      matvec_shader_candidates(matrix_t.dtype(), vec.dtype());
  if (shader_candidates.empty()) {
    return false;
  }

  array out_work(out.shape(), float32, nullptr, {});
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (out_work.size() == 0) {
    copy_gpu(out_work, out, CopyType::General, s);
    return true;
  }

  for (const auto& shader_name : shader_candidates) {
    bool dispatched = false;
    try {
      auto command_buffer = vulkan::begin_command_recording(s.index);
      vulkan::dispatch_mul_mat_vec_op(
          matrix_t, vec, out_work, shader_name, command_buffer, s);
      vulkan::end_command_recording(s.index);
      dispatched = true;
    } catch (const std::runtime_error& e) {
      if (matmul_debug_enabled()) {
        std::cerr << "[vulkan::matvec] shader=" << shader_name
                  << " failed: " << e.what() << "\n";
      }
    }
    if (dispatched) {
      copy_gpu(out_work, out, CopyType::General, s);
      return true;
    }
  }
  return false;
}

bool try_eval_mul_mm_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (!mul_mm_enabled()) {
    return false;
  }
  if (inputs.size() != 2) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  if (a.ndim() < 2 || b.ndim() < 2 || out.ndim() < 2) {
    return false;
  }
  if (a.shape(-1) != b.shape(-2) || out.shape(-2) != a.shape(-2) ||
      out.shape(-1) != b.shape(-1)) {
    return false;
  }
  if (!is_supported_matmul_dtype(a.dtype()) ||
      !is_supported_matmul_dtype(b.dtype()) ||
      !is_supported_matmul_dtype(out.dtype())) {
    return false;
  }
  if (a.dtype() == float32 || b.dtype() == float32 || out.dtype() == float32) {
    return false;
  }

  if (a.ndim() != b.ndim() || a.ndim() != out.ndim()) {
    return false;
  }
  for (int i = 0; i < static_cast<int>(a.ndim()) - 2; ++i) {
    if (a.shape(i) != b.shape(i) || a.shape(i) != out.shape(i)) {
      return false;
    }
  }

  if (a.dtype() != b.dtype()) {
    return false;
  }

  // Keep BF16 inputs in BF16 and dispatch matmul_bf16* directly.
  // This matches ggml's BF16xBF16 path and avoids costly staging casts.

  if (!is_row_contiguous_zero_offset(a)) {
    a = contiguous_copy_gpu(a, s);
  }
  if (!is_row_contiguous_zero_offset(a)) {
    return false;
  }

  array b_t = swapaxes_in_eval(b, -1, -2);
  if (!is_row_contiguous_zero_offset(b_t)) {
    b_t = contiguous_copy_gpu(b_t, s);
  }
  if (!is_row_contiguous_zero_offset(b_t)) {
    return false;
  }

  auto shader_candidates = mul_mm_shader_candidates(a.dtype());
  if (shader_candidates.empty()) {
    return false;
  }

  Shape out_t_shape = out.shape();
  std::swap(
      out_t_shape[out_t_shape.size() - 1], out_t_shape[out_t_shape.size() - 2]);
  array out_t(out_t_shape, float32, nullptr, {});
  out_t.set_data(allocator::malloc(out_t.nbytes()));

  const uint32_t m = static_cast<uint32_t>(out.shape(-2));
  const uint32_t n = static_cast<uint32_t>(out.shape(-1));
  const uint32_t k = static_cast<uint32_t>(a.shape(-1));

  const uint32_t batch_stride_a =
      static_cast<uint32_t>(a.shape(-2) * a.shape(-1));
  const uint32_t batch_stride_b =
      static_cast<uint32_t>(b_t.shape(-2) * b_t.shape(-1));
  const uint32_t batch_stride_d =
      static_cast<uint32_t>(out_t.shape(-2) * out_t.shape(-1));

  uint64_t num_batches_u64 = 1;
  for (int i = 0; i < static_cast<int>(out.ndim()) - 2; ++i) {
    num_batches_u64 *= static_cast<uint64_t>(out.shape(i));
  }
  if (num_batches_u64 == 0) {
    copy_gpu(swapaxes_in_eval(out_t, -1, -2), out, CopyType::General, s);
    return true;
  }
  if (num_batches_u64 > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  const uint32_t num_batches = static_cast<uint32_t>(num_batches_u64);

  vulkan::MatmulPushConstants push_constants{};
  push_constants.M = m;
  push_constants.N = n;
  push_constants.K = k;
  push_constants.stride_a = static_cast<uint32_t>(a.strides(-2));
  push_constants.stride_b = static_cast<uint32_t>(b_t.strides(-2));
  push_constants.stride_d = static_cast<uint32_t>(out_t.strides(-2));
  push_constants.batch_stride_a = batch_stride_a;
  push_constants.batch_stride_b = batch_stride_b;
  push_constants.batch_stride_d = batch_stride_d;
  push_constants.num_batches = num_batches;
  push_constants.k_split = k;
  push_constants.ne02 = num_batches;
  push_constants.ne12 = num_batches;
  push_constants.broadcast2 = 1;
  push_constants.broadcast3 = 1;
  push_constants.padded_N = n;

  const uint32_t blocks_m = (m + kMulMmTileM - 1) / kMulMmTileM;
  const uint32_t blocks_n = (n + kMulMmTileN - 1) / kMulMmTileN;

  if (matmul_debug_enabled()) {
    std::cerr << "[vulkan::mul_mm] a_shape=" << a.shape()
              << " a_strides=" << a.strides() << " b_t_shape=" << b_t.shape()
              << " b_t_strides=" << b_t.strides()
              << " out_t_shape=" << out_t.shape()
              << " out_t_strides=" << out_t.strides() << " pc(M,N,K)=" << m
              << "," << n << "," << k
              << " stride(a,b,d)=" << push_constants.stride_a << ","
              << push_constants.stride_b << "," << push_constants.stride_d
              << " batch=" << num_batches << "\n";
  }

  for (const auto& shader_name : shader_candidates) {
    bool dispatched = true;
    bool should_recover_stream = false;
    for (uint32_t base_z = 0; base_z < num_batches; base_z += kMaxGridZ) {
      const uint32_t chunk_z = std::min(kMaxGridZ, num_batches - base_z);
      push_constants.base_work_group_z = base_z;
      const std::array<uint32_t, 3> grid = {blocks_m, blocks_n, chunk_z};

      try {
        auto command_buffer = vulkan::begin_command_recording(s.index);
        vulkan::dispatch_mul_mm_op(
            a,
            b_t,
            out_t,
            shader_name,
            command_buffer,
            s,
            push_constants,
            grid);
        vulkan::end_command_recording(s.index);
      } catch (const std::runtime_error& e) {
        if (matmul_debug_enabled()) {
          std::cerr << "[vulkan::mul_mm] shader=" << shader_name
                    << " failed: " << e.what() << "\n";
        }
        const std::string message = e.what();
        if (message.find("[vulkan::submit_commands]") != std::string::npos ||
            message.find("VkResult=") != std::string::npos) {
          should_recover_stream = true;
        }
        dispatched = false;
      }
      if (!dispatched) {
        break;
      }
    }

    if (dispatched) {
      if (matmul_debug_enabled()) {
        std::cerr << "[vulkan::mul_mm] shader=" << shader_name
                  << " dispatched\n";
      }
      try {
        copy_gpu(swapaxes_in_eval(out_t, -1, -2), out, CopyType::General, s);
        return true;
      } catch (const std::runtime_error& e) {
        if (matmul_debug_enabled()) {
          std::cerr << "[vulkan::mul_mm] output copy failed: " << e.what()
                    << "\n";
        }
        disable_mul_mm_runtime(e.what());
        return false;
      }
    }

    if (should_recover_stream) {
      disable_mul_mm_runtime("submit failure");
      return false;
    }
  }

  return false;
}

} // namespace

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (try_eval_matvec_vulkan(inputs, out, stream())) {
    log_matmul_path(inputs, "mul_mat_vec");
    return;
  }
  if (try_eval_mul_mm_vulkan(inputs, out, stream())) {
    log_matmul_path(inputs, "mul_mm");
    return;
  }
  log_matmul_path(inputs, "cpu_fallback");
  auto cpu_stream = default_stream(Device::cpu);
  Matmul cpu_matmul(cpu_stream);
  cpu_matmul.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [alpha, beta] = state();
  auto cpu_stream = default_stream(Device::cpu);
  AddMM cpu_addmm(cpu_stream, alpha, beta);
  cpu_addmm.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto block_size = state();
  auto cpu_stream = default_stream(Device::cpu);
  BlockMaskedMM cpu_block_masked_mm(cpu_stream, block_size);
  cpu_block_masked_mm.eval_cpu(inputs, out);
  synchronize(cpu_stream);
}

} // namespace mlx::core
