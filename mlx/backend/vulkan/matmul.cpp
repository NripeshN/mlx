// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/matmul.h"
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
constexpr char kMatvecVectorCastScratchLane[] = "matvec.vec_f16";
constexpr char kMatvecOutScratchLane[] = "matvec.out_work";
constexpr char kMulMmOutScratchLane[] = "mul_mm.out_t";

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

bool matvec_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_ENABLE_MATVEC");
    if (env == nullptr) {
      return false;
    }
    return std::string(env) != "0";
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

bool mul_mm_batch_sync_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_MUL_MM_BATCH_SYNC");
    if (env == nullptr) {
      return false;
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

std::optional<vulkan::StaticShaderId> matvec_shader_name(
    Dtype matrix_dtype,
    Dtype vec_dtype) {
  if (matrix_dtype == float32 && vec_dtype == float32) {
    return vulkan::StaticShaderId::mul_mat_vec_f32_f32_f32;
  }
  if (matrix_dtype == float16 && vec_dtype == float32) {
    return vulkan::StaticShaderId::mul_mat_vec_f16_f32_f32;
  }
  if (matrix_dtype == bfloat16 && vec_dtype == float32) {
    return vulkan::StaticShaderId::mul_mat_vec_bf16_f32_f32;
  }
  if (matrix_dtype == float32 && vec_dtype == float16) {
    return vulkan::StaticShaderId::mul_mat_vec_f32_f16_f32;
  }
  if (matrix_dtype == float16 && vec_dtype == float16) {
    return vulkan::StaticShaderId::mul_mat_vec_f16_f16_f32;
  }
  if (matrix_dtype == bfloat16 && vec_dtype == float16) {
    return vulkan::StaticShaderId::mul_mat_vec_bf16_f16_f32;
  }
  return std::nullopt;
}

std::vector<vulkan::StaticShaderId> matvec_shader_candidates(
    Dtype matrix_dtype,
    Dtype vec_dtype) {
  auto base = matvec_shader_name(matrix_dtype, vec_dtype);
  if (!base.has_value()) {
    return {};
  }
  switch (*base) {
    case vulkan::StaticShaderId::mul_mat_vec_f32_f32_f32:
      return {
          vulkan::StaticShaderId::mul_mat_vec_f32_f32_f32,
          vulkan::StaticShaderId::mul_mat_vec_f32_f32_f32_subgroup,
          vulkan::StaticShaderId::mul_mat_vec_f32_f32_f32_subgroup_no_shmem,
      };
    case vulkan::StaticShaderId::mul_mat_vec_f16_f32_f32:
      return {
          vulkan::StaticShaderId::mul_mat_vec_f16_f32_f32,
          vulkan::StaticShaderId::mul_mat_vec_f16_f32_f32_subgroup,
          vulkan::StaticShaderId::mul_mat_vec_f16_f32_f32_subgroup_no_shmem,
      };
    case vulkan::StaticShaderId::mul_mat_vec_bf16_f32_f32:
      return {
          vulkan::StaticShaderId::mul_mat_vec_bf16_f32_f32,
          vulkan::StaticShaderId::mul_mat_vec_bf16_f32_f32_subgroup,
          vulkan::StaticShaderId::mul_mat_vec_bf16_f32_f32_subgroup_no_shmem,
      };
    case vulkan::StaticShaderId::mul_mat_vec_f32_f16_f32:
      return {
          vulkan::StaticShaderId::mul_mat_vec_f32_f16_f32,
          vulkan::StaticShaderId::mul_mat_vec_f32_f16_f32_subgroup,
          vulkan::StaticShaderId::mul_mat_vec_f32_f16_f32_subgroup_no_shmem,
      };
    case vulkan::StaticShaderId::mul_mat_vec_f16_f16_f32:
      return {
          vulkan::StaticShaderId::mul_mat_vec_f16_f16_f32,
          vulkan::StaticShaderId::mul_mat_vec_f16_f16_f32_subgroup,
          vulkan::StaticShaderId::mul_mat_vec_f16_f16_f32_subgroup_no_shmem,
      };
    case vulkan::StaticShaderId::mul_mat_vec_bf16_f16_f32:
      return {
          vulkan::StaticShaderId::mul_mat_vec_bf16_f16_f32,
          vulkan::StaticShaderId::mul_mat_vec_bf16_f16_f32_subgroup,
          vulkan::StaticShaderId::mul_mat_vec_bf16_f16_f32_subgroup_no_shmem,
      };
    case vulkan::StaticShaderId::Count:
      break;
  }
  return {};
}

std::vector<vulkan::StaticShaderId> mul_mm_shader_candidates(Dtype dtype) {
  switch (dtype) {
    case float16:
      return {
          vulkan::StaticShaderId::matmul_f16,
          vulkan::StaticShaderId::matmul_f16_fp32,
      };
    case bfloat16:
      return {
          vulkan::StaticShaderId::matmul_bf16,
          vulkan::StaticShaderId::matmul_bf16_fp32,
      };
    case float32:
      return {
          vulkan::StaticShaderId::matmul_f32_f32_fp32,
          vulkan::StaticShaderId::matmul_f32_f32,
      };
    default:
      return {};
  }
}

bool is_row_contiguous_zero_offset(const array& arr) {
  return arr.flags().row_contiguous && arr.offset() == 0 &&
      arr.strides(-1) == 1;
}

bool has_vulkan_buffer(const array& arr) {
  auto data = arr.data_shared_ptr();
  return data != nullptr && data->buffer.ptr() != nullptr;
}

bool ensure_vulkan_buffer(array& arr, Stream s) {
  if (has_vulkan_buffer(arr)) {
    return true;
  }

  if (arr.has_primitive()) {
    arr.eval();
  } else {
    arr.wait();
  }

  if (has_vulkan_buffer(arr)) {
    return true;
  }

  auto data = arr.data_shared_ptr();
  if (data == nullptr || data->buffer.ptr() == nullptr) {
    return false;
  }

  arr = contiguous_copy_gpu(arr, s);
  return has_vulkan_buffer(arr);
}

void zero_initialize_output(array& out, Stream s) {
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.nbytes() == 0) {
    return;
  }
  auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());
  auto cmd_buffer = vulkan::begin_command_recording(s.index);
  cmd_buffer.fillBuffer(out_buf->buffer, 0, out.nbytes(), 0);
  vulkan::end_command_recording(s.index);
}

bool try_eval_matvec_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 2) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  if (a.ndim() != 2 || b.ndim() != 2 || out.ndim() != 2) {
    return false;
  }
  if (a.dtype() != b.dtype() || out.dtype() != a.dtype() ||
      !is_supported_matmul_dtype(a.dtype())) {
    return false;
  }

  bool a_is_vec = (a.shape(0) == 1);
  bool b_is_vec = (b.shape(1) == 1);
  bool is_matvec = a_is_vec || b_is_vec;

  if (!is_matvec) {
    return false;
  }

  array vec = a;
  array matrix = b;
  if (a_is_vec && b_is_vec) {
    if (a.shape(1) != b.shape(0)) {
      return false;
    }
    vec = a;
    matrix = b;
  } else if (a_is_vec) {
    if (a.shape(1) != b.shape(0)) {
      return false;
    }
    vec = a;
    matrix = b;
  } else {
    if (b.shape(1) != a.shape(0)) {
      return false;
    }
    vec = b;
    matrix = a;
  }

  if (out.shape(0) != vec.shape(0) || out.shape(1) != matrix.shape(1)) {
    return false;
  }

  if (vec.shape(1) == 0) {
    zero_initialize_output(out, s);
    return true;
  }

  if (!is_row_contiguous_zero_offset(vec)) {
    vec = contiguous_copy_gpu(vec, s);
  }
  if (!is_row_contiguous_zero_offset(vec)) {
    return false;
  }

  if (!is_row_contiguous_zero_offset(matrix)) {
    matrix = contiguous_copy_gpu(matrix, s);
  }
  if (!is_row_contiguous_zero_offset(matrix)) {
    return false;
  }

  Dtype vec_shader_dtype = vec.dtype();
  if (vec_shader_dtype == bfloat16) {
    array vec_f16 = vulkan::acquire_scratch_array(
        s, kMatvecVectorCastScratchLane, vec.shape(), float16);
    copy_gpu(vec, vec_f16, CopyType::General, s);
    vulkan::mark_scratch_array_written(s, kMatvecVectorCastScratchLane);
    vec = vec_f16;
    vec_shader_dtype = float16;
  }

  auto shader_candidates =
      matvec_shader_candidates(matrix.dtype(), vec_shader_dtype);
  if (shader_candidates.empty()) {
    return false;
  }

  array out_work = vulkan::acquire_scratch_array(
      s, kMatvecOutScratchLane, out.shape(), float32);
  if (out_work.size() == 0) {
    copy_gpu(out_work, out, CopyType::General, s);
    return true;
  }

  for (const auto shader_id : shader_candidates) {
    bool dispatched = false;
    try {
      auto command_buffer = vulkan::begin_command_recording(s.index);
      vulkan::dispatch_mul_mat_vec_op(
          matrix, vec, out_work, shader_id, command_buffer, s);
      vulkan::end_command_recording(s.index);
      dispatched = true;
    } catch (const std::runtime_error& e) {
      if (matmul_debug_enabled()) {
        std::cerr << "[vulkan::matvec] shader="
                  << vulkan::static_shader_name(shader_id)
                  << " failed: " << e.what() << "\n";
      }
    }
    if (dispatched) {
      vulkan::mark_scratch_array_written(s, kMatvecOutScratchLane);
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

  if (a.shape(-1) == 0) {
    zero_initialize_output(out, s);
    return true;
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

  Shape out_t_shape = out.shape();
  std::swap(
      out_t_shape[out_t_shape.size() - 1], out_t_shape[out_t_shape.size() - 2]);
  array out_t = vulkan::acquire_scratch_array(
      s, kMulMmOutScratchLane, out_t_shape, float32);

  if (!ensure_vulkan_buffer(a, s) || !ensure_vulkan_buffer(b_t, s) ||
      !ensure_vulkan_buffer(out_t, s)) {
    if (matmul_debug_enabled()) {
      std::cerr << "[vulkan::mul_mm] missing buffer"
                << " a=" << has_vulkan_buffer(a) << " a_status=" << a.status()
                << " a_has_primitive=" << a.has_primitive()
                << " b_t=" << has_vulkan_buffer(b_t)
                << " b_t_status=" << b_t.status()
                << " b_t_has_primitive=" << b_t.has_primitive()
                << " out_t=" << has_vulkan_buffer(out_t) << "\n";
    }
    return false;
  }

  auto shader_candidates = mul_mm_shader_candidates(a.dtype());
  if (shader_candidates.empty()) {
    return false;
  }

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

  if (num_batches > 1) {
    ::mlx::core::gpu::synchronize(s);
  }

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

  for (const auto shader_id : shader_candidates) {
    bool dispatched = true;
    bool should_recover_stream = false;
    for (uint32_t base_z = 0; base_z < num_batches; base_z += kMaxGridZ) {
      const uint32_t chunk_z = std::min(kMaxGridZ, num_batches - base_z);
      push_constants.base_work_group_z = base_z;
      const std::array<uint32_t, 3> grid = {blocks_m, blocks_n, chunk_z};

      try {
        auto command_buffer = vulkan::begin_command_recording(s.index);
        vulkan::dispatch_mul_mm_op(
            a, b_t, out_t, shader_id, command_buffer, s, push_constants, grid);
        vulkan::end_command_recording(s.index);
      } catch (const std::runtime_error& e) {
        if (matmul_debug_enabled()) {
          std::cerr << "[vulkan::mul_mm] shader="
                    << vulkan::static_shader_name(shader_id)
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
        std::cerr << "[vulkan::mul_mm] shader="
                  << vulkan::static_shader_name(shader_id) << " dispatched\n";
      }
      vulkan::mark_scratch_array_written(s, kMulMmOutScratchLane);
      try {
        if (matmul_debug_enabled()) {
          std::cerr << "[vulkan::mul_mm] begin output copy\n";
        }
        copy_gpu(swapaxes_in_eval(out_t, -1, -2), out, CopyType::General, s);
        if (matmul_debug_enabled()) {
          std::cerr << "[vulkan::mul_mm] end output copy\n";
        }
        if (num_batches > 1) {
          ::mlx::core::gpu::synchronize(s);
          if (matmul_debug_enabled()) {
            std::cerr << "[vulkan::mul_mm] post-copy synchronize done\n";
          }
        }
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

bool try_eval_matmul_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() == 2 && (inputs[0].size() == 0 || inputs[1].size() == 0)) {
    zero_initialize_output(out, s);
    return true;
  }
  if (matvec_enabled() && try_eval_matvec_vulkan(inputs, out, s)) {
    return true;
  }
  return try_eval_mul_mm_vulkan(inputs, out, s);
}

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (try_eval_matmul_vulkan(inputs, out, stream())) {
    log_matmul_path(inputs, "mul_mm");
    return;
  }
  throw std::runtime_error(
      "Matmul operation failed on Vulkan (unsupported dtype or layout).");
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("AddMM has no Vulkan implementation.");
}

void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("BlockMaskedMM has no Vulkan implementation.");
}

} // namespace mlx::core
