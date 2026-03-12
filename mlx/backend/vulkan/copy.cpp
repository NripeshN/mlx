// Copyright © 2024 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>

namespace {

using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::Strides;
namespace vulkan = mlx::core::vulkan;

bool has_row_contiguous_strides(const mlx::core::array& arr) {
  if (arr.ndim() == 0) {
    return true;
  }
  int64_t expected = 1;
  for (int i = static_cast<int>(arr.ndim()) - 1; i >= 0; --i) {
    if (arr.shape(i) == 1) {
      continue;
    }
    if (arr.strides(i) != expected) {
      return false;
    }
    expected *= arr.shape(i);
  }
  return true;
}

size_t num_elements(const Shape& shape) {
  return std::accumulate(
      shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
}

int64_t resolve_dynamic_offset(const std::optional<mlx::core::array>& offset) {
  if (!offset.has_value()) {
    return 0;
  }
  auto offset_value = *offset;
  if (offset_value.has_primitive()) {
    offset_value.eval();
  }
  if (offset_value.size() != 1 || offset_value.dtype() != mlx::core::int64) {
    throw std::runtime_error(
        "Dynamic Vulkan copy offsets must be int64 scalars.");
  }
  return offset_value.data<int64_t>()[0];
}

mlx::core::array make_copy_view(
    const mlx::core::array& base,
    const Shape& shape,
    const Strides& strides,
    int64_t elem_offset) {
  mlx::core::array view(shape, base.dtype(), nullptr, {});
  const auto [data_size, row_contiguous, col_contiguous] =
      mlx::core::check_contiguity(shape, strides);
  view.copy_shared_buffer(
      base,
      strides,
      {data_size == num_elements(shape), row_contiguous, col_contiguous},
      data_size,
      elem_offset);
  return view;
}

std::tuple<Shape, Strides, Strides> collapse_copy_dims(
    const Shape& shape,
    const Strides& in_strides,
    const Strides& out_strides) {
  auto [collapsed_shape, collapsed_strides] =
      mlx::core::collapse_contiguous_dims(
          shape,
          {in_strides, out_strides},
          std::numeric_limits<uint32_t>::max());
  return {
      std::move(collapsed_shape),
      std::move(collapsed_strides[0]),
      std::move(collapsed_strides[1])};
}

template <typename SrcT, typename DstT>
void host_cast_copy_vector(const void* src, void* dst, size_t size) {
  auto* src_ptr = static_cast<const SrcT*>(src);
  auto* dst_ptr = static_cast<DstT*>(dst);
  for (size_t i = 0; i < size; ++i) {
    dst_ptr[i] = static_cast<DstT>(src_ptr[i]);
  }
}

template <typename SrcT>
void host_cast_copy_dispatch_dst(
    const void* src,
    void* dst,
    size_t size,
    Dtype dst_dtype) {
  switch (dst_dtype) {
    case mlx::core::bool_:
      host_cast_copy_vector<SrcT, bool>(src, dst, size);
      return;
    case mlx::core::uint8:
      host_cast_copy_vector<SrcT, uint8_t>(src, dst, size);
      return;
    case mlx::core::uint16:
      host_cast_copy_vector<SrcT, uint16_t>(src, dst, size);
      return;
    case mlx::core::uint32:
      host_cast_copy_vector<SrcT, uint32_t>(src, dst, size);
      return;
    case mlx::core::uint64:
      host_cast_copy_vector<SrcT, uint64_t>(src, dst, size);
      return;
    case mlx::core::int8:
      host_cast_copy_vector<SrcT, int8_t>(src, dst, size);
      return;
    case mlx::core::int16:
      host_cast_copy_vector<SrcT, int16_t>(src, dst, size);
      return;
    case mlx::core::int32:
      host_cast_copy_vector<SrcT, int32_t>(src, dst, size);
      return;
    case mlx::core::int64:
      host_cast_copy_vector<SrcT, int64_t>(src, dst, size);
      return;
    case mlx::core::float16:
      host_cast_copy_vector<SrcT, mlx::core::float16_t>(src, dst, size);
      return;
    case mlx::core::float32:
      host_cast_copy_vector<SrcT, float>(src, dst, size);
      return;
    case mlx::core::bfloat16:
      host_cast_copy_vector<SrcT, mlx::core::bfloat16_t>(src, dst, size);
      return;
    case mlx::core::complex64:
      host_cast_copy_vector<SrcT, mlx::core::complex64_t>(src, dst, size);
      return;
    case mlx::core::float64:
      throw std::runtime_error("float64 is not supported on the GPU");
  }
}

void host_cast_copy_dispatch_src(
    const void* src,
    Dtype src_dtype,
    void* dst,
    size_t size,
    Dtype dst_dtype) {
  switch (src_dtype) {
    case mlx::core::bool_:
      host_cast_copy_dispatch_dst<bool>(src, dst, size, dst_dtype);
      return;
    case mlx::core::uint8:
      host_cast_copy_dispatch_dst<uint8_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::uint16:
      host_cast_copy_dispatch_dst<uint16_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::uint32:
      host_cast_copy_dispatch_dst<uint32_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::uint64:
      host_cast_copy_dispatch_dst<uint64_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::int8:
      host_cast_copy_dispatch_dst<int8_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::int16:
      host_cast_copy_dispatch_dst<int16_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::int32:
      host_cast_copy_dispatch_dst<int32_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::int64:
      host_cast_copy_dispatch_dst<int64_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::float16:
      host_cast_copy_dispatch_dst<mlx::core::float16_t>(
          src, dst, size, dst_dtype);
      return;
    case mlx::core::float32:
      host_cast_copy_dispatch_dst<float>(src, dst, size, dst_dtype);
      return;
    case mlx::core::bfloat16:
      host_cast_copy_dispatch_dst<mlx::core::bfloat16_t>(
          src, dst, size, dst_dtype);
      return;
    case mlx::core::complex64:
      host_cast_copy_dispatch_dst<mlx::core::complex64_t>(
          src, dst, size, dst_dtype);
      return;
    case mlx::core::float64:
      throw std::runtime_error("float64 is not supported on the GPU");
  }
}

bool try_host_vector_cast_copy(
    const mlx::core::array& in,
    mlx::core::array& out,
    size_t size,
    int64_t in_offset,
    int64_t out_offset,
    const mlx::core::Stream& s) {
  auto* in_buf = static_cast<mlx::core::vulkan::VulkanBuffer*>(
      const_cast<void*>(static_cast<const void*>(in.buffer().ptr())));
  auto* out_buf =
      static_cast<mlx::core::vulkan::VulkanBuffer*>(out.buffer().ptr());

  auto convert_and_store = [&](const void* src_ptr) {
    std::vector<char> host_out(size * size_of(out.dtype()));
    host_cast_copy_dispatch_src(
        src_ptr, in.dtype(), host_out.data(), size, out.dtype());

    if (mlx::core::vulkan::VulkanContext::get().is_unified_memory() &&
        out_buf->mapped_ptr != nullptr) {
      auto* dst_ptr = static_cast<char*>(out_buf->mapped_ptr) +
          out_offset * size_of(out.dtype());
      std::memcpy(dst_ptr, host_out.data(), host_out.size());
      return;
    }

    mlx::core::vulkan::enqueue_owned_staging_upload(
        s,
        host_out.data(),
        host_out.size(),
        out_buf->buffer,
        out_offset * size_of(out.dtype()));
    mlx::core::vulkan::retain_array_for_stream(s, in);
    mlx::core::vulkan::retain_array_for_stream(s, out);
  };

  if (in_buf->mapped_ptr != nullptr) {
    auto* src_ptr = static_cast<const char*>(in_buf->mapped_ptr) +
        in_offset * size_of(in.dtype());
    convert_and_store(src_ptr);
    return true;
  }

  auto host_in =
      std::make_shared<std::vector<char>>(size * size_of(in.dtype()));
  auto readback_done = std::make_shared<bool>(false);
  mlx::core::vulkan::enqueue_owned_staging_readback(
      s,
      in_buf->buffer,
      in_offset * size_of(in.dtype()),
      host_in->size(),
      [host_in, readback_done](const void* ptr, size_t nbytes) {
        std::memcpy(host_in->data(), ptr, nbytes);
        *readback_done = true;
      });
  mlx::core::vulkan::synchronize_stream(s);
  if (!*readback_done) {
    throw std::runtime_error("Vulkan readback did not complete for cast copy.");
  }
  convert_and_store(host_in->data());
  return true;
}

std::string copy_dtype_suffix(Dtype dtype) {
  switch (dtype) {
    case mlx::core::float16:
      return "f16";
    case mlx::core::float32:
      return "f32";
    case mlx::core::bfloat16:
      return "bf16";
    case mlx::core::bool_:
      return "bool";
    case mlx::core::uint16:
      return "u16";
    case mlx::core::uint8:
      return "u8";
    case mlx::core::int8:
      return "i8";
    case mlx::core::int16:
      return "i16";
    case mlx::core::int32:
      return "i32";
    case mlx::core::uint32:
      return "u32";
    case mlx::core::uint64:
      return "u64";
    case mlx::core::int64:
      return "i64";
    case mlx::core::complex64:
      return "c64";
    default:
      return {};
  }
}

const char* copy_type_name(mlx::core::CopyType ctype) {
  switch (ctype) {
    case mlx::core::CopyType::Scalar:
      return "scalar";
    case mlx::core::CopyType::Vector:
      return "vector";
    case mlx::core::CopyType::General:
      return "general";
    case mlx::core::CopyType::GeneralGeneral:
      return "general_general";
  }
  return "unknown";
}

template <typename Seq>
std::string seq_to_string(const Seq& seq) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < seq.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << seq[i];
  }
  oss << "]";
  return oss.str();
}

bool is_supported_copy_layout(const mlx::core::array& arr) {
  // For contiguous arrays, we can handle any number of dimensions
  // by treating them as 1D
  if (arr.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  const auto item_size = static_cast<int64_t>(size_of(arr.dtype()));
  if (item_size <= 0 || arr.offset() < 0 || (arr.offset() % item_size) != 0) {
    return false;
  }
  const auto elem_offset = arr.offset() / item_size;
  if (elem_offset > 0xFFFF) {
    return false;
  }
  for (auto dim : arr.shape()) {
    if (dim < 0 || dim > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  for (auto stride : arr.strides()) {
    if (stride < 0 || stride > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool trace_copy_dispatch_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_TRACE_COPY_DISPATCH");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

std::optional<vulkan::StaticShaderId> get_copy_shader_id(
    const mlx::core::array& in,
    mlx::core::array& out) {
  // Fast transpose path: source is column-contiguous and destination is
  // row-contiguous with identical shape/dtype.
  if (in.dtype() == out.dtype() && in.shape() == out.shape() &&
      in.ndim() >= 2 && in.ndim() <= 4 && in.offset() == 0 &&
      out.offset() == 0 && in.flags().col_contiguous &&
      out.flags().row_contiguous) {
    const size_t item_size = size_of(in.dtype());
    if (item_size == 2) {
      return vulkan::StaticShaderId::cpy_transpose_16;
    }
    if (item_size == 4) {
      return vulkan::StaticShaderId::cpy_transpose_32;
    }
  }

  // Only use contig_cpy shaders when both arrays have the same size
  // (for simple contiguous copies, not for scatter/concatenate operations)
  if (in.offset() == 0 && out.offset() == 0 && in.flags().row_contiguous &&
      out.flags().row_contiguous && in.size() == out.size()) {
    if (in.dtype() == mlx::core::float32 &&
        out.dtype() == mlx::core::bfloat16) {
      return vulkan::StaticShaderId::contig_cpy_f32_bf16;
    }
    if (in.dtype() == mlx::core::bfloat16 &&
        out.dtype() == mlx::core::float32) {
      return vulkan::StaticShaderId::contig_cpy_bf16_f32;
    }
    if (in.dtype() == mlx::core::bfloat16 &&
        out.dtype() == mlx::core::bfloat16) {
      return vulkan::StaticShaderId::contig_cpy_bf16_bf16;
    }
  }

  const bool supported_pair =
      (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::float32) ||
      (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::float16) ||
      (in.dtype() == mlx::core::float16 && out.dtype() == mlx::core::float16) ||
      (in.dtype() == mlx::core::float16 && out.dtype() == mlx::core::float32) ||
      (in.dtype() == mlx::core::bfloat16 &&
       out.dtype() == mlx::core::float32) ||
      (in.dtype() == mlx::core::bfloat16 &&
       out.dtype() == mlx::core::bfloat16) ||
      (in.dtype() == mlx::core::float32 &&
       out.dtype() == mlx::core::bfloat16) ||
      (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::int32) ||
      (in.dtype() == mlx::core::int8 && out.dtype() == mlx::core::int32) ||
      (in.dtype() == mlx::core::int16 && out.dtype() == mlx::core::int32) ||
      (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int8) ||
      (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int16) ||
      (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::float32) ||
      (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int32) ||
      (in.dtype() == mlx::core::int8 && out.dtype() == mlx::core::int8) ||
      (in.dtype() == mlx::core::int16 && out.dtype() == mlx::core::int16) ||
      (in.dtype() == mlx::core::uint16 && out.dtype() == mlx::core::uint16) ||
      (in.dtype() == mlx::core::uint8 && out.dtype() == mlx::core::uint8) ||
      (in.dtype() == mlx::core::uint8 && out.dtype() == mlx::core::uint32) ||
      (in.dtype() == mlx::core::uint16 && out.dtype() == mlx::core::uint32) ||
      (in.dtype() == mlx::core::bool_ && out.dtype() == mlx::core::bool_) ||
      (in.dtype() == mlx::core::int64 && out.dtype() == mlx::core::int64) ||
      (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint32) ||
      (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint8) ||
      (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint16) ||
      (in.dtype() == mlx::core::uint64 && out.dtype() == mlx::core::uint64) ||
      (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::float32) ||
      (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int64) ||
      (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::complex64);

  if (!supported_pair) {
    return std::nullopt;
  }

  if (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::float32) {
    return vulkan::StaticShaderId::cpy_f32_f32;
  }
  if (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::float16) {
    return vulkan::StaticShaderId::cpy_f32_f16;
  }
  if (in.dtype() == mlx::core::float16 && out.dtype() == mlx::core::float16) {
    return vulkan::StaticShaderId::cpy_f16_f16;
  }
  if (in.dtype() == mlx::core::float16 && out.dtype() == mlx::core::float32) {
    return vulkan::StaticShaderId::cpy_f16_f32;
  }
  if (in.dtype() == mlx::core::bfloat16 && out.dtype() == mlx::core::float32) {
    return vulkan::StaticShaderId::cpy_bf16_f32;
  }
  if (in.dtype() == mlx::core::bfloat16 && out.dtype() == mlx::core::bfloat16) {
    return vulkan::StaticShaderId::cpy_bf16_bf16;
  }
  if (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::bfloat16) {
    return vulkan::StaticShaderId::cpy_f32_bf16;
  }
  if (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::int32) {
    return vulkan::StaticShaderId::cpy_f32_i32;
  }
  if (in.dtype() == mlx::core::int8 && out.dtype() == mlx::core::int32) {
    return vulkan::StaticShaderId::cpy_i8_i32;
  }
  if (in.dtype() == mlx::core::int16 && out.dtype() == mlx::core::int32) {
    return vulkan::StaticShaderId::cpy_i16_i32;
  }
  if (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int8) {
    return vulkan::StaticShaderId::cpy_i32_i8;
  }
  if (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int16) {
    return vulkan::StaticShaderId::cpy_i32_i16;
  }
  if (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::float32) {
    return vulkan::StaticShaderId::cpy_i32_f32;
  }
  if (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int32) {
    return vulkan::StaticShaderId::cpy_i32_i32;
  }
  if (in.dtype() == mlx::core::int8 && out.dtype() == mlx::core::int8) {
    return vulkan::StaticShaderId::cpy_i8_i8;
  }
  if (in.dtype() == mlx::core::int16 && out.dtype() == mlx::core::int16) {
    return vulkan::StaticShaderId::cpy_i16_i16;
  }
  if (in.dtype() == mlx::core::uint16 && out.dtype() == mlx::core::uint16) {
    return vulkan::StaticShaderId::cpy_u16_u16;
  }
  if (in.dtype() == mlx::core::uint8 && out.dtype() == mlx::core::uint8) {
    return vulkan::StaticShaderId::cpy_u8_u8;
  }
  if (in.dtype() == mlx::core::uint8 && out.dtype() == mlx::core::uint32) {
    return vulkan::StaticShaderId::cpy_u8_u32;
  }
  if (in.dtype() == mlx::core::uint16 && out.dtype() == mlx::core::uint32) {
    return vulkan::StaticShaderId::cpy_u16_u32;
  }
  if (in.dtype() == mlx::core::bool_ && out.dtype() == mlx::core::bool_) {
    return vulkan::StaticShaderId::cpy_bool_bool;
  }
  if (in.dtype() == mlx::core::int64 && out.dtype() == mlx::core::int64) {
    return vulkan::StaticShaderId::cpy_i64_i64;
  }
  if (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint32) {
    return vulkan::StaticShaderId::cpy_u32_u32;
  }
  if (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint8) {
    return vulkan::StaticShaderId::cpy_u32_u8;
  }
  if (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint16) {
    return vulkan::StaticShaderId::cpy_u32_u16;
  }
  if (in.dtype() == mlx::core::uint64 && out.dtype() == mlx::core::uint64) {
    return vulkan::StaticShaderId::cpy_u64_u64;
  }
  if (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::float32) {
    return vulkan::StaticShaderId::cpy_u32_f32;
  }
  if (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int64) {
    return vulkan::StaticShaderId::cpy_i32_i64;
  }
  if (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::complex64) {
    return vulkan::StaticShaderId::cpy_f32_c64;
  }

  return std::nullopt;
}

int64_t read_dynamic_index(const mlx::core::array& indices, size_t i) {
  switch (indices.dtype()) {
    case mlx::core::int8:
      return static_cast<int64_t>(indices.data<int8_t>()[i]);
    case mlx::core::uint8:
      return static_cast<int64_t>(indices.data<uint8_t>()[i]);
    case mlx::core::int16:
      return static_cast<int64_t>(indices.data<int16_t>()[i]);
    case mlx::core::uint16:
      return static_cast<int64_t>(indices.data<uint16_t>()[i]);
    case mlx::core::int32:
      return static_cast<int64_t>(indices.data<int32_t>()[i]);
    case mlx::core::uint32:
      return static_cast<int64_t>(indices.data<uint32_t>()[i]);
    case mlx::core::int64:
      return static_cast<int64_t>(indices.data<int64_t>()[i]);
    case mlx::core::uint64:
      return static_cast<int64_t>(indices.data<uint64_t>()[i]);
    default:
      throw std::runtime_error(
          "compute_dynamic_offset requires integer index types.");
  }
}

} // namespace

namespace mlx::core {

void copy_gpu(const array& src, array& out, CopyType ctype, const Stream& s) {
  bool donated = set_copy_output_data(src, out, ctype);
  if (donated && src.dtype() == out.dtype()) {
    // If the output has the same type as the input then there is nothing to
    // copy, just use the buffer.
    return;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_gpu_inplace(src, out, ctype, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    const Stream& s,
    std::optional<array> dynamic_i_offset,
    std::optional<array> dynamic_o_offset) {
  if (out.size() == 0) {
    return;
  }

  const int64_t resolved_i_offset =
      i_offset + resolve_dynamic_offset(dynamic_i_offset);
  const int64_t resolved_o_offset =
      o_offset + resolve_dynamic_offset(dynamic_o_offset);

  auto dispatch_shape = data_shape;
  auto dispatch_i_strides = i_strides;
  auto dispatch_o_strides = o_strides;

  if (dispatch_shape.size() > 4) {
    std::tie(dispatch_shape, dispatch_i_strides, dispatch_o_strides) =
        collapse_copy_dims(
            dispatch_shape, dispatch_i_strides, dispatch_o_strides);
  }

  const auto dispatch_elements = num_elements(dispatch_shape);
  auto in_view =
      make_copy_view(in, dispatch_shape, dispatch_i_strides, resolved_i_offset);
  auto out_view = make_copy_view(
      out, dispatch_shape, dispatch_o_strides, resolved_o_offset);

  const bool same_dtype = in.dtype() == out.dtype();
  const bool raw_buffer_copy = same_dtype && ctype == CopyType::Vector;

  const bool full_tensor_copy = data_shape == in.shape() &&
      data_shape == out.shape() && i_strides == in.strides() &&
      o_strides == out.strides() && resolved_i_offset == 0 &&
      resolved_o_offset == 0;

  const auto shader_id = get_copy_shader_id(in_view, out_view);

  const bool shader_copy_type = ctype == CopyType::General ||
      ctype == CopyType::GeneralGeneral ||
      (ctype == CopyType::Vector && !same_dtype);

  const bool shader_copy = shader_copy_type &&
      is_supported_copy_layout(in_view) && is_supported_copy_layout(out_view) &&
      shader_id.has_value();

  const bool staging_scalar_fill = ctype == CopyType::Scalar &&
      resolved_i_offset == 0 && resolved_o_offset == 0 && full_tensor_copy &&
      (out.flags().contiguous || out.flags().row_contiguous ||
       out.flags().col_contiguous);

  if (staging_scalar_fill) {
    const char* scalar_ptr = static_cast<const char*>(in.data<void>());
    const size_t scalar_size = size_of(in.dtype());
    auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());
    if (vulkan::VulkanContext::get().is_unified_memory()) {
      char* dst = static_cast<char*>(out_buf->mapped_ptr) + out.offset();
      for (size_t offset = 0; offset < out.nbytes(); offset += scalar_size) {
        std::memcpy(dst + offset, scalar_ptr, scalar_size);
      }
    } else {
      std::vector<char> host_fill(out.nbytes());
      for (size_t offset = 0; offset < host_fill.size();
           offset += scalar_size) {
        std::memcpy(host_fill.data() + offset, scalar_ptr, scalar_size);
      }
      vulkan::enqueue_owned_staging_upload(
          s, host_fill.data(), host_fill.size(), out_buf->buffer, out.offset());
      vulkan::retain_array_for_stream(s, in);
      vulkan::retain_array_for_stream(s, out);
    }
    return;
  }

  const bool is_slice_copy =
      shader_copy_type && shader_id.has_value() && in.size() != out.size();

  const bool contiguous_large_rank_copy = same_dtype &&
      dispatch_shape.size() > 4 && in_view.flags().contiguous &&
      out_view.flags().contiguous && in_view.size() == out_view.size() &&
      !is_slice_copy;

  const bool host_contiguous_copy = in_view.flags().row_contiguous &&
      out_view.flags().row_contiguous && dispatch_elements == in_view.size() &&
      dispatch_elements == out_view.size();

  if (!raw_buffer_copy && !shader_copy && !is_slice_copy &&
      !contiguous_large_rank_copy && host_contiguous_copy) {
    if (try_host_vector_cast_copy(
            in_view,
            out_view,
            dispatch_elements,
            in_view.offset(),
            out_view.offset(),
            s)) {
      return;
    }
  }

  if (!raw_buffer_copy && !shader_copy && !is_slice_copy &&
      !contiguous_large_rank_copy) {
    std::ostringstream oss;
    oss << "Copy operation failed on Vulkan (unsupported dtype or layout): "
        << "ctype=" << copy_type_name(ctype) << " "
        << "in_dtype=" << static_cast<int>(in.dtype().val()) << " "
        << "out_dtype=" << static_cast<int>(out.dtype().val()) << " "
        << "in_shape=" << seq_to_string(in_view.shape()) << " "
        << "out_shape=" << seq_to_string(out_view.shape()) << " "
        << "in_strides=" << seq_to_string(in_view.strides()) << " "
        << "out_strides=" << seq_to_string(out_view.strides()) << " "
        << "in_offset=" << in_view.offset() << " "
        << "out_offset=" << out_view.offset() << " "
        << "shader_name="
        << (shader_id.has_value() ? vulkan::static_shader_name(*shader_id)
                                  : "<none>");
    throw std::runtime_error(oss.str());
  }

  VkCommandBuffer cmd_buffer = vulkan::begin_command_recording(s.index);

  // Get buffer handles
  auto* in_buf = static_cast<vulkan::VulkanBuffer*>(
      const_cast<void*>(static_cast<const void*>(in.buffer().ptr())));
  auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());

  if (raw_buffer_copy) {
    // Simple contiguous memory copy using Vulkan command buffer
    VkBufferCopy copy_region{};
    copy_region.srcOffset =
        static_cast<VkDeviceSize>(resolved_i_offset * size_of(in.dtype()));
    copy_region.dstOffset =
        static_cast<VkDeviceSize>(resolved_o_offset * size_of(out.dtype()));
    copy_region.size =
        static_cast<VkDeviceSize>(dispatch_elements * size_of(out.dtype()));

    vkCmdCopyBuffer(
        cmd_buffer, in_buf->buffer, out_buf->buffer, 1, &copy_region);

    vulkan::retain_array_for_stream(s, in);
    vulkan::retain_array_for_stream(s, out);

  } else if (contiguous_large_rank_copy) {
    VkBufferCopy copy_region{};
    copy_region.srcOffset = static_cast<VkDeviceSize>(in_view.offset());
    copy_region.dstOffset = static_cast<VkDeviceSize>(out_view.offset());
    copy_region.size = static_cast<VkDeviceSize>(in_view.nbytes());

    vkCmdCopyBuffer(
        cmd_buffer, in_buf->buffer, out_buf->buffer, 1, &copy_region);

    vulkan::retain_array_for_stream(s, in_view);
    vulkan::retain_array_for_stream(s, out_view);
  } else if (shader_copy || is_slice_copy) {
    if (trace_copy_dispatch_enabled() &&
        (*shader_id == vulkan::StaticShaderId::cpy_bf16_f32 ||
         *shader_id == vulkan::StaticShaderId::cpy_bf16_bf16)) {
      std::cerr << "[vulkan-copy] shader="
                << vulkan::static_shader_name(*shader_id)
                << " ctype=" << copy_type_name(ctype)
                << " in_shape=" << seq_to_string(in_view.shape())
                << " out_shape=" << seq_to_string(out_view.shape())
                << " in_offset=" << in_view.offset()
                << " out_offset=" << out_view.offset()
                << " in_strides=" << seq_to_string(in_view.strides())
                << " out_strides=" << seq_to_string(out_view.strides()) << "\n";
    }
    try {
      if (dispatch_shape.size() > 4) {
        throw std::runtime_error(
            "Copy operation failed on Vulkan: >4D non-contiguous arrays not supported");
      }

      vulkan::dispatch_unary_op(in_view, out_view, *shader_id, cmd_buffer, s);
    } catch (const std::runtime_error& e) {
      vulkan::end_command_recording(s.index);
      throw std::runtime_error(
          std::string("Copy operation failed on Vulkan: ") + e.what());
    }
  } else {
    throw std::runtime_error("Unsupported Vulkan copy type.");
  }

  vulkan::end_command_recording(s.index);
}

// Note: The simpler overload copy_gpu_inplace(in, out, ctype, s) is defined in
// mlx/backend/gpu/copy.cpp and calls the complex version implemented above.

void fill_gpu(const array& val, array& out, const Stream& s) {
  if (out.size() == 0) {
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  // For unified memory, we can directly fill on CPU
  auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());

  if (vulkan::VulkanContext::get().is_unified_memory()) {
    // Direct CPU fill for unified memory
    char* dst_ptr = static_cast<char*>(out_buf->mapped_ptr);
    const char* val_ptr = static_cast<const char*>(val.data<void>());
    size_t val_size = size_of(val.dtype());
    size_t out_size = out.nbytes();

    for (size_t i = 0; i < out_size; i += val_size) {
      std::memcpy(dst_ptr + i, val_ptr, val_size);
    }
  } else {
    // For discrete GPUs, we need to use a compute shader or staging buffer
    // TODO: Implement compute shader fill
    throw std::runtime_error("fill_gpu not yet implemented for discrete GPUs");
  }
}

void reshape_gpu(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu_inplace(
        in,
        out,
        in.shape(),
        in.strides(),
        make_contiguous_strides(in.shape()),
        0,
        0,
        CopyType::General,
        s);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s) {
  std::vector<int> sizes;
  sizes.push_back(0);
  for (const auto& in : inputs) {
    sizes.push_back(in.shape(axis));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  out.set_data(allocator::malloc(out.nbytes()));

  if (axis == 0 && out.flags().row_contiguous) {
    bool supported_axis0_concat = true;
    for (const auto& in : inputs) {
      if (in.dtype() != out.dtype()) {
        supported_axis0_concat = false;
        break;
      }
    }
    if (supported_axis0_concat) {
      auto* out_buf =
          static_cast<mlx::core::vulkan::VulkanBuffer*>(out.buffer().ptr());
      auto command_buffer = vulkan::begin_command_recording(s.index);
      size_t dst_offset = 0;
      for (auto in : inputs) {
        if (!has_row_contiguous_strides(in)) {
          in = contiguous_copy_gpu(in, s);
        }
        auto* in_buf = static_cast<mlx::core::vulkan::VulkanBuffer*>(
            const_cast<void*>(static_cast<const void*>(in.buffer().ptr())));
        VkBufferCopy copy_region{};
        copy_region.srcOffset =
            static_cast<VkDeviceSize>(in.offset() * size_of(in.dtype()));
        copy_region.dstOffset = static_cast<VkDeviceSize>(dst_offset);
        copy_region.size = static_cast<VkDeviceSize>(in.nbytes());
        vkCmdCopyBuffer(
            command_buffer, in_buf->buffer, out_buf->buffer, 1, &copy_region);
        dst_offset += in.nbytes();
      }
      vulkan::end_command_recording(s.index);
      return;
    }
  }

  auto strides = out.strides();
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;

  for (int i = 0; i < inputs.size(); ++i) {
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, flags, out_slice.size(), data_offset);
    const bool vector_copy = has_row_contiguous_strides(inputs[i]) &&
        has_row_contiguous_strides(out_slice) &&
        inputs[i].shape() == out_slice.shape();
    copy_gpu_inplace(
        inputs[i],
        out_slice,
        vector_copy ? CopyType::Vector : CopyType::GeneralGeneral,
        s);
  }
}

array compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes,
    const Stream& s) {
  if (indices.size() != axes.size()) {
    throw std::runtime_error(
        "compute_dynamic_offset expected indices.size() == axes.size().");
  }

  auto indices_eval = indices;
  indices_eval.eval();

  int64_t offset_value = 0;
  for (size_t i = 0; i < axes.size(); ++i) {
    int axis = axes[i];
    if (axis < 0) {
      axis += static_cast<int>(strides.size());
    }
    if (axis < 0 || axis >= static_cast<int>(strides.size())) {
      throw std::out_of_range("compute_dynamic_offset axis out of range.");
    }
    offset_value += read_dynamic_index(indices_eval, i) * strides[axis];
  }

  array offset({1}, int64, nullptr, {});
  offset.set_data(allocator::malloc(offset.itemsize()));
  offset.data<int64_t>()[0] = offset_value;
  return offset;
}

} // namespace mlx::core
