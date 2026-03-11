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
#include <numeric>
#include <string>

namespace {

using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::Strides;

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

std::string copy_dtype_suffix(Dtype dtype) {
  switch (dtype) {
    case mlx::core::float16:
      return "f16";
    case mlx::core::float32:
      return "f32";
    case mlx::core::bfloat16:
      return "bf16";
    case mlx::core::int32:
      return "i32";
    case mlx::core::uint32:
      return "u32";
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
  if (arr.offset() < 0 || arr.offset() > 0xFFFF) {
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

std::string get_copy_shader_name(
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
      return "cpy_transpose_16";
    }
    if (item_size == 4) {
      return "cpy_transpose_32";
    }
  }

  // Only use contig_cpy shaders when both arrays have the same size
  // (for simple contiguous copies, not for scatter/concatenate operations)
  if (in.offset() == 0 && out.offset() == 0 && in.flags().row_contiguous &&
      out.flags().row_contiguous && in.size() == out.size()) {
    if (in.dtype() == mlx::core::float32 &&
        out.dtype() == mlx::core::bfloat16) {
      return "contig_cpy_f32_bf16";
    }
    if (in.dtype() == mlx::core::bfloat16 &&
        out.dtype() == mlx::core::float32) {
      return "contig_cpy_bf16_f32";
    }
    if (in.dtype() == mlx::core::bfloat16 &&
        out.dtype() == mlx::core::bfloat16) {
      return "contig_cpy_bf16_bf16";
    }
  }

  auto src_suffix = copy_dtype_suffix(in.dtype());
  auto dst_suffix = copy_dtype_suffix(out.dtype());
  if (src_suffix.empty() || dst_suffix.empty()) {
    return {};
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
      (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::float32) ||
      (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int32) ||
      (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint32);

  if (!supported_pair) {
    return {};
  }

  return "cpy_" + src_suffix + "_" + dst_suffix;
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

  const auto shader_name = get_copy_shader_name(in_view, out_view);

  const bool shader_copy_type =
      ctype == CopyType::General || ctype == CopyType::GeneralGeneral;

  const bool shader_copy = shader_copy_type &&
      is_supported_copy_layout(in_view) && is_supported_copy_layout(out_view) &&
      !shader_name.empty();

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
      shader_copy_type && !shader_name.empty() && in.size() != out.size();

  const bool contiguous_large_rank_copy = same_dtype &&
      dispatch_shape.size() > 4 && in_view.flags().contiguous &&
      out_view.flags().contiguous && in_view.size() == out_view.size() &&
      !is_slice_copy;

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
        << "shader_name=" << (shader_name.empty() ? "<none>" : shader_name);
    throw std::runtime_error(oss.str());
  }

  VkCommandBuffer cmd_buffer = vulkan::begin_command_recording(s.index);

  // Get buffer handles
  auto* in_buf = static_cast<vulkan::VulkanBuffer*>(
      const_cast<void*>(static_cast<const void*>(in.buffer().ptr())));
  auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());

  if (ctype == CopyType::Vector) {
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
        (shader_name == "cpy_bf16_f32" || shader_name == "cpy_bf16_bf16")) {
      std::cerr << "[vulkan-copy] shader=" << shader_name
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

      vulkan::dispatch_unary_op(in_view, out_view, shader_name, cmd_buffer, s);
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
    copy_gpu_inplace(inputs[i], out_slice, CopyType::GeneralGeneral, s);
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
