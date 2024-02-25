// Copyright © 2023 Apple Inc.

#include <metal_atomic>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/reduce.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

static constant uint8_t simd_size = 32;

template <typename T, typename Op>
[[kernel]] void init_reduce(
    device T* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::init;
}

#define instantiate_init_reduce(name, otype, op)                            \
  template [[host_name("i" #name)]] [[kernel]] void init_reduce<otype, op>( \
      device otype * out [[buffer(1)]], uint tid [[thread_position_in_grid]]);

///////////////////////////////////////////////////////////////////////////////
// All reduce
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
inline U per_thread_all_reduce(
    const device T* in,
    const device size_t& in_size,
    uint gid,
    uint grid_size) {
  Op op;
  U total_val = Op::init;

  if (gid * N_READS < in_size) {
    in += gid * N_READS;

    int r = 0;
    for (; r < (int)ceildiv(in_size, grid_size * N_READS) - 1; r++) {
      U vals[N_READS] = {op.init};

      for (int i = 0; i < N_READS; i++) {
        vals[i] = static_cast<U>(in[i]);
      }
      for (int i = 0; i < N_READS; i++) {
        total_val = op(vals[i], total_val);
      }

      in += grid_size * N_READS;
    }

    // Separate case for the last set as we close the reduction size
    size_t curr_idx = (gid + r * (size_t)grid_size) * N_READS;
    if (curr_idx < in_size) {
      int max_reads = in_size - curr_idx;
      T vals[N_READS];

      for (int i = 0, idx = 0; i < N_READS; i++, idx++) {
        idx = idx < max_reads ? idx : max_reads - 1;
        vals[i] = in[idx];
      }
      for (int i = 0; i < N_READS; i++) {
        U val = i < max_reads ? vals[i] : Op::init;
        total_val = op(static_cast<U>(val), total_val);
      }
    }
  }

  return total_val;
}

// NB: This kernel assumes threads_per_threadgroup is at most
// 1024. This way with a simd_size of 32, we are guaranteed to
// complete the reduction in two steps of simd-level reductions.
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void all_reduce(
    const device T* in [[buffer(0)]],
    device mlx_atomic<U>* out [[buffer(1)]],
    const device size_t& in_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint grid_size [[threads_per_grid]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U local_vals[simd_size];

  U total_val =
      per_thread_all_reduce<T, U, Op, N_READS>(in, in_size, gid, grid_size);

  // Reduction within simd group
  total_val = op.simd_reduce(total_val);
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }

  // Reduction within thread group
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total_val = lid < simd_per_group ? local_vals[lid] : op.init;
  total_val = op.simd_reduce(total_val);

  // Reduction across threadgroups
  if (lid == 0) {
    op.atomic_update(out, total_val);
  }
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void all_reduce_no_atomics(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const device size_t& in_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint grid_size [[threads_per_grid]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint thread_group_id [[threadgroup_position_in_grid]]) {
  Op op;
  threadgroup U local_vals[simd_size];

  U total_val =
      per_thread_all_reduce<T, U, Op, N_READS>(in, in_size, gid, grid_size);

  // Reduction within simd group (simd_add isn't supported for uint64/int64
  // types)
  for (uint16_t lane_offset = simd_size / 2; lane_offset > 0;
       lane_offset /= 2) {
    total_val = op(total_val, simd_shuffle_down(total_val, lane_offset));
  }
  // Write simd group reduction results to local memory
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduction of simdgroup reduction results within threadgroup.
  total_val = lid < simd_per_group ? local_vals[lid] : op.init;
  for (uint16_t lane_offset = simd_size / 2; lane_offset > 0;
       lane_offset /= 2) {
    total_val = op(total_val, simd_shuffle_down(total_val, lane_offset));
  }

  // Reduction across threadgroups
  if (lid == 0) {
    out[thread_group_id] = total_val;
  }
}

#define instantiate_all_reduce(name, itype, otype, op)        \
  template [[host_name("all_reduce_" #name)]] [[kernel]] void \
  all_reduce<itype, otype, op>(                               \
      const device itype* in [[buffer(0)]],                   \
      device mlx_atomic<otype>* out [[buffer(1)]],            \
      const device size_t& in_size [[buffer(2)]],             \
      uint gid [[thread_position_in_grid]],                   \
      uint lid [[thread_position_in_threadgroup]],            \
      uint grid_size [[threads_per_grid]],                    \
      uint simd_per_group [[simdgroups_per_threadgroup]],     \
      uint simd_lane_id [[thread_index_in_simdgroup]],        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_all_reduce_no_atomics(name, itype, otype, op)        \
  template [[host_name("all_reduce_no_atomics_" #name)]] [[kernel]] void \
  all_reduce_no_atomics<itype, otype, op>(                               \
      const device itype* in [[buffer(0)]],                              \
      device otype* out [[buffer(1)]],                                   \
      const device size_t& in_size [[buffer(2)]],                        \
      uint gid [[thread_position_in_grid]],                              \
      uint lid [[thread_position_in_threadgroup]],                       \
      uint grid_size [[threads_per_grid]],                               \
      uint simd_per_group [[simdgroups_per_threadgroup]],                \
      uint simd_lane_id [[thread_index_in_simdgroup]],                   \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],             \
      uint thread_group_id [[threadgroup_position_in_grid]]);

///////////////////////////////////////////////////////////////////////////////
// Row atomics
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
inline U per_thread_row_reduce(
    const device T* in,
    const constant size_t& reduction_size,
    const constant size_t& out_size,
    const constant int* shape,
    const constant size_t* strides,
    const constant int& ndim,
    uint lsize_x,
    uint lid_x,
    uint2 tid) {
  Op op;

  // Each threadgroup handles 1 reduction
  // TODO: Specializing elem_to_loc would be slightly faster
  int idx = tid.y * out_size + tid.x;
  int extra_offset = elem_to_loc(idx, shape, strides, ndim);
  in += extra_offset + lid_x * N_READS;

  // The reduction is accumulated here
  U total_val = Op::init;

  // Loop over the reduction size within thread group
  int r = 0;
  for (; r < (int)ceildiv(reduction_size, N_READS * lsize_x) - 1; r++) {
    T vals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      vals[i] = in[i];
    }
    for (int i = 0; i < N_READS; i++) {
      total_val = op(static_cast<U>(vals[i]), total_val);
    }

    in += lsize_x * N_READS;
  }

  // Separate case for the last set as we close the reduction size
  size_t reduction_index = (lid_x + (size_t)lsize_x * r) * N_READS;
  if (reduction_index < reduction_size) {
    int max_reads = reduction_size - reduction_index;

    T vals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      int idx = min(i, max_reads - 1);
      vals[i] = static_cast<U>(in[idx]);
    }
    for (int i = 0; i < N_READS; i++) {
      T val = i < max_reads ? vals[i] : Op::init;
      total_val = op(static_cast<U>(val), total_val);
    }
  }

  return total_val;
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_general(
    const device T* in [[buffer(0)]],
    device mlx_atomic<U>* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U local_vals[simd_size];

  U total_val = per_thread_row_reduce<T, U, Op, N_READS>(
      in,
      reduction_size,
      out_size,
      shape,
      strides,
      ndim,
      lsize.x,
      lid.x,
      tid.xy);

  total_val = op.simd_reduce(total_val);

  // Prepare next level
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduction within thread group
  //    Only needed if multiple simd groups
  if (reduction_size > simd_size) {
    total_val = lid.x < simd_per_group ? local_vals[lid.x] : op.init;
    total_val = op.simd_reduce(total_val);
  }
  // Update output
  if (lid.x == 0) {
    op.atomic_update(out, total_val, tid.x);
  }
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_general_no_atomics(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 gsize [[threads_per_grid]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;

  threadgroup U local_vals[simd_size];
  U total_val = per_thread_row_reduce<T, U, Op, N_READS>(
      in,
      reduction_size,
      out_size,
      shape,
      strides,
      ndim,
      lsize.x,
      lid.x,
      tid.xy);

  // Reduction within simd group - simd_add isn't supported for int64 types
  for (uint16_t i = simd_size / 2; i > 0; i /= 2) {
    total_val = op(total_val, simd_shuffle_down(total_val, i));
  }

  // Prepare next level
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduction within thread group
  // Only needed if thread group has multiple simd groups
  if (ceildiv(reduction_size, N_READS) > simd_size) {
    total_val = lid.x < simd_per_group ? local_vals[lid.x] : op.init;
    for (uint16_t i = simd_size / 2; i > 0; i /= 2) {
      total_val = op(total_val, simd_shuffle_down(total_val, i));
    }
  }
  // Write row reduce output for threadgroup with 1st thread in thread group
  if (lid.x == 0) {
    out[(ceildiv(gsize.y, lsize.y) * tid.x) + tid.y] = total_val;
  }
}

#define instantiate_row_reduce_general(name, itype, otype, op)        \
  template [[host_name("row_reduce_general_" #name)]] [[kernel]] void \
  row_reduce_general<itype, otype, op>(                               \
      const device itype* in [[buffer(0)]],                           \
      device mlx_atomic<otype>* out [[buffer(1)]],                    \
      const constant size_t& reduction_size [[buffer(2)]],            \
      const constant size_t& out_size [[buffer(3)]],                  \
      const constant int* shape [[buffer(4)]],                        \
      const constant size_t* strides [[buffer(5)]],                   \
      const constant int& ndim [[buffer(6)]],                         \
      uint3 lid [[thread_position_in_threadgroup]],                   \
      uint3 lsize [[threads_per_threadgroup]],                        \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint simd_lane_id [[thread_index_in_simdgroup]],                \
      uint simd_per_group [[simdgroups_per_threadgroup]],             \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_row_reduce_general_no_atomics(name, itype, otype, op)   \
  template                                                                  \
      [[host_name("row_reduce_general_no_atomics_" #name)]] [[kernel]] void \
      row_reduce_general_no_atomics<itype, otype, op>(                      \
          const device itype* in [[buffer(0)]],                             \
          device otype* out [[buffer(1)]],                                  \
          const constant size_t& reduction_size [[buffer(2)]],              \
          const constant size_t& out_size [[buffer(3)]],                    \
          const constant int* shape [[buffer(4)]],                          \
          const constant size_t* strides [[buffer(5)]],                     \
          const constant int& ndim [[buffer(6)]],                           \
          uint3 lid [[thread_position_in_threadgroup]],                     \
          uint3 lsize [[threads_per_threadgroup]],                          \
          uint3 gsize [[threads_per_grid]],                                 \
          uint3 tid [[threadgroup_position_in_grid]],                       \
          uint simd_lane_id [[thread_index_in_simdgroup]],                  \
          uint simd_per_group [[simdgroups_per_threadgroup]],               \
          uint simd_group_id [[simdgroup_index_in_threadgroup]]);

///////////////////////////////////////////////////////////////////////////////
// Column reduce
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
inline U _contiguous_strided_reduce(
    const device T* in,
    threadgroup U* local_data,
    uint in_idx,
    uint reduction_size,
    uint reduction_stride,
    uint2 tid,
    uint2 lid,
    uint2 lsize) {
  Op op;
  U total_val = Op::init;

  uint base_offset = (tid.y * lsize.y + lid.y) * N_READS;
  for (uint r = 0; r < N_READS && (base_offset + r) < reduction_size; r++) {
    uint offset = base_offset + r;
    total_val =
        op(static_cast<U>(total_val), in[in_idx + offset * reduction_stride]);
  }
  local_data[lsize.y * lid.x + lid.y] = total_val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  U val = Op::init;
  if (lid.y == 0) {
    // Perform reduction across columns in thread group
    for (uint i = 0; i < lsize.y; i++) {
      val = op(val, local_data[lsize.y * lid.x + i]);
    }
  }

  return val;
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void col_reduce_general(
    const device T* in [[buffer(0)]],
    device mlx_atomic<U>* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    threadgroup U* local_data [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]) {
  auto out_idx = tid.x * lsize.x + lid.x;
  auto in_idx = elem_to_loc(out_idx + tid.z * out_size, shape, strides, ndim);

  Op op;
  if (out_idx < out_size) {
    U val = _contiguous_strided_reduce<T, U, Op, N_READS>(
        in,
        local_data,
        in_idx,
        reduction_size,
        reduction_stride,
        tid.xy,
        lid.xy,
        lsize.xy);

    // Write out reduction results generated by threadgroups working on specific
    // output element, contiguously.
    if (lid.y == 0) {
      op.atomic_update(out, val, out_idx);
    }
  }
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void col_reduce_general_no_atomics(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    threadgroup U* local_data [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 gsize [[threads_per_grid]]) {
  auto out_idx = tid.x * lsize.x + lid.x;
  auto in_idx = elem_to_loc(out_idx + tid.z * out_size, shape, strides, ndim);

  if (out_idx < out_size) {
    U val = _contiguous_strided_reduce<T, U, Op, N_READS>(
        in,
        local_data,
        in_idx,
        reduction_size,
        reduction_stride,
        tid.xy,
        lid.xy,
        lsize.xy);

    // Write out reduction results generated by threadgroups working on specific
    // output element, contiguously.
    if (lid.y == 0) {
      uint tgsize_y = ceildiv(gsize.y, lsize.y);
      uint tgsize_z = ceildiv(gsize.z, lsize.z);
      out[tgsize_y * tgsize_z * gid.x + tgsize_y * tid.z + tid.y] = val;
    }
  }
}

#define instantiate_col_reduce_general(name, itype, otype, op)        \
  template [[host_name("col_reduce_general_" #name)]] [[kernel]] void \
  col_reduce_general<itype, otype, op>(                               \
      const device itype* in [[buffer(0)]],                           \
      device mlx_atomic<otype>* out [[buffer(1)]],                    \
      const constant size_t& reduction_size [[buffer(2)]],            \
      const constant size_t& reduction_stride [[buffer(3)]],          \
      const constant size_t& out_size [[buffer(4)]],                  \
      const constant int* shape [[buffer(5)]],                        \
      const constant size_t* strides [[buffer(6)]],                   \
      const constant int& ndim [[buffer(7)]],                         \
      threadgroup otype* local_data [[threadgroup(0)]],               \
      uint3 tid [[threadgroup_position_in_grid]],                     \
      uint3 lid [[thread_position_in_threadgroup]],                   \
      uint3 lsize [[threads_per_threadgroup]]);

#define instantiate_col_reduce_general_no_atomics(name, itype, otype, op)   \
  template                                                                  \
      [[host_name("col_reduce_general_no_atomics_" #name)]] [[kernel]] void \
      col_reduce_general_no_atomics<itype, otype, op>(                      \
          const device itype* in [[buffer(0)]],                             \
          device otype* out [[buffer(1)]],                                  \
          const constant size_t& reduction_size [[buffer(2)]],              \
          const constant size_t& reduction_stride [[buffer(3)]],            \
          const constant size_t& out_size [[buffer(4)]],                    \
          const constant int* shape [[buffer(5)]],                          \
          const constant size_t* strides [[buffer(6)]],                     \
          const constant int& ndim [[buffer(7)]],                           \
          threadgroup otype* local_data [[threadgroup(0)]],                 \
          uint3 tid [[threadgroup_position_in_grid]],                       \
          uint3 lid [[thread_position_in_threadgroup]],                     \
          uint3 gid [[thread_position_in_grid]],                            \
          uint3 lsize [[threads_per_threadgroup]],                          \
          uint3 gsize [[threads_per_grid]]);

///////////////////////////////////////////////////////////////////////////////
// Instantiations
///////////////////////////////////////////////////////////////////////////////

// clang-format off
#define instantiate_reduce(name, itype, otype, op)       \
  instantiate_all_reduce(name, itype, otype, op)         \
  instantiate_row_reduce_general(name, itype, otype, op) \
  instantiate_col_reduce_general(name, itype, otype, op) // clang-format on

// clang-format off
#define instantiate_reduce_no_atomics(name, itype, otype, op)       \
  instantiate_all_reduce_no_atomics(name, itype, otype, op)         \
  instantiate_row_reduce_general_no_atomics(name, itype, otype, op) \
  instantiate_col_reduce_general_no_atomics(name, itype, otype, op) // clang-format on

// clang-format off
#define instantiate_same_reduce_no_atomics(name, tname, type, op) \
  instantiate_init_reduce(name ##tname, type, op<type>)           \
  instantiate_reduce_no_atomics(name ##tname, type, type, op<type>) // clang-format on

// clang-format off
#define instantiate_same_reduce(name, tname, type, op)  \
  instantiate_init_reduce(name ##tname, type, op<type>) \
  instantiate_reduce(name ##tname, type, type, op<type>) // clang-format on

#define instantiate_reduce_from_types_helper(name, tname, itype, otype, op) \
  instantiate_reduce(name##tname, itype, otype, op)

// clang-format off
#define instantiate_reduce_from_types(name, otype, op)                    \
  instantiate_reduce_from_types_helper(name, bool_, bool, otype, op)      \
  instantiate_reduce_from_types_helper(name, uint8, uint8_t, otype, op)   \
  instantiate_reduce_from_types_helper(name, uint16, uint16_t, otype, op) \
  instantiate_reduce_from_types_helper(name, uint32, uint32_t, otype, op) \
  instantiate_reduce_from_types_helper(name, int8, int8_t, otype, op)     \
  instantiate_reduce_from_types_helper(name, int16, int16_t, otype, op)   \
  instantiate_reduce_from_types_helper(name, int32, int32_t, otype, op)   \
  instantiate_reduce_from_types_helper(name, int64, int64_t, otype, op)   \
  instantiate_reduce_from_types_helper(name, float16, half, otype, op)    \
  instantiate_reduce_from_types_helper(name, float32, float, otype, op)   \
  instantiate_reduce_from_types_helper(name, bfloat16, bfloat16_t, otype, op)

// special case bool with larger output type
instantiate_reduce(sumbool_, bool, uint32_t, Sum<uint32_t>)
instantiate_same_reduce(sum, uint8, uint8_t, Sum)
instantiate_same_reduce(sum, uint16, uint16_t, Sum)
instantiate_same_reduce(sum, uint32, uint32_t, Sum)
instantiate_same_reduce(sum, int8, int8_t, Sum)
instantiate_same_reduce(sum, int16, int16_t, Sum)
instantiate_same_reduce(sum, int32, int32_t, Sum)
instantiate_same_reduce(sum, float16, half, Sum)
instantiate_same_reduce(sum, float32, float, Sum)

instantiate_same_reduce_no_atomics(sum, int64, int64_t, Sum)
instantiate_same_reduce_no_atomics(sum, uint64, uint64_t, Sum)

instantiate_same_reduce(prod, uint8, uint8_t, Prod)
instantiate_same_reduce(prod, uint16, uint16_t, Prod)
instantiate_same_reduce(prod, uint32, uint32_t, Prod)
instantiate_same_reduce(prod, int8, int8_t, Prod)
instantiate_same_reduce(prod, int16, int16_t, Prod)
instantiate_same_reduce(prod, int32, int32_t, Prod)
instantiate_same_reduce(prod, float16, half, Prod)
instantiate_same_reduce(prod, float32, float, Prod)

instantiate_same_reduce_no_atomics(prod, int64, int64_t, Prod)
instantiate_same_reduce_no_atomics(prod, uint64, uint64_t, Prod)

instantiate_same_reduce(sum, bfloat16, bfloat16_t, Sum)
instantiate_same_reduce(prod, bfloat16, bfloat16_t, Prod)

instantiate_init_reduce(andbool_, bool, And)
instantiate_reduce_from_types(and, bool, And)

instantiate_init_reduce(orbool_, bool, Or)
instantiate_reduce_from_types(or, bool, Or)

// Compiler segfaulted with the names "min" or "max" ...
instantiate_same_reduce(min_, uint8, uint8_t, Min)
instantiate_same_reduce(min_, uint16, uint16_t, Min)
instantiate_same_reduce(min_, uint32, uint32_t, Min)
instantiate_same_reduce(min_, int8, int8_t, Min)
instantiate_same_reduce(min_, int16, int16_t, Min)
instantiate_same_reduce(min_, int32, int32_t, Min)
instantiate_same_reduce(min_, float16, half, Min)
instantiate_same_reduce(min_, float32, float, Min)

instantiate_same_reduce_no_atomics(min_, int64, int64_t, Min)
instantiate_same_reduce_no_atomics(min_, uint64, uint64_t, Min)

instantiate_same_reduce(max_, uint8, uint8_t, Max)
instantiate_same_reduce(max_, uint16, uint16_t, Max)
instantiate_same_reduce(max_, uint32, uint32_t, Max)
instantiate_same_reduce(max_, int8, int8_t, Max)
instantiate_same_reduce(max_, int16, int16_t, Max)
instantiate_same_reduce(max_, int32, int32_t, Max)
instantiate_same_reduce(max_, float16, half, Max)
instantiate_same_reduce(max_, float32, float, Max)

instantiate_same_reduce_no_atomics(max_, int64, int64_t, Max)
instantiate_same_reduce_no_atomics(max_, uint64, uint64_t, Max)

instantiate_same_reduce(min_, bfloat16, bfloat16_t, Min)
instantiate_same_reduce(max_, bfloat16, bfloat16_t, Max) // clang-format on
