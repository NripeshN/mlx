// Copyright © 2025 Apple Inc.
//
// ROCm allocator — mirrors mlx/backend/cuda/allocator.h exactly.
// Device free/alloc: BufferCache recycle, hipMallocAsync / hipFreeAsync
// (or hipMalloc / hipFree when no stream/pool). Only ROCm-specific extras
// are host_shadow (discrete GPU CPU access) and optional decode arena /
// graph deferred free.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"

#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>
#include <vector>

namespace mlx::core::rocm {

using allocator::Buffer;

// Matches CudaBuffer + fields required for discrete-GPU host access and
// stream-ordered free (CUDA uses move_to_unified_memory instead of shadow).
// Field order for the first 7 members is ABI-stable for brace-init sites
// outside this file (e.g. gemms/naive_gemm.hip): data, size, is_managed,
// device, host_shadow, host_dirty, alloc_stream. New fields append only.
struct RocmBuffer {
  void* data;
  size_t size;
  bool is_managed;
  int device; // -1 for unified / non-pool
  void* host_shadow;
  bool host_dirty;
  void* alloc_stream; // stream used for hipMallocAsync / hipFreeAsync
  // Allocation generation for phase-scoped freelist drop (prefill → decode).
  // Default 0 keeps brace-inits that omit it safe.
  uint32_t generation{0};
};

// ---------------------------------------------------------------------------
// SmallSizePool — identical to CUDA (8-byte scalar freelist only).
// ---------------------------------------------------------------------------

class SmallSizePool {
 private:
  union Block {
    Block* next;
    RocmBuffer buf;
  };

  Block* buffer_{nullptr};
  void* data_{nullptr};
  bool data_managed_{true};
  Block* next_free_{nullptr};

 public:
  SmallSizePool();
  ~SmallSizePool();

  SmallSizePool(const SmallSizePool&) = delete;
  SmallSizePool& operator=(const SmallSizePool&) = delete;

  RocmBuffer* malloc();
  void free(RocmBuffer* buf);
  bool in_pool(RocmBuffer* buf);
};

// ---------------------------------------------------------------------------
// DecodeArena — optional bump allocator for HIP-graph decode (not in CUDA).
// Training does not use this; only the graph-decode engine does.
// ---------------------------------------------------------------------------
struct DecodeArena {
  void* base{nullptr};
  size_t capacity{0};
  size_t offset{0};
  bool active{false};
  bool overflowed{false};
  int device{-1};
  std::deque<RocmBuffer> wrappers;
  size_t next_wrapper{0};
  size_t high_water{0};
  size_t floor_offset{0};
  size_t floor_wrapper{0};

  bool contains(const void* p) const {
    return base && p >= base &&
        p < static_cast<const char*>(base) + capacity;
  }
};

// ---------------------------------------------------------------------------
// RocmAllocator — same control flow as CudaAllocator.
// ---------------------------------------------------------------------------

class RocmAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  Buffer malloc_async(size_t size, int device, void* stream);
  void free(Buffer buffer) override {
    free(buffer, /*force=*/false);
  }
  void free(Buffer buffer, bool force);
  size_t size(Buffer buffer) const override;

  Buffer make_buffer(void* ptr, size_t size) override;
  void release(Buffer buffer) override;

  // Free device storage only (CUDA free_async).
  void free_async(RocmBuffer& buf, void* stream = nullptr);
  // Free device + delete shell (CUDA free_cuda_buffer).
  void free_rocm_buffer(RocmBuffer* buf);

  void ensure_host_shadow(RocmBuffer& buf);
  void flush_host_shadow(RocmBuffer& buf);

  bool decode_arena_begin(size_t capacity, int device, void* stream);
  void decode_arena_reset();
  void decode_arena_freeze_floor();
  void decode_arena_reset_to_floor();
  void decode_arena_end();
  bool decode_arena_active() const {
    return decode_arena_.active;
  }
  size_t decode_arena_high_water() const {
    return decode_arena_.high_water;
  }
  bool decode_arena_overflowed() const {
    return decode_arena_.overflowed;
  }

  bool train_arena_begin(size_t capacity, int device, void* stream) {
    return decode_arena_begin(capacity, device, stream);
  }
  void train_arena_reset() {
    decode_arena_reset();
  }
  void train_arena_freeze_floor() {
    decode_arena_freeze_floor();
  }
  void train_arena_reset_to_floor() {
    decode_arena_reset_to_floor();
  }
  void train_arena_end() {
    decode_arena_end();
  }
  bool train_arena_active() const {
    return decode_arena_active();
  }
  size_t train_arena_high_water() const {
    return decode_arena_high_water();
  }
  bool train_arena_overflowed() const {
    return decode_arena_overflowed();
  }

  // --- Phase-scoped memory (lifetime-aware freelist) ---
  // Phase ints match rocm::MemoryPhase in rocm.h (avoid circular includes).
  void set_memory_phase(int phase);
  int memory_phase() const {
    return phase_;
  }
  uint32_t memory_generation() const {
    return generation_;
  }
  // Drop freelist blocks stamped with the given generation (or the last
  // prefill generation if gen==0 and a prefill phase was recorded).
  // Returns bytes returned to the driver via the cache free callback.
  size_t drop_generation(uint32_t gen = 0);
  // Prefill → Decode handoff: drop prefill-gen freelist + switch phase.
  size_t end_prefill();

  size_t get_active_memory() const;
  size_t get_peak_memory() const;
  void reset_peak_memory();
  size_t get_memory_limit();
  size_t set_memory_limit(size_t limit);
  size_t get_cache_memory() const;
  size_t set_cache_limit(size_t limit);
  void clear_cache();

 private:
  RocmAllocator();
  friend RocmAllocator& allocator();

  RocmBuffer* arena_alloc(size_t size);
  void apply_phase_policy_locked();
  uint32_t stamp_generation() const {
    return generation_;
  }

  std::mutex mutex_;
  size_t memory_limit_;
  size_t free_limit_;
  size_t total_memory_{0};
  size_t max_pool_size_;
  BufferCache<RocmBuffer> buffer_cache_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  std::vector<void*> free_streams_;
  std::vector<void*> mem_pools_;
  SmallSizePool scalar_pool_;
  DecodeArena decode_arena_;

  int phase_{0}; // MemoryPhase::Idle
  uint32_t generation_{1};
  uint32_t prefill_generation_{0};
  size_t last_drop_bytes_{0};
  int last_drop_buffers_{0};
};

RocmAllocator& allocator();

class CommandEncoder;
Buffer malloc_async(size_t size, CommandEncoder& encoder);

} // namespace mlx::core::rocm
