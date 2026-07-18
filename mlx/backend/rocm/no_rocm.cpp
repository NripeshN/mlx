// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/rocm.h"
#include "mlx/fast.h"

#include <stdexcept>
#include <vector>

namespace mlx::core {

namespace rocm {

bool is_available() {
  return false;
}

bool train_arena_begin(size_t) {
  return false;
}
void train_arena_reset() {}
void train_arena_end() {}
bool train_arena_active() {
  return false;
}
size_t train_arena_high_water() {
  return 0;
}
bool train_arena_overflowed() {
  return false;
}

void set_memory_phase(MemoryPhase) {}
MemoryPhase memory_phase() {
  return MemoryPhase::Idle;
}
size_t memory_end_prefill() {
  return 0;
}
size_t memory_drop_generation(uint32_t) {
  return 0;
}
uint32_t memory_generation() {
  return 0;
}

array moe_swiglu_sorted(
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    StreamOrDevice) {
  throw std::runtime_error("moe_swiglu_sorted requires ROCm");
}

std::vector<array> moe_swiglu_sorted_vjp(
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    const array&,
    StreamOrDevice) {
  throw std::runtime_error("moe_swiglu_sorted_vjp requires ROCm");
}

} // namespace rocm

namespace fast {

CustomKernelFunction hip_kernel(
    const std::string&,
    const std::vector<std::string>&,
    const std::vector<std::string>&,
    const std::string&,
    const std::string&,
    bool,
    int,
    std::vector<std::pair<int, int>>) {
  throw std::runtime_error("[hip_kernel] No ROCm back-end.");
}

} // namespace fast

} // namespace mlx::core
