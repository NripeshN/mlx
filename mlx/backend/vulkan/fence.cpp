// Copyright © 2024 Apple Inc.

#include "mlx/fence.h"

#include <condition_variable>
#include <mutex>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/scheduler.h"

namespace mlx::core {

namespace {

struct FenceImpl {
  uint32_t count{0};
  uint32_t value{0};
  Stream stream{0, Device::cpu};
  std::mutex mutex;
  std::condition_variable cv;
};

void signal_fence_value(const std::shared_ptr<void>& fence, uint32_t value) {
  auto* impl = static_cast<FenceImpl*>(fence.get());
  {
    std::lock_guard<std::mutex> lock(impl->mutex);
    impl->value = value;
  }
  impl->cv.notify_all();
}

void wait_fence_value(const std::shared_ptr<void>& fence, uint32_t value) {
  auto* impl = static_cast<FenceImpl*>(fence.get());
  std::unique_lock<std::mutex> lock(impl->mutex);
  if (impl->value >= value) {
    return;
  }
  impl->cv.wait(lock, [impl, value] { return impl->value >= value; });
}

} // namespace

Fence::Fence(Stream stream) {
  auto dtor = [](void* ptr) { delete static_cast<FenceImpl*>(ptr); };
  fence_ = std::shared_ptr<void>(new FenceImpl{}, dtor);
  static_cast<FenceImpl*>(fence_.get())->stream = stream;
}

void Fence::wait(Stream stream, const array&) {
  auto* impl = static_cast<FenceImpl*>(fence_.get());
  uint32_t target = 0;
  {
    std::lock_guard<std::mutex> lock(impl->mutex);
    target = impl->count;
  }

  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [fence = fence_, target]() mutable {
      wait_fence_value(fence, target);
    });
    return;
  }

  if (impl->stream.device == Device::gpu) {
    vulkan::synchronize_stream(impl->stream);
  }
  wait_fence_value(fence_, target);
}

void Fence::update(Stream stream, const array&, bool) {
  auto* impl = static_cast<FenceImpl*>(fence_.get());
  uint32_t target = 0;
  {
    std::lock_guard<std::mutex> lock(impl->mutex);
    target = ++impl->count;
  }

  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [fence = fence_, target]() mutable {
      signal_fence_value(fence, target);
    });
    return;
  }

  // Signal only after prior work in the stream is complete.
  vulkan::add_completion_callback_for_stream(
      stream, [fence = fence_, target]() mutable {
        signal_fence_value(fence, target);
      });
}

} // namespace mlx::core
