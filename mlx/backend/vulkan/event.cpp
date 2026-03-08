// Copyright © 2024 Apple Inc.

#include "mlx/event.h"

#include <condition_variable>
#include <mutex>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/scheduler.h"

namespace mlx::core {

namespace {

struct EventCounter {
  uint64_t value{0};
  std::mutex mutex;
  std::condition_variable cv;
};

void set_event_value(const std::shared_ptr<void>& event, uint64_t value) {
  auto* counter = static_cast<EventCounter*>(event.get());
  {
    std::lock_guard<std::mutex> lock(counter->mutex);
    counter->value = value;
  }
  counter->cv.notify_all();
}

} // namespace

Event::Event(Stream stream) : stream_(stream) {
  auto dtor = [](void* ptr) { delete static_cast<EventCounter*>(ptr); };
  event_ = std::shared_ptr<void>(new EventCounter{}, dtor);
}

void Event::wait() {
  auto* counter = static_cast<EventCounter*>(event_.get());
  std::unique_lock<std::mutex> lock(counter->mutex);
  if (counter->value >= value()) {
    return;
  }
  counter->cv.wait(lock, [counter, wait_value = value()] {
    return counter->value >= wait_value;
  });
}

void Event::wait(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable { wait(); });
    return;
  }

  // Vulkan stream waits are currently host-mediated.
  wait();
}

void Event::signal(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(
        stream, [event = event_, signal_value = value()]() mutable {
          set_event_value(event, signal_value);
        });
    return;
  }

  // Signal only after prior work in the stream is complete.
  vulkan::add_completion_callback_for_stream(
      stream, [event = event_, signal_value = value()]() mutable {
        set_event_value(event, signal_value);
      });
}

bool Event::is_signaled() const {
  auto* counter = static_cast<EventCounter*>(event_.get());
  std::lock_guard<std::mutex> lock(counter->mutex);
  return counter->value >= value();
}

} // namespace mlx::core
