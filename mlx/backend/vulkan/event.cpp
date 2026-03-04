#include "mlx/event.h"

namespace mlx::core {

Event::Event(Stream) {}

void Event::wait() {}

void Event::wait(Stream) {}

void Event::signal(Stream) {}

bool Event::is_signaled() const {
  return true;
}

} // namespace mlx::core