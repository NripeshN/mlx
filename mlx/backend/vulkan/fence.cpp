#include "mlx/fence.h"

namespace mlx::core {

Fence::Fence(Stream) {}

void Fence::wait(Stream, const array&) {}

void Fence::update(Stream, const array&, bool) {}

} // namespace mlx::core