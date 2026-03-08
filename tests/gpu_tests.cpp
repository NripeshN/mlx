// Copyright © 2023-2024 Apple Inc.

#include <array>

#include "doctest/doctest.h"
#include "mlx/mlx.h"

using namespace mlx::core;

static const std::array<Dtype, 5> types =
    {bool_, uint32, int32, int64, float32};

TEST_CASE("test gpu arange") {
  for (auto t : types) {
    if (t == bool_) {
      continue;
    }
    auto out_cpu = arange(1, 100, 2, t, Device::cpu);
    auto out_gpu = arange(1, 100, 2, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());

    out_cpu = arange(1, 5, 0.25, t, Device::cpu);
    out_gpu = arange(1, 5, 0.25, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }
}

TEST_CASE("test gpu full") {
  for (auto t : types) {
    auto out_cpu = full({4, 4}, 2, t, Device::cpu);
    auto out_gpu = full({4, 4}, 2, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }

  // Check broadcasting works
  {
    auto x = full({2, 2}, array({3, 4}, {2, 1}), Device::gpu);
    CHECK(
        array_equal(x, array({3, 3, 4, 4}, {2, 2}), Device::cpu).item<bool>());
    x = full({2, 2}, array({3, 4}, {1, 2}), Device::gpu);
    CHECK(
        array_equal(x, array({3, 4, 3, 4}, {2, 2}), Device::cpu).item<bool>());
  }

  // Check zeros and ones
  {
    auto x = zeros({2, 2}, float32, Device::gpu);
    auto y = array({0.0, 0.0, 0.0, 0.0}, {2, 2});
    CHECK(array_equal(x, y, Device::cpu).item<bool>());

    x = ones({2, 2}, float32, Device::gpu);
    y = array({1.0, 1.0, 1.0, 1.0}, {2, 2});
    CHECK(array_equal(x, y, Device::cpu).item<bool>());
  }
}

TEST_CASE("test gpu astype") {
  array x = array({-4, -3, -2, -1, 0, 1, 2, 3});
  // Check all types work
  for (auto t : types) {
    auto out_cpu = astype(x, t, Device::cpu);
    auto out_gpu = astype(x, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }

  x = transpose(reshape(x, {2, 2, 2}), {1, 2, 0});
  for (auto t : types) {
    auto out_cpu = astype(x, t, Device::cpu);
    auto out_gpu = astype(x, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }
}

TEST_CASE("test gpu reshape") {
  array x = array({0, 1, 2, 3, 4, 5, 6, 7});
  auto out_cpu = reshape(x, {2, 2, 2});
  auto out_gpu = reshape(x, {2, 2, 2}, Device::gpu);
  CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());

  x = transpose(reshape(x, {2, 2, 2}), {1, 2, 0});
  out_cpu = reshape(x, {4, 2});
  out_gpu = reshape(x, {4, 2}, Device::gpu);
  CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());

  out_cpu = reshape(x, {8});
  out_gpu = reshape(x, {8}, Device::gpu);
  CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
}

TEST_CASE("test gpu reduce") {
  {
    array a(true);
    CHECK_EQ(all(a, Device::gpu).item<bool>(), true);
    CHECK_EQ(any(a, Device::gpu).item<bool>(), true);

    a = array(std::initializer_list<bool>{});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), true);
    CHECK_EQ(any(a, Device::gpu).item<bool>(), false);
  }

  {
    std::vector<int> vals(33, 1);
    array a(vals.data(), {33});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), true);

    vals[32] = 0;
    a = array(vals.data(), {33});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), false);
  }

  {
    std::vector<int> vals(33, 0);
    array a(vals.data(), {33});
    CHECK_EQ(any(a, Device::gpu).item<bool>(), false);

    vals[32] = 1;
    a = array(vals.data(), {33});
    CHECK_EQ(any(a, Device::gpu).item<bool>(), true);
  }

  {
    std::vector<int> vals(1 << 14, 0);
    array a(vals.data(), {1 << 14});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), false);
    CHECK_EQ(any(a, Device::gpu).item<bool>(), false);

    vals[4] = 1;
    vals[999] = 1;
    vals[2000] = 1;
    a = array(vals.data(), {1 << 14});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), false);
    CHECK_EQ(any(a, Device::gpu).item<bool>(), true);
  }

  // sum and prod
  {
    array a = array({true, false, true});
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 2);
    CHECK_EQ(prod(a, Device::gpu).item<bool>(), false);

    a = array({true, true, true});
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 3);
    CHECK_EQ(prod(a, Device::gpu).item<bool>(), true);

    a = full({2, 2, 2}, 2.0f);
    CHECK_EQ(sum(a, Device::gpu).item<float>(), 16.0f);
    CHECK_EQ(prod(a, Device::gpu).item<float>(), 256.0f);

    a = full({500, 2, 2}, 1u);
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 2000);
    CHECK_EQ(prod(a, Device::gpu).item<uint32_t>(), 1u);

    a = full({500, 2, 2}, 1);
    CHECK_EQ(sum(a, Device::gpu).item<int32_t>(), 2000);
    CHECK_EQ(prod(a, Device::gpu).item<int32_t>(), 1);
  }

  // sum and prod overflow
  {
    auto a = full({256, 2, 2}, 1u, uint8);
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 256 * 4);
    CHECK_EQ(prod(a, Device::gpu).item<uint32_t>(), 1);

    a = full({65535, 2, 2}, 1u, uint16);
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 65535 * 4);
    CHECK_EQ(prod(a, Device::gpu).item<uint32_t>(), 1);
  }
}

TEST_CASE("test gpu reduce with axes") {
  // reducing only some axes and irregular layouts
  {
    array a(1.0f);
    a = broadcast_to(a, {2, 2, 2});
    CHECK_EQ(sum(a, Device::gpu).item<float>(), 8.0f);

    a = ones({2, 4, 8, 16});
    for (auto ax : {0, 1, 2, 3}) {
      auto out_gpu = sum(a, ax, false, Device::gpu);
      auto out_cpu = sum(a, ax, false, Device::cpu);
      CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
    }

    for (auto ax : {1, 2, 3}) {
      auto out_gpu = sum(a, {0, ax}, false, Device::gpu);
      auto out_cpu = sum(a, {0, ax}, false, Device::cpu);
      CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
    }
    for (auto ax : {2, 3}) {
      auto out_gpu = sum(a, {0, 1, ax}, false, Device::gpu);
      auto out_cpu = sum(a, {0, 1, ax}, false, Device::cpu);
      CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
    }

    auto b = reshape(arange(0.0f, 2.0f * 3.0f * 4.0f * 5.0f), {2, 3, 4, 5});
    auto b_nc = swapaxes(b, 1, 3);
    auto out_gpu_keepdims = sum(b_nc, {0, 2}, true, Device::gpu);
    auto out_cpu_keepdims = sum(b_nc, {0, 2}, true, Device::cpu);
    CHECK(array_equal(out_gpu_keepdims, out_cpu_keepdims, Device::cpu)
              .item<bool>());

    auto b_nc_f16 = astype(b_nc, float16, Device::gpu);
    auto out_gpu_f16 = sum(b_nc_f16, {0, 2}, true, Device::gpu);
    auto out_cpu_f16 = sum(b_nc_f16, {0, 2}, true, Device::cpu);
    CHECK(allclose(out_gpu_f16, out_cpu_f16, 5e-3, 5e-3, false, Device::cpu)
              .item<bool>());
  }
}

TEST_CASE("test gpu binary ops") {
  // scalar-scalar
  {
    array a(2.0f);
    array b(4.0f);
    auto out = add(a, b, Device::gpu);
    CHECK_EQ(out.item<float>(), 6.0f);
  }

  // scalar-vector and vector-scalar
  {
    array a(2.0f);
    array b({2.0f, 4.0f, 6.0f});
    auto out = add(a, b, Device::gpu);
    auto expected = array({4.0f, 6.0f, 8.0f});
    CHECK(array_equal(out, expected, Device::cpu).item<bool>());
    out = add(b, a, Device::gpu);
    CHECK(array_equal(out, expected, Device::cpu).item<bool>());
  }

  // vector-vector
  {
    array a({0.0f, 1.0f, 2.0f});
    array b({3.0f, 4.0f, 5.0f});
    auto out = add(a, b, Device::gpu);
    auto expected = array({3.0f, 5.0f, 7.0f});
    CHECK(array_equal(out, expected, Device::cpu).item<bool>());
  }

  // general
  {
    array a({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, {2, 2, 2});
    array b({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, {2, 2, 2});
    a = transpose(a, {0, 2, 1});
    b = transpose(b, {1, 0, 2});
    auto out_gpu = add(a, b, Device::gpu);
    auto out_cpu = add(a, b, Device::cpu);
    auto expected =
        array({0.0f, 3.0f, 5.0f, 8.0f, 6.0f, 9.0f, 11.0f, 14.0f}, {2, 2, 2});
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
    CHECK(array_equal(out_gpu, expected, Device::cpu).item<bool>());
  }

  // Check all types work
  for (auto t : types) {
    auto a = astype(array({0, 1, 2}), t);
    auto b = astype(array({3, 4, 5}), t);
    auto out_cpu = add(a, b, Device::cpu);
    auto out_gpu = add(a, b, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }

  // Check subtraction
  {
    auto a = array({3, 2, 1});
    auto b = array({1, 1, 1});
    auto out = subtract(a, b, Device::gpu);
    CHECK(array_equal(out, array({2, 1, 0}), Device::cpu).item<bool>());
  }

  // Check multiplication
  {
    auto a = array({1, 2, 3});
    auto b = array({2, 2, 2});
    auto out = multiply(a, b, Device::gpu);
    CHECK(array_equal(out, array({2, 4, 6}), Device::cpu).item<bool>());
  }

  // Check division
  {
    auto x = array(1.0f);
    auto y = array(1.0f);
    CHECK_EQ(divide(x, y, Device::gpu).item<float>(), 1.0f);

    x = array(1.0f);
    y = array(0.5);
    CHECK_EQ(divide(x, y, Device::gpu).item<float>(), 2.0f);

    x = array(1.0f);
    y = array(0.0f);
    CHECK(std::isinf(divide(x, y, Device::gpu).item<float>()));

    x = array(0.0f);
    y = array(0.0f);
    CHECK(std::isnan(divide(x, y, Device::gpu).item<float>()));
  }

  // Check maximum and minimum
  {
    auto x = array(1.0f);
    auto y = array(0.0f);
    CHECK_EQ(maximum(x, y, Device::gpu).item<float>(), 1.0f);
    CHECK_EQ(minimum(x, y, Device::gpu).item<float>(), 0.0f);
    y = array(2.0f);
    CHECK_EQ(maximum(x, y, Device::gpu).item<float>(), 2.0f);
    CHECK_EQ(minimum(x, y, Device::gpu).item<float>(), 1.0f);

    auto nan = array(std::numeric_limits<float>::quiet_NaN());
    CHECK(std::isnan(maximum(nan, y, Device::gpu).item<float>()));
    CHECK(std::isnan(maximum(y, nan, Device::gpu).item<float>()));
    CHECK(std::isnan(minimum(nan, y, Device::gpu).item<float>()));
    CHECK(std::isnan(minimum(y, nan, Device::gpu).item<float>()));
  }

  // Check equal
  {
    array x(1.0f);
    array y(1.0f);
    CHECK(equal(x, y, Device::gpu).item<bool>());
    x = array(0.0f);
    CHECK(!equal(x, y, Device::gpu).item<bool>());
  }

  // Greater and less
  {
    array x(1.0f);
    array y(0.0f);
    CHECK(greater(x, y, Device::gpu).item<bool>());
    CHECK(greater_equal(x, y, Device::gpu).item<bool>());
    CHECK(!greater(y, x, Device::gpu).item<bool>());
    CHECK(!greater_equal(y, x, Device::gpu).item<bool>());
    y = array(1.0f);
    CHECK(!greater(x, y, Device::gpu).item<bool>());
    CHECK(greater_equal(x, y, Device::gpu).item<bool>());

    x = array(0.0f);
    y = array(1.0f);
    CHECK(less(x, y, Device::gpu).item<bool>());
    CHECK(less_equal(x, y, Device::gpu).item<bool>());
    CHECK(!less(y, x, Device::gpu).item<bool>());
    CHECK(!less_equal(y, x, Device::gpu).item<bool>());
    y = array(0.0f);
    CHECK(!less(x, y, Device::gpu).item<bool>());
    CHECK(less_equal(x, y, Device::gpu).item<bool>());
  }

  // Check logaddexp
  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    array x(inf);
    array y(2.0f);
    auto out = logaddexp(x, y, Device::gpu);
    CHECK_EQ(out.item<float>(), inf);

    x = array(-inf);
    out = logaddexp(x, y, Device::gpu);
    CHECK_EQ(out.item<float>(), 2.0f);

    y = array(-inf);
    out = logaddexp(x, y, Device::gpu);
    CHECK_EQ(out.item<float>(), -inf);
  }
}

TEST_CASE("test gpu unary ops") {
  // contiguous
  {
    array x({-1.0f, 0.0f, 1.0f});
    auto expected = array({1.0f, 0.0f, 1.0f});
    CHECK(array_equal(abs(x, Device::gpu), expected, Device::cpu).item<bool>());
  }

  // general
  {
    array x({-1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f, 3.0f, -3.0f});
    auto y = slice(x, {0}, {8}, {2});
    auto expected = array({1.0f, 1.0f, 1.0f, 3.0f});
    CHECK(array_equal(abs(y, Device::gpu), expected, Device::cpu).item<bool>());

    y = slice(x, {4}, {8});
    expected = array({1.0f, 1.0f, 3.0f, 3.0f});
    CHECK(array_equal(abs(y, Device::gpu), expected, Device::cpu).item<bool>());
  }

  // Test negative
  {
    array x(1.0f);
    CHECK_EQ(negative(x, Device::gpu).item<float>(), -1.0f);
  }

  // Check all types work
  for (auto t : types) {
    if (t == bool_) {
      continue;
    }
    auto in = astype(array({1}), t);
    auto out_cpu = negative(in, Device::cpu);
    auto out_gpu = negative(in, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }

  // Test log1p
  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    array x(-1.0f);
    CHECK_EQ(log1p(x, Device::gpu).item<float>(), -inf);

    x = array(0.0f);
    CHECK_EQ(log1p(x, Device::gpu).item<float>(), 0.0f);

    x = array(1e-9f);
    CHECK_EQ(log1p(x, Device::gpu).item<float>(), 1e-9f);

    x = array(-2.0f);
    CHECK(std::isnan(log1p(x, Device::gpu).item<float>()));
  }

  {
    auto x = array({-0.999f, -0.5f, 0.0f, 0.5f, 0.999f}, {5}, float32);
    auto out_gpu = erfinv(x, Device::gpu);
    auto out_cpu = erfinv(x, Device::cpu);
    CHECK(allclose(out_gpu, out_cpu, 1e-5, 1e-5, false, Device::cpu)
              .item<bool>());
  }
}

TEST_CASE("test gpu random") {
  {
    auto key = random::key(0);
    auto x = random::bits({}, 4, key, Device::gpu);
    auto y = random::bits({}, 4, key, Device::gpu);
    CHECK_EQ(x.item<uint32_t>(), 1797259609u);
    CHECK_EQ(x.item<uint32_t>(), y.item<uint32_t>());
  }

  {
    auto key = random::key(1);
    auto x = random::bits({}, 4, key, Device::gpu);
    CHECK_EQ(x.item<uint32_t>(), 507451445u);
  }

  {
    auto key = random::key(0);
    auto x = random::bits({3, 1}, 4, key, Device::gpu);
    auto expected = array({4146024105u, 1351547692u, 2718843009u}, {3, 1});
    CHECK(array_equal(x, expected, Device::cpu).item<bool>());
  }
}

TEST_CASE("test gpu matmul") {
  {
    auto a = ones({2, 2});
    auto b = ones({2, 2});
    auto out = matmul(a, b, Device::gpu);
    CHECK(array_equal(out, full({2, 2}, 2.0f), Device::cpu).item<bool>());
  }

  // Batched matmul
  {
    auto a = ones({3, 2, 2});
    auto b = ones({3, 2, 2});
    auto out = matmul(a, b, Device::gpu);
    CHECK(array_equal(out, full({3, 2, 2}, 2.0f), Device::cpu).item<bool>());
  }

  // Broadcast batched matmul
  {
    auto a = ones({1, 3, 2, 2});
    auto b = ones({3, 1, 2, 2});
    auto out = matmul(a, b, Device::gpu);
    CHECK(array_equal(out, full({3, 3, 2, 2}, 2.0f), Device::cpu).item<bool>());
  }
}

TEST_CASE("test gpu scan softmax trig parity") {
  {
    auto x = reshape(arange(1.0f, 1.0f + 2 * 3 * 5, 1.0f), {2, 3, 5});
    x = transpose(x, {1, 2, 0});

    auto cumsum_gpu = cumsum(x, 1, true, false, Device::gpu);
    auto cumsum_cpu = cumsum(x, 1, true, false, Device::cpu);
    CHECK(array_equal(cumsum_gpu, cumsum_cpu, Device::cpu).item<bool>());
  }

  {
    auto logits = reshape(arange(-12.0f, 12.0f, 0.5f), {4, 3, 4});
    logits = transpose(logits, {1, 2, 0});

    auto sm_gpu = softmax(logits, std::vector<int>{1}, false, Device::gpu);
    auto sm_cpu = softmax(logits, std::vector<int>{1}, false, Device::cpu);
    CHECK(
        allclose(sm_gpu, sm_cpu, 1e-5, 1e-5, false, Device::cpu).item<bool>());
  }

  {
    auto x = reshape(arange(-20.0f, 20.0f, 0.25f), {8, 5, 4});
    x = transpose(x, {2, 0, 1});

    auto sin_gpu = sin(x, Device::gpu);
    auto sin_cpu = sin(x, Device::cpu);
    CHECK(allclose(sin_gpu, sin_cpu, 1e-6, 1e-6, false, Device::cpu)
              .item<bool>());

    auto cos_gpu = cos(x, Device::gpu);
    auto cos_cpu = cos(x, Device::cpu);
    CHECK(allclose(cos_gpu, cos_cpu, 1e-6, 1e-6, false, Device::cpu)
              .item<bool>());
  }

  {
    auto x = reshape(arange(-9.0f, 9.0f, 0.25f), {3, 4, 6});
    x = transpose(x, {2, 0, 1});

    auto x_f16 = astype(x, float16, Device::gpu);
    auto argmax_f16_gpu = argmax(x_f16, 1, false, Device::gpu);
    auto argmax_f16_cpu = argmax(x_f16, 1, false, Device::cpu);
    CHECK(
        array_equal(argmax_f16_gpu, argmax_f16_cpu, Device::cpu).item<bool>());

    auto x_bf16 = astype(x, bfloat16, Device::gpu);
    auto argmax_bf16_gpu = argmax(x_bf16, 2, true, Device::gpu);
    auto argmax_bf16_cpu = argmax(x_bf16, 2, true, Device::cpu);
    CHECK(array_equal(argmax_bf16_gpu, argmax_bf16_cpu, Device::cpu)
              .item<bool>());
  }
}

TEST_CASE("test gpu validation") {
  // Run this test with Metal validation enabled
  // METAL_DEVICE_WRAPPER_TYPE=1 METAL_DEBUG_ERROR_MODE=0 ./tests/tests \
  //     -tc="test metal validation"

  auto x = array({});
  eval(exp(x));

  auto y = array({});
  eval(add(x, y));

  eval(sum(x));

  x = array({1, 2, 3});
  y = array(0);
  eval(gather(x, y, 0, {0}));
  eval(gather(x, y, 0, {2}));

  eval(gather(x, y, 0, {0}));
  eval(gather(x, y, 0, {2}));

  eval(scatter(x, y, array({2}), 0));

  x = arange(0, -3, 1);
  eval(x);
  array_equal(x, array({})).item<bool>();

  x = array({1.0, 0.0});
  eval(argmax(x));

  eval(scatter_max(array(1), {}, array(2), std::vector<int>{}));
}

TEST_CASE("test memory info") {
  // Test cache limits
  {
    auto old_limit = set_cache_limit(0);
    {
      auto a = zeros({4096});
      eval(a);
    }
    CHECK_EQ(get_cache_memory(), 0);
    CHECK_EQ(set_cache_limit(old_limit), 0);
    CHECK_EQ(set_cache_limit(old_limit), old_limit);
  }

  // Test memory limits
  {
    auto old_limit = set_memory_limit(10);
    CHECK_EQ(set_memory_limit(old_limit), 10);
    CHECK_EQ(set_memory_limit(old_limit), old_limit);
  }

  // Query active and peak memory
  {
    auto a = zeros({4096});
    eval(a);
    synchronize();
    auto active_mem = get_active_memory();
    CHECK(active_mem >= 4096 * 4);
    {
      auto b = zeros({4096});
      eval(b);
    }
    synchronize();
    auto new_active_mem = get_active_memory();
    CHECK_EQ(new_active_mem, active_mem);
    auto peak_mem = get_peak_memory();
    CHECK(peak_mem >= 4096 * 8);

    auto cache_mem = get_cache_memory();
    if (cache_mem != 0) {
      CHECK(cache_mem >= 4096 * 4);
    }
  }

  clear_cache();
  CHECK_EQ(get_cache_memory(), 0);
}

TEST_CASE("test gpu deferred slice update alias hazard") {
  auto src_gpu = arange(0.0f, 8.0f, 1.0f, float32, Device::gpu);
  auto upd_gpu = slice(src_gpu, {1}, {5}, Device::gpu);
  auto out_gpu =
      slice_update(src_gpu, upd_gpu, Shape{0}, Shape{4}, Device::gpu);

  auto src_cpu = arange(0.0f, 8.0f, 1.0f, float32, Device::cpu);
  auto upd_cpu = slice(src_cpu, {1}, {5}, Device::cpu);
  auto out_cpu =
      slice_update(src_cpu, upd_cpu, Shape{0}, Shape{4}, Device::cpu);

  const bool equal = array_equal(out_gpu, out_cpu, Device::cpu).item<bool>();
  CHECK(equal);
}

TEST_CASE("test gpu deferred scalar upload") {
  auto gpu = full({3, 4}, 7.0f, float32, Device::gpu);
  auto cpu = full({3, 4}, 7.0f, float32, Device::cpu);
  CHECK(array_equal(gpu, cpu, Device::cpu).item<bool>());
}

TEST_CASE("test gpu deferred rope readback") {
  auto x_gpu =
      reshape(arange(0.0f, 32.0f, 1.0f, float32, Device::gpu), {2, 2, 8});
  auto x_cpu =
      reshape(arange(0.0f, 32.0f, 1.0f, float32, Device::cpu), {2, 2, 8});
  auto offset_gpu = array({1, 3}, int32);
  auto offset_cpu = array({1, 3}, int32);
  auto freqs_gpu = arange(1.0f, 5.0f, 1.0f, float32, Device::gpu);
  auto freqs_cpu = arange(1.0f, 5.0f, 1.0f, float32, Device::cpu);

  auto out_gpu = fast::rope(
      x_gpu, 8, false, std::nullopt, 1.0f, offset_gpu, freqs_gpu, Device::gpu);
  auto out_cpu = fast::rope(
      x_cpu, 8, false, std::nullopt, 1.0f, offset_cpu, freqs_cpu, Device::cpu);

  CHECK(allclose(out_gpu, out_cpu, 1e-5f, 1e-5f, false, Device::cpu)
            .item<bool>());
}

TEST_CASE("test gpu deferred multi-submit retirement") {
  auto gpu_x = arange(0.0f, 1024.0f, 1.0f, float32, Device::gpu);
  auto cpu_x = arange(0.0f, 1024.0f, 1.0f, float32, Device::cpu);
  array one(1.0f);

  for (int i = 0; i < 48; ++i) {
    gpu_x = add(cos(gpu_x, Device::gpu), one, Device::gpu);
    cpu_x = add(cos(cpu_x, Device::cpu), one, Device::cpu);
  }

  CHECK(allclose(gpu_x, cpu_x, 1e-5f, 1e-5f, false, Device::cpu).item<bool>());
}
