# Agent Guidelines for MLX

Essential information for AI coding agents working on the MLX codebase.

## Branch Purpose: Vulkan Backend

This branch adds Vulkan GPU support to MLX as a new backend.

**Vulkan Backend Location**: `mlx/backend/vulkan/`
- Compute shader files: `*.comp`
- C++ implementation: `*.cpp` / `*.hpp`
- Key components: allocator, device, primitives, eval, event, fence, device_info

**Pattern**: Follow Metal backend (`mlx/backend/metal/`) or CUDA backend (`mlx/backend/cuda/`) as reference for structure and conventions.

## Build Commands

### Quick Development Build (Recommended for daily development)

```bash
# Fast incremental build for development
# Assumes: pip install -e . has been run once
./build-editable.sh
```

**What it does:**
- Uses the virtual environment at `./venv`
- Builds only the `core` Python extension target (skips tests/examples)
- Automatically copies the resulting `.so` to `python/mlx/` for editable installs
- Supports CUDA with `MLX_BUILD_CUDA=1 ./build-editable.sh`

### Full Build with Tests

```bash
# Complete build including C++ tests
./build-vulkan.sh

# Build wheel for distribution
./build-vulkan.sh --wheel
```

**What it does:**
- Runs `pip install -e .` with Vulkan backend enabled
- Builds all targets including C++ tests (`./build/tests/test_mlx`)
- Uses virtual environment at `./venv`

### Manual Build Options

```bash
# Manual Vulkan build
CMAKE_ARGS="-DMLX_BUILD_VULKAN=ON" pip install -e .

# C++ library only (Release with tests)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_TESTS=ON
cmake --build . -j$(nproc)

# Python development install (CPU only)
pip install --no-build-isolation -e ".[dev]"
```

### Build Scripts Summary

| Script | Use Case | Time | Output |
|--------|----------|------|--------|
| `build-editable.sh` | Daily development, quick iterations | ~2-5 min | Updates `python/mlx/core*.so` |
| `build-vulkan.sh` | Full build with tests, CI | ~10-15 min | Complete install + `./build/tests/` |

CMake options: `-DMLX_BUILD_TESTS`, `-DMLX_BUILD_EXAMPLES`, `-DMLX_BUILD_METAL`, `-DMLX_BUILD_CUDA`, `-DMLX_BUILD_VULKAN`, `-DMLX_BUILD_CPU`, `-DMLX_USE_CCACHE=ON`

## Test Commands

```bash
# All Python tests
python -m unittest discover python/tests -v

# Run on specific device
DEVICE=gpu python -m unittest discover python/tests -v
DEVICE=cpu python -m unittest discover python/tests -v

# C++ tests
./build/tests/test_mlx

# Distributed tests
mpirun --bind-to none -np 8 python python/tests/mpi_test_distributed.py
mlx.launch --verbose -n 8 python/tests/ring_test_distributed.py
```

## Lint/Format Commands

```bash
pre-commit run --all-files    # Run all checks
clang-format -i file.cpp      # Format C++
black file.py                 # Format Python
isort --profile=black file.py # Sort Python imports
cmake-format -i CMakeLists.txt
```

## Code Style

### C++ (C++20, clang-format)
- Indentation: 2 spaces
- Namespaces: `mlx::core`
- Naming: `PascalCase` for classes, `snake_case` for functions/variables
- Private members: `trailing_underscore_`
- Headers: Use `#pragma once`
- Public API: Add `MLX_API` macro (see `mlx/api.h`)

### C++ Import Order
1. System headers (`<vector>`, `<memory>`)
2. Third-party (`<nanobind/...>`, `<doctest/...>`, `<vulkan/vulkan.hpp>`)
3. MLX headers (`"mlx/..."`)
4. Local headers (`"..."`)

### Python (black, isort)
- Naming: `PascalCase` for classes, `snake_case` for functions/variables
- All files start with: `# Copyright © 20XX Apple Inc.`

### Python Import Order
1. Standard library
2. Third-party (numpy, etc.)
3. `mlx.core` and mlx imports
4. Test utilities (`mlx_tests`)

## Error Handling

### C++
- Use exceptions: `std::invalid_argument`, `std::runtime_error`, `std::out_of_range`
- Throw with descriptive messages

### Python
- Raise: `ValueError`, `TypeError`, `RuntimeError`
- Use numpy-compatible error behavior

## Testing

### C++ Tests (doctest)
- Location: `tests/` directory
- Naming: `*_tests.cpp`
- Macros: `TEST_CASE("name")`, `CHECK()`, `CHECK_EQ()`, `CHECK_THROWS_AS()`
- Running with fail-fast: `./build/tests/tests --abort-after=1`

### Python Tests (unittest)
- Location: `python/tests/`
- Naming: `test_*.py`
- Base class: `mlx_tests.MLXTestCase`
- Array comparison: `self.assertEqualArray(mx_res, expected, atol=, rtol=)`
- Set `MLX_ENABLE_TF32=0` for deterministic results

## Project Structure

```
mlx/               # C++ core library
  backend/         # cpu, metal, cuda, vulkan backends
    vulkan/        # Vulkan backend (this branch)
      *.cpp/*.hpp  # C++ implementation
      kernels/     # Additional shader resources
        *.comp     # Vulkan compute shaders (GLSL)
  io/              # safetensors, gguf I/O
python/src/        # nanobind bindings
python/tests/      # Python unit tests
tests/             # C++ unit tests
```

## Vulkan-Specific Details

### Compute Shaders (`.comp` files)
- Shaders are compiled to SPIR-V at build time using `glslc`
- Compiled headers are generated in `${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.spv.h`
- Shaders follow GLSL compute shader syntax
- Reference existing shaders in `mlx/backend/metal/kernels/` for algorithm patterns

### Build Dependencies
- `Vulkan::Vulkan` CMake package
- `glslc` (SPIR-V compiler) from Vulkan SDK

### Key Components
- **allocator.cpp/h**: Memory allocation for Vulkan buffers
- **device.cpp**: Vulkan device initialization and management
- **device_info.cpp/h**: Device capability queries
- **primitives.cpp**: Operation implementations using Vulkan compute shaders
- **eval.cpp**: Evaluation logic for Vulkan backend
- **event.cpp/fence.cpp**: Synchronization primitives
- **vulkan.cpp/h**: Core Vulkan context and utilities

## Common Patterns

### Adding a New Operation
1. Declare in `mlx/ops.h` with `MLX_API` macro
2. Implement in `mlx/ops.cpp`
3. Add primitive in `mlx/primitives.h/cpp` if needed
4. Add Vulkan kernel in `mlx/backend/vulkan/` (`.comp` shader + C++ wrapper)
5. Add Python binding in `python/src/ops.cpp`
6. Add tests in both C++ and Python

### Working with Arrays
- Use `mlx::core::array` for all array operations
- Arrays are immutable - operations return new arrays
- Use `StreamOrDevice s = {}` for device placement

## Notes for Agents

- Run tests after making changes
- Format code before committing
- Follow existing patterns in the codebase
- Check both CPU and GPU backends when applicable
- For Vulkan work: reference Metal backend (`mlx/backend/metal/`) for compute patterns and CUDA backend (`mlx/backend/cuda/`) for structure
- For Vulkan Reference implementation read ./llama.cpp/ggml/src/ggml-vulkan/
- ALWAYS check for existing shaders in ./mlx/backend/vulkan/kernels/ before introducing new onces
- Shaders should be compiled automatically by CMake; check build output if shaders fail
- NEVER edit source files outside of mlx/backends/vulkan !! (test files are allowed)

## Github instructions
- Every PR that you create, should contain the results of qwen3 benchmark by running it @benchmark_qwen3.sh against bf16 anf 8bit quants. either as pr desc or as comment 
