#!/bin/bash

set -euo pipefail

# Script for fast editable builds during development
# Assumes: pip install -e . has already been run once
# Usage: ./build-editable.sh

BUILD_DIR="build"

# Activate virtual environment
source venv/bin/activate

# Detect Python version and platform for the .so filename using venv python
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
PLATFORM=$(python -c "import platform; print(platform.machine().lower())")

# Determine the expected output filename
# Format: core.cpython-XY-platform-linux-gnu.so (Linux) or core.cpython-XY-darwin.so (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    SO_FILENAME="core.cpython-${PYTHON_VERSION}-darwin.so"
else
    SO_FILENAME="core.cpython-${PYTHON_VERSION}-${PLATFORM}-linux-gnu.so"
fi

echo "Building for Python ${PYTHON_VERSION} on ${PLATFORM}..."
echo "Expected output: ${SO_FILENAME}"

# Set up CMake arguments - always build with Vulkan
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=RelWithDebInfo -DMLX_BUILD_VULKAN=ON -DMLX_USE_CCACHE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
CMAKE_ARGS="-DMLX_BUILD_TESTS=ON $CMAKE_ARGS"

# Enable CUDA backend if requested
if [[ "${MLX_BUILD_CUDA:-0}" == "1" ]]; then
    CMAKE_ARGS="-DMLX_BUILD_CUDA=ON $CMAKE_ARGS"
fi

# Enable ccache if available
if command -v ccache >/dev/null 2>&1; then
    export CCACHE_BASEDIR="$(pwd)"
    export CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/ccache}"
    export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
    export CCACHE_COMPILERCHECK=content
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
fi

# Check if build directory exists and configure if needed
if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "Configuring CMake..."
    mkdir -p "$BUILD_DIR"
    cmake -S . -B "$BUILD_DIR" $CMAKE_ARGS \
        -DMLX_BUILD_PYTHON_BINDINGS=ON \
        -DMLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY="$(pwd)/${BUILD_DIR}"
fi

# Build the core target (Python extension)
echo "Building mlx Python extension..."
cmake --build "$BUILD_DIR" --target core -j$(nproc)

# Find the built .so file
BUILT_SO="${BUILD_DIR}/${SO_FILENAME}"

if [ ! -f "$BUILT_SO" ]; then
    # Try to find it with a pattern if exact name doesn't match
    BUILT_SO=$(find "$BUILD_DIR" -maxdepth 1 -name "core.cpython*.so" -type f | head -1)
    if [ -z "$BUILT_SO" ]; then
        echo "Error: Could not find built .so file in $BUILD_DIR"
        echo "Looking for: ${SO_FILENAME}"
        ls -la "$BUILD_DIR/"
        exit 1
    fi
fi

echo "Found built extension: $BUILT_SO"

# Copy to the editable install location
mkdir -p "python/mlx"
cp -v "$BUILT_SO" "python/mlx/${SO_FILENAME}"

# Also copy any shared libraries if they exist
if [ -d "$BUILD_DIR/lib" ]; then
    echo "Copying shared libraries..."
    mkdir -p "python/mlx/lib"
    cp -v "$BUILD_DIR/lib"/*.so* "python/mlx/lib/" 2>/dev/null || true
fi

# Show ccache stats if available
if command -v ccache >/dev/null 2>&1; then
    echo ""
    ccache -s
fi

echo ""
echo "Build complete! The extension is available at: python/mlx/${SO_FILENAME}"
