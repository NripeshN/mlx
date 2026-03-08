#!/bin/bash

set -euo pipefail

BUILD_WHEEL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --wheel)
      BUILD_WHEEL=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

source venv/bin/activate

if [ "${OPENCODE:-0}" = "1" ]; then
  CMAKE_VERBOSE_MAKEFILE=OFF
  PIP_VERBOSE_FLAG=""
else
  CMAKE_VERBOSE_MAKEFILE=ON
  PIP_VERBOSE_FLAG="-v"
fi

CMAKE_ARGS="-DMLX_BUILD_VULKAN=ON -DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE} -DMLX_USE_CCACHE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
CMAKE_ARGS="-DMLX_BUILD_TESTS=ON $CMAKE_ARGS"

if command -v ccache >/dev/null 2>&1; then
  export CCACHE_BASEDIR="$(pwd)"
  export CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/ccache}"
  export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
  export CCACHE_COMPILERCHECK=content
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
fi

if [ "$BUILD_WHEEL" = "1" ]; then
  rm -rf wheelhouse
  mkdir -p wheelhouse
  CMAKE_ARGS="$CMAKE_ARGS" uv build --wheel --python 3.13 --out-dir wheelhouse .
  ln -sf wheelhouse/*.whl mlx-0.31.1-cp313-cp313-linux_x86_64.whl
else
  if [ -n "${PIP_VERBOSE_FLAG}" ]; then
    CMAKE_ARGS="$CMAKE_ARGS" pip install "$PIP_VERBOSE_FLAG" -e . \
      --config-settings=build_ext.build-temp=build \
      --config-settings=build_ext.build-lib=build/lib
  else
    CMAKE_ARGS="$CMAKE_ARGS" pip install -e . \
      --config-settings=build_ext.build-temp=build \
      --config-settings=build_ext.build-lib=build/lib
  fi
  # Build C++ tests
  echo "Building C++ tests..."
  cmake --build build --target tests -j$(nproc)
fi

if command -v ccache >/dev/null 2>&1; then
  ccache -s
fi
