# FindShaderc.cmake - Finds shaderc library for runtime JIT compilation
#
# This module finds the shaderc library installed on the system.
# For systems without system shaderc, the user can specify:
#   SHADERC_ROOT - Root directory of shaderc installation
#
# Defines:
#   Shaderc_FOUND - TRUE if shaderc is available
#   Shaderc::shaderc_combined - Imported target for the combined static library
#   SPIRV-Tools - Imported target for SPIRV-Tools (static)
#   SPIRV-Tools-opt - Imported target for SPIRV-Tools optimizer (static)
#
# Variables:
#   SHADERC_INCLUDE_DIR - Directory containing shaderc headers
#   SHADERC_LIBRARY - Path to libshaderc_combined.a

include(FindPackageHandleStandardArgs)

# Search for shaderc headers
find_path(SHADERC_INCLUDE_DIR
  NAMES shaderc/shaderc.hpp
  PATHS
    ${SHADERC_ROOT}/include
    /usr/include
    /usr/local/include
    /opt/shaderc/include
)

# Search for shaderc combined static library (includes all dependencies)
find_library(SHADERC_LIBRARY
  NAMES shaderc_combined
  PATHS
    ${SHADERC_ROOT}/lib
    /usr/lib
    /usr/local/lib
    /opt/shaderc/lib
)

# Search for additional libraries required for static linking
find_library(GLSLANG_LIBRARY
  NAMES glslang
  PATHS
    ${SHADERC_ROOT}/lib
    /usr/lib
    /usr/local/lib
)

find_library(SPIRV_TOOLS_LIBRARY
  NAMES SPIRV-Tools
  PATHS
    ${SHADERC_ROOT}/lib
    /usr/lib
    /usr/local/lib
)

find_library(SPIRV_TOOLS_OPT_LIBRARY
  NAMES SPIRV-Tools-opt
  PATHS
    ${SHADERC_ROOT}/lib
    /usr/lib
    /usr/local/lib
)

find_library(GLSLANG_SPIRV_LIBRARY
  NAMES SPIRV
  PATHS
    ${SHADERC_ROOT}/lib
    /usr/lib
    /usr/local/lib
)

# Handle the REQUIRED argument - shaderc is mandatory for Vulkan backend
find_package_handle_standard_args(Shaderc
  REQUIRED_VARS SHADERC_LIBRARY SHADERC_INCLUDE_DIR
  FAIL_MESSAGE "shaderc not found. Install shaderc or set SHADERC_ROOT to the installation directory."
)

if(Shaderc_FOUND)
  # Create shaderc imported target
  if(NOT TARGET Shaderc::shaderc_combined)
    add_library(Shaderc::shaderc_combined STATIC IMPORTED)
    set_target_properties(Shaderc::shaderc_combined PROPERTIES
      IMPORTED_LOCATION "${SHADERC_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${SHADERC_INCLUDE_DIR}"
    )
  endif()

  # Create glslang imported target
  if(GLSLANG_LIBRARY AND NOT TARGET glslang)
    add_library(glslang STATIC IMPORTED)
    set_target_properties(glslang PROPERTIES
      IMPORTED_LOCATION "${GLSLANG_LIBRARY}"
    )
  endif()

  # Create SPIRV-Tools imported target
  if(SPIRV_TOOLS_LIBRARY AND NOT TARGET SPIRV-Tools)
    add_library(SPIRV-Tools STATIC IMPORTED)
    set_target_properties(SPIRV-Tools PROPERTIES
      IMPORTED_LOCATION "${SPIRV_TOOLS_LIBRARY}"
    )
  endif()

  # Create SPIRV-Tools-opt imported target
  if(SPIRV_TOOLS_OPT_LIBRARY AND NOT TARGET SPIRV-Tools-opt)
    add_library(SPIRV-Tools-opt STATIC IMPORTED)
    set_target_properties(SPIRV-Tools-opt PROPERTIES
      IMPORTED_LOCATION "${SPIRV_TOOLS_OPT_LIBRARY}"
    )
  endif()

  # Create glslang SPIRV imported target
  if(GLSLANG_SPIRV_LIBRARY AND NOT TARGET glslang-SPIRV)
    add_library(glslang-SPIRV STATIC IMPORTED)
    set_target_properties(glslang-SPIRV PROPERTIES
      IMPORTED_LOCATION "${GLSLANG_SPIRV_LIBRARY}"
    )
  endif()

  mark_as_advanced(SHADERC_INCLUDE_DIR SHADERC_LIBRARY)
endif()
