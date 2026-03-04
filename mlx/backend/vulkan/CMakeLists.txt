find_package(Vulkan COMPONENTS glslc REQUIRED)

if(Vulkan_FOUND)
  message(STATUS "Vulkan found: ${Vulkan_VERSION}")
  
  # Test shader extension support
  function(test_shader_extension_support EXTENSION_NAME TEST_SHADER_FILE RESULT_VARIABLE)
    execute_process(
      COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o - -fshader-stage=compute --target-env=vulkan1.3 "${TEST_SHADER_FILE}"
      OUTPUT_VARIABLE glslc_output
      ERROR_VARIABLE glslc_error
    )
    
    if(${glslc_error} MATCHES ".*extension not supported: ${EXTENSION_NAME}.*")
      message(STATUS "${EXTENSION_NAME} not supported by glslc")
      set(${RESULT_VARIABLE} OFF PARENT_SCOPE)
    else()
      message(STATUS "${EXTENSION_NAME} supported by glslc")
      set(${RESULT_VARIABLE} ON PARENT_SCOPE)
    endif()
  endfunction()
  
  # Test extensions
  set(VULKAN_SHADER_GEN_CMAKE_ARGS "")
  
  test_shader_extension_support(
    "GL_KHR_cooperative_matrix"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/feature-tests/coopmat.comp"
    "MLX_VULKAN_COOPMAT_GLSLC_SUPPORT"
  )
  
  test_shader_extension_support(
    "GL_NV_cooperative_matrix2"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/feature-tests/coopmat2.comp"
    "MLX_VULKAN_COOPMAT2_GLSLC_SUPPORT"
  )
  
  test_shader_extension_support(
    "GL_EXT_integer_dot_product"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/feature-tests/integer_dot.comp"
    "MLX_VULKAN_INTEGER_DOT_GLSLC_SUPPORT"
  )
  
  test_shader_extension_support(
    "GL_EXT_bfloat16"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/feature-tests/bfloat16.comp"
    "MLX_VULKAN_BFLOAT16_GLSLC_SUPPORT"
  )
  
  # Build shader generator tool
  set(SHADER_GEN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kernels/vulkan-shaders-gen.cpp")
  set(SHADER_GEN_BIN "${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders-gen")
  
  add_custom_command(
    OUTPUT ${SHADER_GEN_BIN}
    COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -O2 -o ${SHADER_GEN_BIN} ${SHADER_GEN_SRC} -lpthread
    DEPENDS ${SHADER_GEN_SRC}
    COMMENT "Building Vulkan shader generator"
  )
  
  # Get all shader files
  file(GLOB SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.comp")
  set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
  set(SHADER_HEADER "${CMAKE_CURRENT_BINARY_DIR}/vulkan_shaders.hpp")
  
  # Generate shader compilation commands
  set(SHADER_CPP_FILES "")
  foreach(shader_file ${SHADER_FILES})
    get_filename_component(shader_name ${shader_file} NAME)
    set(shader_cpp "${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.cpp")
    
    add_custom_command(
      OUTPUT ${shader_cpp}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
      COMMAND ${SHADER_GEN_BIN}
        --glslc ${Vulkan_GLSLC_EXECUTABLE}
        --source ${shader_file}
        --output-dir ${SHADER_OUTPUT_DIR}
        --target-hpp ${SHADER_HEADER}
        --target-cpp ${shader_cpp}
      DEPENDS ${shader_file} ${SHADER_GEN_BIN}
      COMMENT "Generating shader variants for ${shader_name}"
    )
    
    list(APPEND SHADER_CPP_FILES ${shader_cpp})
  endforeach()
  
  # Generate main header
  add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND ${SHADER_GEN_BIN}
      --output-dir ${SHADER_OUTPUT_DIR}
      --target-hpp ${SHADER_HEADER}
    DEPENDS ${SHADER_GEN_BIN}
    COMMENT "Generating Vulkan shaders header"
  )
  
  add_custom_target(vulkan_shaders_target DEPENDS ${SHADER_HEADER} ${SHADER_CPP_FILES})
  
  # Vulkan sources
  target_sources(mlx PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/vulkan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
    ${SHADER_CPP_FILES}
  )
  
  target_include_directories(mlx PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(mlx PRIVATE Vulkan::Vulkan)
  add_dependencies(mlx vulkan_shaders_target)
  
  # Set definitions based on extension support
  if(MLX_VULKAN_COOPMAT_GLSLC_SUPPORT)
    target_compile_definitions(mlx PRIVATE MLX_VULKAN_COOPMAT_GLSLC_SUPPORT)
  endif()
  if(MLX_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    target_compile_definitions(mlx PRIVATE MLX_VULKAN_COOPMAT2_GLSLC_SUPPORT)
  endif()
  if(MLX_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    target_compile_definitions(mlx PRIVATE MLX_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
  endif()
  if(MLX_VULKAN_BFLOAT16_GLSLC_SUPPORT)
    target_compile_definitions(mlx PRIVATE MLX_VULKAN_BFLOAT16_GLSLC_SUPPORT)
  endif()
  
else()
  message(WARNING "Vulkan not found, Vulkan backend will not be built")
endif()
