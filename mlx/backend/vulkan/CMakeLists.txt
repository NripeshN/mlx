find_package(Vulkan COMPONENTS glslc REQUIRED)

if(Vulkan_FOUND)
  message(STATUS "Vulkan found: ${Vulkan_VERSION}")
  
  # Test shader extension support
  function(test_shader_extension_support EXTENSION_NAME TEST_SHADER_FILE RESULT_VARIABLE)
    execute_process(
      COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o - -fshader-stage=compute --target-env=vulkan1.3 "${TEST_SHADER_FILE}"
      RESULT_VARIABLE glslc_result
      OUTPUT_VARIABLE glslc_output
      ERROR_VARIABLE glslc_error
    )

    if(glslc_result EQUAL 0)
      message(STATUS "${EXTENSION_NAME} supported by glslc")
      set(${RESULT_VARIABLE} ON PARENT_SCOPE)
    elseif(glslc_error MATCHES ".*extension not supported: ${EXTENSION_NAME}.*")
      message(STATUS "${EXTENSION_NAME} not supported by glslc")
      set(${RESULT_VARIABLE} OFF PARENT_SCOPE)
    else()
      message(STATUS "${EXTENSION_NAME} probe failed; assuming unsupported")
      set(${RESULT_VARIABLE} OFF PARENT_SCOPE)
    endif()
  endfunction()

  # Test extensions
  set(VULKAN_SHADER_GEN_CMAKE_ARGS)

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

  if(MLX_VULKAN_COOPMAT_GLSLC_SUPPORT)
    list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS -DGGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
  endif()
  if(MLX_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS -DGGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
  endif()
  if(MLX_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS -DGGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
  endif()
  if(MLX_VULKAN_BFLOAT16_GLSLC_SUPPORT)
    list(APPEND VULKAN_SHADER_GEN_CMAKE_ARGS -DGGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
  endif()

  # Find shaderc for shader generation (required)
  include(FindShaderc REQUIRED)
  message(STATUS "shaderc found for shader generation: ${SHADERC_LIBRARY}")

  # Build shader generator tool
  set(SHADER_GEN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kernels/vulkan-shaders-gen.cpp")

  add_executable(vulkan-shaders-gen ${SHADER_GEN_SRC})
  set_target_properties(vulkan-shaders-gen PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_link_libraries(vulkan-shaders-gen PRIVATE 
    Shaderc::shaderc_combined
    glslang
    glslang-SPIRV
    SPIRV-Tools-opt
    SPIRV-Tools
  )
  target_compile_features(vulkan-shaders-gen PRIVATE cxx_std_17)
  if(VULKAN_SHADER_GEN_CMAKE_ARGS)
    target_compile_definitions(vulkan-shaders-gen PRIVATE ${VULKAN_SHADER_GEN_CMAKE_ARGS})
  endif()
  
  # Get all shader files
  file(
    GLOB SHADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.comp"
  )
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
      COMMAND vulkan-shaders-gen
        --source ${shader_file}
        --output-dir ${SHADER_OUTPUT_DIR}
        --target-hpp ${SHADER_HEADER}
        --target-cpp ${shader_cpp}
      DEPENDS ${shader_file} vulkan-shaders-gen
      COMMENT "Generating shader variants for ${shader_name}"
    )
    
    list(APPEND SHADER_CPP_FILES ${shader_cpp})
  endforeach()
  
  # Generate main header - must happen after all shader variants are compiled
  add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND vulkan-shaders-gen
      --output-dir ${SHADER_OUTPUT_DIR}
      --target-hpp ${SHADER_HEADER}
    DEPENDS vulkan-shaders-gen ${SHADER_CPP_FILES}
    COMMENT "Generating Vulkan shaders header"
  )
  
  add_custom_target(vulkan_shaders_target DEPENDS ${SHADER_HEADER} ${SHADER_CPP_FILES})
  
  # Vulkan sources
  target_sources(mlx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vulkan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/primitives_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/binary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/arange.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rope.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
    ${SHADER_CPP_FILES}
  )
  
  target_include_directories(mlx PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(mlx PRIVATE 
    Vulkan::Vulkan
    Shaderc::shaderc_combined
    glslang
    glslang-SPIRV
    SPIRV-Tools-opt
    SPIRV-Tools
  )
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
