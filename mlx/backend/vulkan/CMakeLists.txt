find_package(Vulkan COMPONENTS glslc REQUIRED)

file(GLOB VULKAN_SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.comp")
set(VULKAN_SHADER_HEADERS "")

foreach(shader ${VULKAN_SHADERS})
  get_filename_component(shader_name ${shader} NAME_WE)
  set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.spv.h")
  
  add_custom_command(
    OUTPUT ${output_file}
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} -mfmt=c ${shader} -o ${output_file}
    DEPENDS ${shader}
    COMMENT "Compiling Vulkan compute shader ${shader_name}.comp"
  )
  list(APPEND VULKAN_SHADER_HEADERS ${output_file})
endforeach()

add_custom_target(vulkan_shaders DEPENDS ${VULKAN_SHADER_HEADERS})

target_sources(mlx PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/vulkan.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
)

target_include_directories(mlx PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(mlx PRIVATE Vulkan::Vulkan)
add_dependencies(mlx vulkan_shaders)
