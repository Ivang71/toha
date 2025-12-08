cmake_minimum_required(VERSION 3.20)

project(voxel_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Vulkan REQUIRED)

include(FetchContent)

set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
file(GLOB SHADER_SOURCES
  ${SHADER_DIR}/*.vert
  ${SHADER_DIR}/*.frag
  ${SHADER_DIR}/*.comp
)

set(COMPILED_SHADERS)
foreach(SHADER ${SHADER_SOURCES})
  get_filename_component(FILE_NAME ${SHADER} NAME)
  set(SPV ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${FILE_NAME}.spv)
  add_custom_command(
    OUTPUT ${SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders
    COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o ${SPV} ${SHADER}
    DEPENDS ${SHADER}
    VERBATIM
  )
  list(APPEND COMPILED_SHADERS ${SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${COMPILED_SHADERS})

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

add_executable(voxel_engine
  src/main.cpp
  src/core/logging.cpp
  src/render/vulkan/vulkan_debug.cpp
  src/render/vulkan/vulkan_app.cpp
)

target_include_directories(voxel_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(voxel_engine PRIVATE
  Vulkan::Vulkan
  glfw
)

add_dependencies(voxel_engine shaders)

if(MSVC)
  target_compile_options(voxel_engine PRIVATE /W4 /permissive-)
else()
  target_compile_options(voxel_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()


