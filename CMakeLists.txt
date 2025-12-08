cmake_minimum_required(VERSION 3.20)

project(voxel_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Vulkan REQUIRED)

include(FetchContent)

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
)

target_link_libraries(voxel_engine PRIVATE
  Vulkan::Vulkan
  glfw
)

if(MSVC)
  target_compile_options(voxel_engine PRIVATE /W4 /permissive-)
else()
  target_compile_options(voxel_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()


