cmake_minimum_required(VERSION 3.19)
project(RendererProject LANGUAGES CXX OBJCXX)

# compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# metal-cpp requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compile GLFW directly from the library folder
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "")
set(GLFW_INSTALL OFF CACHE INTERNAL "")

# Add your C++ executable
file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
)
add_executable(Renderer ${SOURCES})

# Point this to your metal-cpp directory path
set(METAL_CPP_DIR "./library/metal-cpp")

# Point to your custom local GLFW path [3]
add_subdirectory("./library/glfw")

# Locate the required native Apple Frameworks
find_library(APPKIT_FRAMEWORK AppKit REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

# Include the headers
target_include_directories(Renderer PRIVATE ${METAL_CPP_DIR})

# Link the discovered frameworks to your binary
target_link_libraries(Renderer PRIVATE
    ${APPKIT_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
    glfw
)
