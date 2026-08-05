cmake_minimum_required(VERSION 3.19)
project(RendererProject LANGUAGES CXX)

# 1. metal-cpp requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 2. Add your C++ executable
add_executable(Renderer main.cpp)

# 3. Locate the required native Apple Frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

# 4. Point this to your metal-cpp directory path
# (Tip: Use a relative path like "./metal-cpp" if it's inside your project folder)
set(METAL_CPP_DIR "./metal-cpp")

# 5. Include the headers
target_include_directories(Renderer PRIVATE ${METAL_CPP_DIR})

# 6. Link the discovered frameworks to your binary
target_link_libraries(Renderer PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
)
