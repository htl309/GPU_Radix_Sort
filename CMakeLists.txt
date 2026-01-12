cmake_minimum_required(VERSION 3.26)

project(GPURadixSort
    VERSION 0.1.0
    DESCRIPTION "SDU-ZQ"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)





find_package(Vulkan REQUIRED)

include_directories(${Vulkan_INCLUDE_DIRS})

add_subdirectory(src)







