set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_ROOT_DIR "${CMAKE_SOURCE_DIR}")
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


include_directories(
${ROOT_DIR}
)


# 递归收集所有 .cpp 和 .h 文件
file(GLOB_RECURSE ALL_SRC_FILES
    "${ROOT_DIR}/*.cpp"
    "${ROOT_DIR}/*.c"
    "${ROOT_DIR}/*.h"
    "${ROOT_DIR}/*.hpp"
	 "${ROOT_DIR}/*.cu"
	  "${ROOT_DIR}/*.cuh"
)


# 添加可执行文件
add_executable(GPURS
 ${ALL_SRC_FILES}
)

target_link_libraries(GPURS 
Vulkan::Vulkan
cudart.lib
)
set_target_properties(GPURS PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    VS_DEBUGGER_WORKING_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}"
    VS_DEBUGGER_WORKING_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}"
)
set_target_properties(GPURS PROPERTIES CUDA_ARCHITECTURES "75;86")

# 设置 source_group 让 VS 显示目录结构
foreach(FILE_PATH IN LISTS ALL_SRC_FILES)
    file(RELATIVE_PATH REL_PATH "${ROOT_DIR}" "${FILE_PATH}")
    get_filename_component(DIR_PATH "${REL_PATH}" PATH)
    string(REPLACE "/" "\\" GROUP_PATH "${DIR_PATH}")
    source_group("${GROUP_PATH}" FILES "${FILE_PATH}")
endforeach()
