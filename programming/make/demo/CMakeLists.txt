# 设置项目所允许的最低的 CMake 版本：
cmake_minimum_required(VERSION 3.0)
# 设置项目信息（名称、版本等）：
project(CMakeDemo)
# 创建供用户选择的可选项：
option(BUILD_SHARED_LIBS "Create SHARED libraries if ON." ON)
# 添加头文件搜索路径：
include_directories("${PROJECT_SOURCE_DIR}/include")
# 添加子目录：
add_subdirectory(src)
add_subdirectory(test)
