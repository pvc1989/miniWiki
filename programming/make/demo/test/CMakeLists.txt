# 添加构建可执行文件的目标:
add_executable(test_math test_math.c)
# 设置链接选项和依赖库:
target_link_libraries(test_math math)
