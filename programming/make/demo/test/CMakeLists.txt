# 添加构建可执行文件的目标:
add_executable(test_math math.c)
set_target_properties(test_math PROPERTIES OUTPUT_NAME math)
# 设置链接选项和依赖库:
target_link_libraries(test_math lib_math)
