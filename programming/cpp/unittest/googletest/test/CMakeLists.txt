add_executable(test_math math.cpp)
set_target_properties(test_math PROPERTIES OUTPUT_NAME math)
target_link_libraries(test_math lib_math gtest_main)
add_test(NAME TestMath COMMAND test_math)
