add_executable(test_math test_math.cpp)
target_link_libraries(test_math math gtest_main)
add_test(NAME GoogleTestMath COMMAND test_math)
