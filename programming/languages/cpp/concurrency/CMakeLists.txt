cmake_minimum_required(VERSION 3.14)
project(test_future)

set (cases
  future
)

foreach (case ${cases})
  add_executable(test_${case} test_${case}.cpp)
  install(TARGETS test_${case} DESTINATION bin)
endforeach (case ${cases})
