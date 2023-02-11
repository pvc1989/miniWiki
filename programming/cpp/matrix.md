---
title: 矩阵计算
---

# Eigen

```cpp
/*
build:
  g++ -std=c++14 -I/usr/local/include/eigen3 -O2 -o demo demo.cpp
run:
  ./solve 4000
 */
#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

int main(int argc, char *argv[]) {
  using Float = double;
  using Matrix = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<Float, Eigen::Dynamic, 1>;
  int n = std::atoi(argv[1]);
  auto A = Matrix(n, n);
  A.setRandom();
  auto b = Vector(n);
  b.setRandom();
#ifdef C_STYLE_TIMING
  auto t_start = std::clock();
  Vector u = A.partialPivLu().solve(b);
  auto t_end = std::clock();
  std::cout << double(t_end-t_start) / CLOCKS_PER_SEC << " s" << std::endl;
#else
  auto t_start = std::chrono::high_resolution_clock::now();
  Vector u = A.partialPivLu().solve(b);
  auto t_end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration<double>(t_end-t_start).count() << " s" << std::endl;
#endif
  Vector r = A * u - b;
  std::cout << r.norm() << std::endl;
}
```
