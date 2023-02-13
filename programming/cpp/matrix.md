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

# [LAPACK](https://netlib.org/lapack)

> LAPACK is written in Fortran 90 and provides routines for solving systems of simultaneous linear equations, least-squares solutions of linear systems of equations, eigenvalue problems, and singular value problems.

## [LAPACK Users' Guide](https://netlib.org/lapack/lug/)

The subroutines in LAPACK are classified as follows:
- [driver](https://netlib.org/lapack/lug/node25.html#secdrivers) routines: each of which solves a complete problem.
  - [Linear Equations](https://netlib.org/lapack/lug/node26.html):
    - prefix: `S` for single real, `D` for double real, `C` for single complex, `Z` for double complex, respectively.
    - middle: `GE` for general, `GB` for general banded, `GT` for general tridiagonal; `PO` for symmetric/Hermitian, `PO` for positive definite, ...
    - suffix: `SV` for a simple driver, `SVX` for an expert driver.
  - [Standard Eigenvalue and Singular Value Problems](https://netlib.org/lapack/lug/node29.html)
- [computational](https://netlib.org/lapack/lug/node37.html#seccomp) routines: each of which performs a distinct computational task.

## [The LAPACKE C Interface to LAPACK](https://netlib.org/lapack/lapacke.html)

- Variables with the FORTRAN type `integer` are converted to `lapack_int` in LAPACKE.
- FORTRAN logicals are converted to `lapack_logical`, which is defined as `lapack_int`.
- All the LAPACKE routines that take one or more 2D arrays as a pointer, not as a pointer to pointers.
- LAPACKE function name is `LAPACKE_xbase` or `LAPACKE_xbase_work` where `x` is the type:
  - `s` or `d` for single or double precision real,
  - `c` or `z` for single or double precision complex,
  - with `base` representing the LAPACK base name.
