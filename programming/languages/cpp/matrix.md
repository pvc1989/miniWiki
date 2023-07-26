---
title: 矩阵计算
---

# Eigen

```cpp
/*
build:
  g++ -std=c++14 -I$EIGEN_INC -O2 -o demo demo.cpp
run:
  ./solve 400
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
  std::cout << std::chrono::duration<double>(t_end-t_start).count()
      << " s" << std::endl;
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

```cpp
/*
build:
  g++ -std=c++20 -O2 -I$LAPACKE_INC -L$LAPACKE_LIB -llapacke -o demo demo.cpp
run:
  ./demo 400
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>  // std::inner_product
#include <random>
#include <vector>
#include <lapacke.h>

double my_rand() {
  static std::uniform_real_distribution<double> distr(0.0, 1.0);
  static std::random_device device;
  static std::mt19937 engine{device()};
  return distr(engine);
}

int main (int argc, const char *argv[]) {
  int n = std::atoi(argv[1]);
  // build a random matrix and a random vector
  auto a = std::vector<double>(n * n);
  std::ranges::generate(a, my_rand);
  auto b = std::vector<double>(n);
  std::ranges::generate(b, my_rand);
  // Call DGESV to solve A * X = B for a general square matrix A.
  lapack_int info;
  auto const a_copy = a;  // A will be overwritten by its L and U.
  auto const b_copy = b;  // B will be overwritten by (A \ B).
  auto pivot = std::vector<lapack_int>(n);  /* The pivot indices 
      that define the permutation matrix P;
      row i of the matrix was interchanged with row pivot(i). */
  auto t_start = std::chrono::high_resolution_clock::now();
  info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n/* number of rows of A */,
    1/* NRHS := the number of columns of B and X */,
    a.data(), n/* leading dimension of a */,
    pivot.data(), b.data(), 1/* leading dimension of b */);
  auto t_end = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration<double>(t_end-t_start).count()
      << " s" << std::endl;
  if (info) {
    return info;
  }
  // build the residual vector and print its norm
  auto const &x = b;
  auto r = b_copy;
  for (int i = 0; i < n; ++i) {
    auto row_i = a_copy.begin() + i * n;
    r[i] -= std::inner_product(x.begin(), x.end(), row_i, 0.0);
  }
  std::cout << std::sqrt(std::inner_product(r.begin(), r.end(),
      r.begin(), 0.0)) << std::endl;
  return(info);
}
```
