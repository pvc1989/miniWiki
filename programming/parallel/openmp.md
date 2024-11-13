---
title: OpenMP
---

# [Hello](./openmp/hello.cpp)

```cpp
// ...
#include <omp.h>

// the thread function
void hello() {
  int rank = omp_get_thread_num();
  int size = omp_get_num_threads();
  // ...
}

int main(int argc, char *argv[]) {
  int n_thread = std::atoi(argv[1]);

# pragma omp parallel num_threads(n_thread)
  hello();

  return 0;
}
```

# [Critical](./openmp/trapezoid.cpp)

```cpp
double thread_by_return(F f, double a, double b, int n_global);

void thread_by_pointer(F f, double a, double b, int n_global, double *riemann_sum_global_ptr) {
  auto riemann_sum_local = thread_by_return(f, a, b, n_global);

# pragma omp critical
  *riemann_sum_global_ptr += riemann_sum_local;
}

double parallel_by_pointer(F f, double a, double b, int n, int n_thread) {
  auto riemann_sum_global = 0.0;

# pragma omp parallel num_threads(n_thread)
  thread_by_pointer(f, a, b, n, &riemann_sum_global);
 
  return riemann_sum_global;
}
```

# Scope

默认地

- `parallel`-block *前*定义的变量，被所有线程*共享*。 
- `parallel`-block *内*定义的变量，被当前线程*独有*。 

```cpp
double parallel_by_return(F f, double a, double b, int n, int n_thread) {
  double riemann_sum_global = 0.0;  // shared
# pragma omp parallel num_threads(n_thread)
  {
    double riemann_sum_local/* private */
        = thread_by_return(f, a, b, n)/* f, a, b, n are shared, too */;
#   pragma omp critical
    riemann_sum_global += riemann_sum_local;
  }
  return riemann_sum_global;
}
```

# Reduction

```cpp
double parallel_by_reduction(F f, double a, double b, int n, int n_thread) {
  double riemann_sum_global = 0.0;
  # pragma omp parallel num_threads(n_thread) \
        reduction(+/* 规约运算 */: riemann_sum_global/* 规约变量 */)
  {
    riemann_sum_global += thread_by_return(f, a, b, n);
  }
  return riemann_sum_global;
}
```

相当于 OpenMP 自动创建了私有变量 `riemann_sum_local` 及 `critical`-block。
