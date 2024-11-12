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

# [梯形求积](./openmp/trapezoid.cpp)

```cpp
void thread_critical(F f, double a, double b, int n_global, double *riemann_sum_global_ptr) {
  auto riemann_sum_local = /* ... */;

# pragma omp critical
  *riemann_sum_global_ptr += riemann_sum_local;
}


double parallel(F f, double a, double b, int n, int n_thread) {
  auto riemann_sum_global = 0.0;

# pragma omp parallel num_threads(n_thread)
  thread_critical(f, a, b, n, &riemann_sum_global);

  return riemann_sum_global;
}
```
