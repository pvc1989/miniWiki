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

可以在 `#pragma` 里，用 `private(var)` 将默认*共享的*变量 `var` 改为*私有的*，此时需在各线程内部对其初始化。

建议在 `#pragma` 里，用 `default(none)` 取消默认行为，强制要求用 `private(var)` 或 `shared(var)` 或 `reduction(+: var)` 显式指定变量的作用域。

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

# `parallel for`

- 必须是 `for` 循环。
- 循环变量 `i` 必须是整型，且为各线程独有。
- 循环总数必须能确定，不能有无限循环、循环体内不能有 `break` 或 `return`、循环体内修改起、止、步长等情况。

即使编译通过，运行期仍有可能出错：
如 dynamic programming 存在*数据依赖*，用 `parallel for` 可能发生*数据竞争*。

```cpp
double parallel_for(F f, double a, double b, int n, int n_thread) {
  int i;  // 定义在外，但作为循环变量，为各线程独有
  auto h = (b - a) / n;
  auto riemann_sum = (f(a) + (b)) * h / 2;
# pragma omp parallel for num_threads(n_thread) \
      reduction(+: riemann_sum)
  for (int i = 1; i < n; ++i) {
    riemann_sum += f(a + i * h) * h;
  }
  return riemann_sum;
}
```
