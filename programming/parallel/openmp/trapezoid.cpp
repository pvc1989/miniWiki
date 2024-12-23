// c++ --std=c++20 -g -Wall -fopenmp -o trapezoid trapezoid.cpp
// ./trapezoid 1000 16

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

#include <omp.h>

double f(double x) {
  return std::sqrt(1 - x * x);
}

using F = decltype(f);

double serial_for(F f, double a, double b, int n) {
  auto h = (b - a) / n;
  auto riemann_sum = (f(a) + (b)) * h / 2;
  for (int i = 1; i < n; ++i) {
    riemann_sum += f(a + i * h) * h;
  }
  return riemann_sum;
}

double parallel_for(F f, double a, double b, int n, int n_thread) {
  auto h = (b - a) / n;
  auto riemann_sum = (f(a) + (b)) * h / 2;
# pragma omp parallel for num_threads(n_thread) \
      reduction(+: riemann_sum)
  for (int i = 1; i < n; ++i) {
    riemann_sum += f(a + i * h) * h;
  }
  return riemann_sum;
}

inline std::pair<int, int> get_range(int i_thread, int n_thread, int n_global) {
  auto n_local_upper = (n_global + n_thread - 1) / n_thread;
  auto n_local_lower = n_global / n_thread;
  auto n_upper = n_global % n_thread;

  int first, last;
  if (i_thread < n_upper) {
    first = n_local_upper * i_thread;
    last = first + n_local_upper;
  } else {
    first = n_local_upper * n_upper + n_local_lower * (i_thread - n_upper);
    last = first + n_local_lower;
  }
  assert(last <= n_global);
  assert(i_thread + 1 == n_thread ? last == n_global : true);
  return { first, last };
}

double thread_by_return(F f, double a, double b, int n_global) {
  int i_thread = omp_get_thread_num();
  int n_thread = omp_get_num_threads();

  auto [first, last] = get_range(i_thread, n_thread, n_global);
  auto n_local = last - first;

  if (!n_local) { return 0.0; }

  auto h = (b - a) / n_global;
  auto a_local = a + h * first;
  auto b_local = a + h * last;

  return serial_for(f, a_local, b_local, n_local);
}

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

double parallel_by_reduction(F f, double a, double b, int n, int n_thread) {
  double riemann_sum_global = 0.0;
  # pragma omp parallel num_threads(n_thread) \
        reduction(+/* 规约运算 */: riemann_sum_global/* 规约变量 */)
  {
    riemann_sum_global += thread_by_return(f, a, b, n);
  }
  return riemann_sum_global;
}

int main(int argc, char *argv[]) {
  int n_interval = std::atoi(argv[1]);
  int n_thread = std::atoi(argv[2]);

  auto pi = std::numbers::pi;
  std::printf("pi - serial_for() = %.2e\n",
      pi - 4 * serial_for(f, 0.0, 1.0, n_interval));
  std::printf("pi - parallel_for() = %.2e\n",
      pi - 4 * parallel_for(f, 0.0, 1.0, n_interval, n_thread));
  std::printf("pi - parallel_by_pointer() = %.2e\n",
      pi - 4 * parallel_by_pointer(f, 0.0, 1.0, n_interval, n_thread));
  std::printf("pi - parallel_by_return() = %.2e\n",
      pi - 4 * parallel_by_return(f, 0.0, 1.0, n_interval, n_thread));
  std::printf("pi - parallel_by_reduction() = %.2e\n",
      pi - 4 * parallel_by_reduction(f, 0.0, 1.0, n_interval, n_thread));

}
