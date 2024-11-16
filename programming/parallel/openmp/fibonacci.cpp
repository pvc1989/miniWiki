// c++ --std=c++20 -g -Wall -fopenmp -o fibonacci fibonacci.cpp
// ./fibonacci 40

#include <cstdio>
#include <chrono>
#include <vector>

#include <omp.h>

int serial_by_recursion(int n) {
  return n < 2 ? n : serial_by_recursion(n - 1) + serial_by_recursion(n - 2);
}

int parallel_by_tasks(int n, std::vector<int> *dp) {
  if (n < 2) {
    return dp->at(n) = n;
  }

  int i = 0, j = 0;
# pragma omp task shared(i) if(n > 30)
  i = parallel_by_tasks(n - 1, dp);  // executed by a subtask

  j = parallel_by_tasks(n - 2, dp);  // executed by the current task

# pragma omp taskwait
  dp->at(n) = i + j;  // use the subtask's result j

  return dp->at(n);
}

int main(int argc, char *argv[]) {
  int n = std::atoi(argv[1]);
  int value;

  using Clock = std::chrono::high_resolution_clock;
  std::chrono::time_point<Clock> start, stop;
  std::chrono::milliseconds cost;

  start = Clock::now();
  value = serial_by_recursion(n);
  stop = Clock::now();
  cost = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::printf("serial_by_recursion() = %d, costs %ld milliseconds\n", value, cost.count());

  start = Clock::now();
  auto dp = std::vector<int>(n + 1);
  value = parallel_by_tasks(n, &dp);
  stop = Clock::now();
  cost = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::printf("parallel_by_tasks() = %d, costs %ld milliseconds\n", value, cost.count());
}
