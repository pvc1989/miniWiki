#include <cassert>
#include <cmath>
#include <future>
#include <iostream>
#include <numeric>
#include <vector>
#include <thread>
#include <chrono>


inline double single_flop(double x) {
  return x * x;
}

double slow_task(size_t n_global, int i_task, int n_task) {
  auto n_local = (n_global + n_task - 1) / n_task;
  auto first = n_local * i_task;
  auto last = n_local * (i_task + 1);
  last = std::min(last, n_global);
  if (i_task + 1 == n_task) {
    n_local = last - first;
    assert(last == n_global);
  }
  double sum = 0.0;
  for (auto curr = first; curr < last; ++curr) {
    sum += single_flop(curr);
  }
  return sum;
}


int main(int argc, char *argv[]) {
  auto n_global = std::atoll(argv[1]);
  int n_task = std::atoi(argv[2]);

  using clock = std::chrono::high_resolution_clock;

  auto start = clock::now();
  using Task = decltype(slow_task);
  auto futures = std::vector<std::future<double>>();
  auto threads = std::vector<std::thread>();
  // 1 task / 1 thread, oversubscription or thread exhaustion might happen:
  for (int i_task = 0; i_task < n_task; i_task++) {
    auto task = std::packaged_task<Task>(slow_task);
    futures.emplace_back(task.get_future());
    threads.emplace_back(std::move(task), n_global, i_task, n_task);
  }
  double sum = 0.0;
  for (int i_task = 0; i_task < n_task; i_task++) {
    threads[i_task].join();
    sum += futures[i_task].get();
  }
  auto stop = clock::now();
  std::chrono::duration<double> cost_conc = stop - start;
  std::cout << "by multi-threads: sum = " << sum
      << ", cost = " << cost_conc.count() << " seconds" << std::endl;

  start = clock::now();
  sum = 0.0;
  for (auto x = n_global - n_global; x < n_global; x++) {
    sum += single_flop(x);
  }
  stop = clock::now();
  std::chrono::duration<double> cost_serial = stop - start;
  std::cout << "by single thread: sum = " << sum
      << ", cost = " << cost_serial.count() << " seconds" << std::endl;

  auto speed_up = cost_serial.count() / cost_conc.count();
  std::cout << "speed up = " << speed_up << std::endl;

  int n_core = std::thread::hardware_concurrency();
  std::cout << n_core << " concurrent threads are supported.\n";
  std::cout << "efficiency = " << speed_up / std::min(n_task, n_core) << std::endl;

  return 0;
}
