#include <cassert>
#include <cmath>
#include <future>
#include <iostream>
#include <numeric>
#include <vector>
#include <thread>
#include <chrono>


inline double simple_op(double x) {
  double value = 0.0;
  for (int i = 0; i < 1024; i++) {
    value += x * x;
  }
  return value;
}

double slow_task(std::vector<int> const &values, int i_task, int n_task) {
  auto n_global = values.size();
  auto n_local = (n_global + n_task - 1) / n_task;
  auto first = n_local * i_task;
  auto last = n_local * (i_task + 1);
  last = std::min(last, n_global);
  if (i_task + 1 == n_task) {
    n_local = last - first;
    assert(last == n_global);
  }
  double sum = 0.0;
  for (auto i = first; i < last; ++i) {
    sum += simple_op(values[i]);
  }
  return sum;
}

double thread_based(std::vector<int> const &values, int n_task) {
  using Task = decltype(slow_task);
  auto futures = std::vector<std::future<double>>();
  auto threads = std::vector<std::thread>();
  // 1 task / 1 thread, oversubscription or thread exhaustion might happen:
  for (int i_task = 0; i_task < n_task; i_task++) {
    auto task = std::packaged_task<Task>(slow_task);
    futures.emplace_back(task.get_future());
    threads.emplace_back(std::move(task), values, i_task, n_task);
  }
  double sum = 0.0;
  for (int i_task = 0; i_task < n_task; i_task++) {
    threads[i_task].join();
    sum += futures[i_task].get();
  }
  return sum;
}

double async_based(std::vector<int> const &values, int n_task) {
  using Task = decltype(slow_task);
  auto futures = std::vector<std::future<double>>();
  for (int i_task = 0; i_task < n_task; i_task++) {
    futures.emplace_back(std::async(std::launch::async,
        slow_task, values, i_task, n_task));
  }
  double sum = 0.0;
  for (auto &future : futures) {
    sum += future.get();
  }
  return sum;
}

int main(int argc, char *argv[]) {
  int n_core = std::thread::hardware_concurrency();
  std::cout << n_core << " concurrent threads are supported.\n\n";

  int n_global = std::atoll(argv[1]);
  auto values = std::vector<int>(n_global);
  std::iota(values.begin(), values.end(), 0);

  int n_task = std::atoi(argv[2]);
  int n_worker = std::min(n_task, n_core);

  using clock = std::chrono::high_resolution_clock;

  auto start = clock::now();
  auto sum = 0.0;
  for (auto i = n_global - n_global; i < n_global; i++) {
    sum += simple_op(values[i]);
  }
  auto stop = clock::now();
  std::chrono::duration<double> cost_serial = stop - start;
  std::cout << "by single thread: sum = " << sum
      << ", cost = " << cost_serial.count() << " seconds\n\n";

  start = clock::now();
  sum = thread_based(values, n_task);
  stop = clock::now();
  std::chrono::duration<double> cost_conc = stop - start;
  std::cout << "by multi-threads: sum = " << sum
      << ", cost = " << cost_conc.count() << " seconds\n";
  auto speed_up = cost_serial.count() / cost_conc.count();
  std::cout << "speed up = " << speed_up
      << ", efficiency = " << speed_up / n_worker << "\n\n";

  start = clock::now();
  sum = async_based(values, n_task);
  stop = clock::now();
  std::chrono::duration<double> cost_async = stop - start;
  std::cout << "by std::async: sum = " << sum
      << ", cost = " << cost_async.count() << " seconds\n";
  speed_up = cost_serial.count() / cost_async.count();
  std::cout << "speed up = " << speed_up
      << ", efficiency = " << speed_up / n_worker << "\n\n";

  return 0;
}
