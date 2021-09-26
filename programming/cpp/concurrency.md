---
title: 多线程并发
---

# `<thread>`

```cpp
#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>

std::mutex mtx;

void f(std::vector<int> const &v, int *res) {
  *res = *std::max_element(v.begin(), v.end());
  std::cout << "v.max = " << *res << "; ";
}

struct F {
  std::vector<int> v_;
  int *res_;

  F(std::vector<int> const &v, int *res)
      : v_(v), res_(res) {
  }
  void operator()() {
    *res_ = *std::min_element(v_.begin(), v_.end());
    std::cout << "v.min = " << *res_ << "; ";
  }
};

int main() {
  auto v1 = std::vector<int>{ 1, 2, 3, 4, 5, 6, 7, 8 };
  auto v2 = std::vector<int>{ 1, 2, 3, 4, 5, 6, 7, 8 };
  int x1, x2;
  auto t1 = std::thread{f, v1, &x1};  //     f() executes in thread-1
  auto t2 = std::thread{F{v2, &x2}};  // F{v2}() executes in thread-2
  t1.join();  // wait for t1 to exit
  t2.join();  // wait for t2 to exit
  std::cout << std::endl;
}
```

其中 `std::cout` 是共享资源，故运行结果可能出错：

- 正确结果：
  ```
  v.max = 8; v.min = 1; 
  v.min = 1; v.max = 8; 
  ```
- 错误结果：
  ```
  v.max = v.min = 81; ; 
  v.min = v.max = 18; ; 
  v.max = v.min = 8; 1; 
  v.min = v.max = 1; 8; 
  ...
  ```

# `<mutex>`

## `unique_lock`

```cpp
#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

std::mutex mtx;

void f(std::vector<int> const &v, int *res) {
  *res = *std::max_element(v.begin(), v.end());
  auto ul = std::unique_lock<std::mutex>{mtx};
  std::cout << "v.max = " << *res << std::endl;
}

struct F {
  std::vector<int> v_;
  int *res_;

  F(std::vector<int> const &v, int *res)
      : v_(v), res_(res) {
  }
  void operator()() {
    *res_ = *std::min_element(v_.begin(), v_.end());
    auto ul = std::unique_lock<std::mutex>{mtx};
    std::cout << "v.min = " << *res_ << std::endl;
  }
};

int main() {
  auto v1 = std::vector<int>{ 1, 2, 3, 4, 5, 6, 7, 8 };
  auto v2 = std::vector<int>{ 9, 10, 11, 12, 13, 14 };
  int x1, x2;
  auto t1 = std::thread{f, v1, &x1};  //     f() executes in thread-1
  auto t2 = std::thread{F{v2, &x2}};  // F{v2}() executes in thread-2
  t1.join();  // wait for t1 to exit
  t2.join();  // wait for t2 to exit
}
```

## `defer_lock`

```c++
#include <mutex>

std::mutex m1, m2;

int main() {
  auto ul1 = std::unique_lock<std::mutex>{m1, std::defer_lock};
  auto ul2 = std::unique_lock<std::mutex>{m2, std::defer_lock};
  // ...
  std::lock(ul1, ul2);
  // ...
}
```

# `<condition_variable>`

```c++
#include <condition_variable>
#include <mutex>
#include <queue>

class Message {
  // ...
};

std::queue<Message>     msg_queue;
std::condition_variable msg_cond;
std::mutex              msg_mutex;
```

## `wait`

```c++
void Consume() {
  while (true) {
    auto ul = std::unique_lock<std::mutex>{msg_mutex};
    while (msg_cond.wait(ul))  // release `ul` until `!msg_queue.empty()`
      /* do nothing */;
    // relock upon wakeup
    auto msg = msg_queue.front();
    msg_queue.pop();
    ul.unlock();
    // parse msg ...
  }
}
```

## `notify_one`

```c++
void Produce() {
  while (true) {
    auto msg = Message{/* ... */};
    auto ul = std::unique_lock<std::mutex>{msg_mutex};
    msg_queue.push(msg);
    msg_cond.notify_one();
  }
}
```

# `<future>`

## `promise`

```cpp
void Put(std::promise<X> &px) {  // a task: put the result in px
  try {
    auto res = X(/* ... */);
    // may throw an exception
    px.set_value(res);
  } catch (...) {
    // pass the exception to the future's thread:
    px.set_exception(std::current_execption());
  }
}
```

## `future`

```cpp
void Get(std::future<X> &fx) {  // a task: get the result from fx
  try {
    auto x = fx.get();  // if necessary, wait for the value to get computed
    // use x ...
  } catch (...) {
    // handle the exception ...
  }
}
```

## `packaged_task`

```cpp
#include <future>
#include <iostream>
#include <numeric>
#include <vector>
#include <thread>

double accum(double const *beg, double const *end, double init) {
  return std::accumulate(beg, end, init);
}

int main() {
  auto v = std::vector<double>{ 0.1, 0.2, 0.3, 0.4, 0.5 };
  using Task = decltype(accum);
  auto pt0 = std::packaged_task<Task>{accum};
  auto pt1 = std::packaged_task<Task>{accum};
  auto f0 = std::future<double>{pt0.get_future()};
  auto f1 = std::future<double>{pt1.get_future()};
  auto *head = &v[0];
  auto *half = head + v.size() / 2;
  auto *tail = head + v.size();
  auto t1 = std::thread{std::move(pt0), head, half, 0.0};
  auto t2 = std::thread{std::move(pt1), half, tail, 0.0};
  t1.join(); t2.join();
  std::cout << f0.get() + f1.get() << std::endl;
}
```

## `async`

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <future>

template <typename RandomIt>
int parallel_sum(RandomIt beg, RandomIt end) {
  auto len = end - beg;
  if (len < 1000)
    return std::accumulate(beg, end, 0);

  RandomIt mid = beg + len/2;
  auto future = std::async(
      /* std::launch::async, */parallel_sum<RandomIt>, mid, end);
  int sum = parallel_sum(beg, mid);
  return sum + future.get();
}

int main() {
  std::vector<int> v(10000, 1);
  std::cout << "The sum is " << parallel_sum(v.begin(), v.end()) << '\n';
}
```


# `<atomic>`

## `volatile`

关键词 `volatile` 用于禁止编译器对内存访问的优化，适用于可能被外部因素（如：传感器、计时器）修改的变量：

```cpp
volatile const long clock_register;  // updated by the hardware clock
auto t1 {clock_register};
// ... no use of clock_register here ...
auto t2 {clock_register};  // 不保证 t1 == t2
```

