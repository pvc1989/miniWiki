---
title: 多线程并发
---

# <a href id="thread"></a>[`<thread>`](https://en.cppreference.com/w/cpp/header/thread) --- 支持多核并发的轻量级进程

构造 `thread` 对象时，需传入可调用的 task，如函数、函数对象等：

```cpp
template< class F, class... Args >
explicit thread( F&& f, Args&&... args );
// 若 Args 中含 T &，为避免其被推断为 T，需借助 std::ref 或 std::cref
```

析构前，需显式调用以下二者之一：
- `t.join()` 以阻塞调用线程，等待线程 `t` 执行完成；此后可使用其结果。
- `t.detach()` 以允许线程 `t` 独立地执行；此后再调用 `t.join()` 会抛出异常。

C++20 引入 [`std::jthread`](https://en.cppreference.com/w/cpp/thread/jthread)，其析构函数自带 `join()`。

典型用例：

```cpp
#include <algorithm>
#include <functional>  // std::cref
#include <iostream>
#include <thread>
#include <vector>

void f(std::vector<int> const &v, int *res) {
  *res = *std::max_element(v.begin(), v.end());
  std::cout << "v.max = " << *res << "; ";
}

struct F {
  std::vector<int> const &v_;
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
  auto v = std::vector<int>{ 1, 2, 3, 4, 5, 6, 7, 8 };  // 只读共享数据
  int x1, x2;

  {
    auto t1 = std::jthread{f, std::cref(v), &x1};  // f(v, &x1) 在 t1 中执行
  }  // 退出作用域时 ~jthread() 自带 join()

  auto t2 = std::thread{F{v, &x2}};  // F{v, &x2}() 在 t2 中执行
  t2.join();  // 等待 t2 执行完成，并回收资源

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

# <a href id="mutex"></a>[`<mutex>`](https://en.cppreference.com/w/cpp/header/mutex) --- 保护共享资源免于竞争的互斥机制

`std::mutex` 提供*互斥的*、*非递归的*所有权机制。假设 `mtx` 为某一 `std::mutex` 对象：
- 一个 `std::thread` 从其成功调用 `mtx.lock()` 或 `mtx.try_lock()` 起、至其调用 `mtx.unlock()` 为止，获得 `mtx` 的所有权。
- 当 `mtx` 被某一 `std::thread` 占有时，若其他 `std::thread`s 尝试
  - 调用 `mtx.lock()`，则被阻塞；
  - 调用 `mtx.try_lock()`，则获得返回值 `false`。
- 若某一 `std::mutex` 被析构时仍被某一 `std::thread` 占有，或某一 `std::thread` 终止时仍占有 某一 `std::mutex`，则行为未定义。

`std::mutex` 通常不被直接访问，而是配合以下机制使用：
- [`std::lock`](https://en.cppreference.com/w/cpp/thread/lock) 支持支持同时获得多个 [Lockable](https://en.cppreference.com/w/cpp/named_req/Lockable) 对象的所有权且避免死锁。
- [`std::lock_guard`](https://en.cppreference.com/w/cpp/thread/lock_guard) 以 RAII 的方式获得某个 `mtx` 的所有权，构造时调用 `mtx.lock()`，析构时调用 `mtx.unlock()`。
- [`std::scoped_lock`](https://en.cppreference.com/w/cpp/thread/scoped_lock) (C++17) 与 `std::lock_guard` 类似，但支持同时获得多个 `std::mutex`es 的所有权且避免死锁。
- [`std::unique_lock`](https://en.cppreference.com/w/cpp/thread/unique_lock) 与 `std::lock_guard` 类似，但其[构造函数](https://en.cppreference.com/w/cpp/thread/unique_lock/unique_lock)支持附加实参：
  - `std::defer_lock_t` 不获取所有权，待后续调用 `lock()`, `try_lock()`, `try_lock_for()`, `try_lock_until()` 之一；
  - `std::try_to_lock_t` 非阻塞地尝试获取所有权，相当于构造时调用 `mtx.try_lock()`；
  - `std::adopt_lock_t` 假设已获得所有权（否则其行为未定义），通常用于 RAII。
- [`std::shared_lock`](https://en.cppreference.com/w/cpp/thread/shared_lock) (C++14/17) 与 `std::unique_lock` 类似，但要求支持 `mtx.lock_shared()` 及 `mtx.try_lock_shared()`。

## `unique_lock<>`

```cpp
#include <algorithm>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

std::mutex mtx;

void f(std::vector<int> const &v, int *res) {
  *res = *std::max_element(v.begin(), v.end());
  auto ul = std::unique_lock<std::mutex>{mtx};  // 隐式调用 mtx.lock()
  std::cout << "v.max = " << *res << std::endl;
}  // 隐式调用 mtx.unlock()

struct F {
  std::vector<int> const &v_;
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
  {
    auto t1 = std::jthread{f, std::cref(v), &x1};
    auto t2 = std::jthread{F{v2, &x2}};
  }
}
```

## [`<shared_mutex>`](https://en.cppreference.com/w/cpp/header/shared_mutex) (C++14/17)

- 若某一线程已获得*互斥的*所有权，则其他线程既不能获得*互斥的*所有权，也不能获得*共享的*所有权；
- 若某一线程已获得*共享的*所有权，则其他线程仍不能获得*互斥的*所有权，但可以获得*共享的*所有权。

## Reader--Writer

【典型用例】[读写锁](../../csapp/12_concurrent_programming.md#读者作者)：

```cpp
#include <shared_mutex>

std::shared_mutex mtx;  // 可共享的 mutex (C++17)

void read() {
  /* 同一时间允许多个 threads 调用 read() */
  std::shared_lock<std::shared_mutex> lck {mtx};
  // ...
}

void write() {
  /* 同一时间只允许一个 thread 调用 write() */
  std::unique_lock<std::shared_mutex> lck {mtx};
  // ...
}
```

# [`<condition_variable>`](https://en.cppreference.com/w/cpp/thread/condition_variable) --- 基于等待唤醒的线程间通信机制

`std::condition_variable` 主要提供两组操作
- 【等待】包括 `wait()`, `wait_for()`, `wait_until()`
- 【唤醒】包括 `notify_one()`, `notify_all()`

其中 `wait(lck)` 用于阻塞当前线程，直到被其他线程 `notify_one()` 或 `notify_all()` 唤醒。
等待过程中，会调用 `lck.unlock()`，以允许其他线程访问资源；唤醒后会调用 `lck.lock()`，以禁止其他线程访问资源。
为避免***虚假唤醒 (spurious wakeup)***，可传入唤醒条件，即 `wait(lck, pred)`，相当于

```cpp
while (!pred()) {
  wait(lck);
}
```

## Consumer--Producer

【典型用例】[生产--消费](../../csapp/12_concurrent_programming.md#生产者消费者)问题：

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

`wait()` 只接受 `unique_lock` 作为第一个实参。

```c++
void Consume() {
  while (true) {
    auto lck = std::unique_lock<std::mutex>{msg_mutex};
    msg_cond.wait(lck,
        /* 唤醒条件： */[]() { return !msg_queue.empty(); });
    // relock upon wakeup
    auto msg = msg_queue.front();
    msg_queue.pop();
    lck.unlock();
    // parse msg ...
  }
}

void Produce() {
  while (true) {
    auto msg = Message{/* ... */};
    auto lck = std::scoped_lock<std::mutex>{msg_mutex};
    msg_queue.push(msg);
    msg_cond.notify_one();
  }
}
```

其中
- `Consume` 中只能用 `unique_lock`，因为要将其传递给 `wait()`。
- `Produce` 中可以用 `scoped_lock`，因为只需获得互斥访问权限。

# [`<semaphore>`](https://en.cppreference.com/w/cpp/header/semaphore) --- 限制访问数量的轻量级同步机制

# [`<future>`](https://en.cppreference.com/w/cpp/header/future) --- 基于共享状态的异步任务机制

## `future<>` and `promise<>`

支持在一对任务间通过共享状态（免锁地）传递数据。

典型用例：

```cpp
void Get(std::future<X> &fx) {  // 获取 X-型 值
  try {
    auto x = fx.get();  // 可能需要等待 px.set_value() 完成
    // use x ...
  } catch (...) {
    // handle the exception ...
  }
}

void Put(std::promise<X> &px) {  // 设置 X-型 值
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

## `packaged_task<>`

提供对可调用对象（返回 `X` 型对象）的封装，以简化 `future<X>`/`promise<X>` 调用。
其 `get_future()` 方法返回其对应的 `future<X>`。

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
  auto f0 = pt0.get_future();  // 返回 future<double>
  auto f1 = pt1.get_future();  // 返回 future<double>
  auto *head = &v[0];
  auto *half = head + v.size() / 2;
  auto *tail = head + v.size();
  auto t1 = std::thread{std::move(pt0), head, half, 0.0};
  auto t2 = std::thread{std::move(pt1), half, tail, 0.0};
  t1.join(); t2.join();
  std::cout << f0.get() + f1.get() << std::endl;
}
```

## `async()`

避免显式创建线程的任务级并发机制，由系统在运行期决定创建多少线程。

- 【入参】返回 `X` 型对象的可调用对象（及其入参）
- 【返回值】`future<X>` 型对象，用 `get()` 获取结果

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
  // 异步（非阻塞）地执行 parallel_sum(mid, end)
  int sum = parallel_sum(beg, mid);
  return sum + future.get();
}

int main() {
  std::vector<int> v(10000, 1);
  std::cout << "The sum is " << parallel_sum(v.begin(), v.end()) << '\n';
}
```

# [`<atomic>`](https://en.cppreference.com/w/cpp/header/atomic) --- 支持免锁互斥的细粒度操作

## `volatile`

关键词 `volatile` 用于禁止编译器对内存访问的优化，适用于可能被外部因素（如：传感器、计时器）修改的变量：

```cpp
volatile const long clock_register;  // updated by the hardware clock
auto t1 {clock_register};
// ... no use of clock_register here ...
auto t2 {clock_register};  // 不保证 t1 == t2
```

