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
