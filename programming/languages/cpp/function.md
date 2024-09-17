---
title: 函数
---

# 普通函数

# 函数指针

# 可调用对象
**可调用对象 (callable objects)** 是对函数的进一步抽象，包括[普通函数](#普通函数)、[函数指针](#函数指针)、[函数对象](#函数对象)、[lambda](#lambda)。

## 函数对象
支持 [`operator()`](./operator#函数调用运算符) 的对象被称为**函数对象 (function object)** 或**函子 (functor)**。

### `std::greater`

标准库以[模板类](./generic#模板类)的形式在头文件 [`<functional>`](https://en.cppreference.com/w/cpp/utility/functional) 中为[算术运算符](./operator.md#算术运算符)、[关系运算符](./operator.md#关系运算符)、逻辑运算符、[位运算符](./operator.md#位运算符)定义了相应的函数对象类型。

`std::sort()` 默认按*升序*排列对象，即用 `operator<` 比较对象。
如果要按『降序』排列对象，可以为其提供一个相当于 `operator>` 的 `std::greater<T>` 对象：

```cpp
#include <cassert>
#include <algorithm>
#include <functional>
#include <vector>

int main() {
  std::vector<int> ints = { 1, 2, 3 };
  std::sort(ints.begin(), ints.end(), std::greater<int>());
  assert(ints[0] == 3);
  assert(ints[1] == 2);
  assert(ints[2] == 1);
}
```

## lambda

C++11 引入了 lambda 机制。
每个 lambda 表达式都定义了一种匿名类型，并生成了该类型的一个实例。
该实例支持 [`operator()`](./operator#函数调用运算符)，因此是一个[函数对象](#函数对象)。

```cpp
struct Point {
  int x, y;
  int norm2() const { return x*x + y*y; }
};
// 自 C++14 起，形参类型也可以用 `auto`：
auto cmp = [](auto const &p, auto const &q) {
  return left.norm2() > right.norm2();
};  // `;` 不可省略
// `norm2()` 值最小的 `Point` 位于顶部：
auto min_pq = priority_queue<Point, vector<Point>, decltype(cmp)>(cmp);
```

## `std::function`

定义在头文件 [`<functional>`](https://en.cppreference.com/w/cpp/utility/functional) 中的模板类 `std::function` 为各种[可调用对象](#可调用对象)提供了统一的**包装 (wrapper)**：

```cpp
#include <functional>
#include <map>

int add(int i, int j) { return i + j; }  // 普通函数
auto mod = [](int i, int j) { return i % j; };  // 命名的 lambda
struct Divide {
  int operator()(int denominator, int divisor) { 
    return denominator / divisor;
  }
};
auto divide = Divide();  // 自定义函数对象

int main() {
  std::map<char, std::function<int(int, int)>> binops;
  binops['+'] = add;  // 函数指针
  binops['-'] = std::minus<int>();  // 标准库函数对象
  binops['*'] = [](int i, int j) { return i * j; };  // 匿名的 lambda
  binops['/'] = divide;  // 自定义函数对象
  binops['%'] = mod;  // 命名的 lambda
  // 无差别地调用
  binops['+'](10, 5);
  binops['-'](10, 5);
  binops['/'](10, 5);
  binops['*'](10, 5);
  binops['%'](10, 5);
}
```

# [Coroutines](https://en.cppreference.com/w/cpp/language/coroutines) --- 允许挂起、恢复执行状态的函数

## `co_await` --- 暂停执行、直到被恢复

## `co_yield` --- 暂停执行、返回当前值

```c
#include <iostream>
#include <generator>  // C++23

/**
 * @brief 生成（无穷多项）Fibonacci 序列
 */
std::generator<int> fib() {
  int a = 0, b = 1;
  while (true) {
    co_yield b;  // 保存执行状态、返回当前值、等待下次调用
    auto c = a + b;
    a = b;
    b = c;
  }
}

/**
 * @brief 打印 Fibonacci 序列的前 n 项
 */
void printFib(int n) {
  int i = 1;
  for (int x : fib()) {
    std::printf("fib(%d) = %d\n", i, x);
    if (i++ == n) {
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  int n = std::atoi(argv[1]);
  printFib(n);
}
```

```shell
> g++ -std=c++23 -O2 main.cpp && ./a.out 10
fib(1) = 1
fib(2) = 1
fib(3) = 2
fib(4) = 3
fib(5) = 5
fib(6) = 8
fib(7) = 13
fib(8) = 21
fib(9) = 34
fib(10) = 55
```

## `co_return` --- 结束执行、返回当前值
