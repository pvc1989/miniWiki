---
title: 异常
---

# 运行期错误及处置策略

## 运行期错误
**运行期 (run-time)** 错误是指程序在运行时可能会遇到的非正常状态。
这种错误往往由*运行期*产生的非法参数所导致，因此无法在**编译期 (compile-time)** 被发现和排除。

对于简单的错误，通常可以在本地进行**处置 (handle)**。
而对于如下更复杂、更一般的场景，就需要引入专门的**错误处置策略 (error-handling strategy)**：

- *作者*知道哪里会发生运行期错误，但不知道该如何去处置这种错误。
- *用户*知道该如何处置运行期错误，但不知道这种错误会在哪里发生。

⚠️ 对于新系统，应当尽早确定错误处置策略；对于现有系统，应当与现有的错误处置策略保持一致。

## 传统处置策略
在[异常机制](#异常机制)被引入 C++ 之前，业界已经形成了一些常用的错误处置策略。
这些策略是[异常机制](#异常机制)的替代项，但都有各自的缺点。

### 终止程序
调用 `exit()` 或 `abort()` 来终止（整个）程序。

主要缺点：用户可能无法承受程序终止所造成的影响。

### 返回错误码
返回一个表示错误信息的整数（或*枚举值*）。

主要缺点：用户可能忘记检测错误码。

### 设置状态变量
返回一个合法值，将某个用于记录状态的变量设为错误状态。

主要缺点：用户可能忘记检测状态变量。

### 调用处置函数
调用处置函数的抽象接口，由用户提供具体实现。

主要缺点：系统行为依赖于用户提供的具体实现。

## 异常机制

C++ 语言提供的**异常 (exception)** 机制是一种应用级[异常控制流](../csapp/8_exceptional_control_flow.md)，
- 只能用来处置由正在执行的指令引起的**同步事件 (synchronous events)**，如*数组越界*、*读写错误*等。
- 不能（直接）处置由其他原因引起的**异步事件 (asynchronous events)**，如*键盘中断*、*电源故障*等。

### 标准库异常类
任何可复制的对象都可以被用作异常。

标准库中用到的异常以定义在 [`<exception>`](https://en.cppreference.com/w/cpp/error/exception) 中的 `std::exception` 为公共基类。
除构造和析构函数外，它有两个公共方法成员：

- `operator=()` 用于异常对象的复制。
- `what()` 返回构造时传入的字符串。

标准库中用到的其他异常定义在 [`<stdexcept>`](https://en.cppreference.com/w/cpp/header/stdexcept) 等头文件中。
它们形成了一个继承体系（用缩进层次表示）：

- 常用的标准库异常类：
  ```cpp
  // <stdexcept>
  logic_error
    invalid_argument
    domain_error
    length_error
    out_of_range
    future_error (C++11)
  runtime_error
    range_error
    overflow_error
    underflow_error
    regex_error (C++11)
    nonexistent_local_time (C++20)
    ambiguous_local_time (C++20)
    tx_exception(TM TS)
    system_error (C++11)
      ios_base::failure (C++11)
      filesystem::filesystem_error (C++17)
  // <typeinfo>
  bad_typeid
  bad_cast
    bad_any_cast (C++17)
  // <new>
  bad_alloc
    bad_array_new_length (C++11)
  // <memory>
  bad_weak_ptr (C++11)
  ```
- 不太常用的标准库异常类：
  ```cpp
  bad_exception        // <exception>
  bad_function_call    // <functional> (C++11)
  ios_base::failure    // <ios> (until C++11)
  bad_variant_access   // <variant> (C++17)
  bad_optional_access  // <optional> (C++17)
  ```

### 抛出 (`throw`)
如果一个操作可能发生运行期错误，则应当用 `throw` 语句*抛出*一个异常：
```cpp
// array.h
#include <stdexcept>

template <int N>
class Array {
  int _a[N];
 public:
  int size() const noexcept {  // 不会抛出异常的操作应当用 noexcept 标识
    return N;
  }
  int& at(int i) {
    if (i < 0 or i >= N) {
      // 如果发生下标越界，则抛出一个 std::out_of_range 对象
      throw std::out_of_range("The given index is out of range.");
    }
    return _a[i];
  }
};
```

### 捕获 (`catch`)
用户应当将可能抛出的异常的操作置于 `try{}` 代码块中，并紧随其后用一个或多个 `catch` 子句进行*捕获*：
```cpp
#include <iostream>
#include "array.h"

int main() {
  auto anArray = Array<10>();
  try {
    for (int i = 0; i != anArray.size(); ++i) {
      anArray.at(i) = i;  // OK
    }
    anArray.at(anArray.size()) = anArray.size();  // 越界
  } catch (std::out_of_range& e) {  // 捕获越界异常
    std::cerr << e.what() << std::endl;
  } catch (...) {  // 捕获其他异常
    throw;
  }
  for (int i = 0; i != anArray.size(); ++i) {
    std::cout << anArray.at(i) << ' ';
  }
  std::cout << std::endl;
}
```
这里的两条 `catch` 子句体现了两种典型的处置策略：
- 对于 `std::out_of_range` 类型的异常，用 `what()` 方法打印提示信息。
- 对于其他类型的异常，用 `throw` 语句将其重新抛出，交由调用者捕获。

# 异常安全
## 概念
### 类不变量
每个类的定义中都隐含一个**类不变量 (class invariant)**，它表示这个类的对象在程序运行过程中所必须保持的性质。
对于每个对象，该不变量在构造函数运行后就被建立起来。
在析构函数运行前，所有访问该对象内部表示的操作都必须维护该不变量。

### 广义不变量
不变量的概念可以推广到多个对象之间。
在下面的例子中，*`x.size() == y.size()` 始终为真*就是一种广义不变量：

```cpp
struct Points {
  vector<int> x;
  vector<int> y;
};
```

### 有效状态

对于含有*类不变量*的某个对象（或含有*广义不变量*的一组对象），如果这种不变量在程序运行过程中没有被破坏，则称该对象（或这组对象）处于有效状态。

### 异常保证

标准库的每一个操作都至少满足以下三种**保证 (gaurantee)** 之一：
- 所有操作都满足**基本保证 (basic gaurantee)**：类不变量得到维护，没有资源泄露。
- 关键操作（例如 `std::vector::push_back()`）满足**强保证 (strong gaurantee)**：如果操作失败，则不产生影响。
- 简单操作（例如 `std::vector::pop_back()`）满足**无抛出保证 (nothrow gaurantee)**：该操作的实现不会抛出异常（用 `noexcept` 标识）。

*基本保证*和*强保证*都要求：

- 自定义操作不会造成资源泄露。
- 容器操作所依赖的自定义操作（`operator=()`、`swap()`）不会使容器处于无效状态。
- 析构函数不会抛出异常（即使没有用 `noexcept` 声明）。

### `noexcept`

- `noexcept` 是接口的一部分（与 `const` 类似）。
- 被声明为 `noexcept` 的函数更容易被编译器优化。
- `swap()`、资源释放操作、析构函数、移动操作，应当被声明为 `noexcept` 并给出相应的实现。
- 绝大多数函数是*异常中立的*，即本身不抛出异常、但其所调用的其他函数可能抛出异常，因此不应声明为 `noexcept`。
- 被声明为 `noexcept` 的函数 `foo()`，可在其实现中调用可能抛出异常的函数 `bar()`。若 `foo()` 中的 `bar()` 在运行期抛出了异常，且该异常直到 `foo()` 这一层都没有被捕获，则程序被 `std::terminate()` 终止。

## 资源管理

### 资源泄露
在 C++ 中，动态资源是通过成对的**获取 (acquire)** 和**释放 (release)** 操作来管理的。
例如，动态内存通过成对的 `new` 和 `delete` 语句来管理：

```cpp
#include <cassert>
#include <iostream>

void Use(int* a, int n) {
  for (int i = 0; i != n; ++i) {
    a[i] = i;
    std::cout << a[i] << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, const char* argv[]) {
  assert(argc > 1);
  int n = atoi(argv[1]);
  auto a = new int[n];  // 获取资源
  Use(a, n);            // 使用资源
  delete[] a;           // 释放资源
}
```
如果在 `Use()` 中增加越界检查，则它抛出的异常可能使 `main()` 中的 `delete` 语句无法被执行，从而造成内存泄露：
```cpp
void Use(int* a, int n) {
  for (int i = 0; i != n; ++i) {
    if (i < 0 or i >= n) {
      throw std::out_of_range("The given index is out of range.");
    }
    a[i] = i;
    std::cout << a[i] << ' ';
  }
  std::cout << std::endl;
}
```

### RAII
**RAII (Resource Acquisition Is Initialization)** 用一个[代理](../patterns/Proxy/README.md)对象来管理动态资源：在构造函数里获取资源，在析构函数里释放资源。
该技术利用了以下事实：当程序的执行点即将离开一个**作用域 (scope)** 时（无论是因为正常执行完该作用域内的所有语句，还是因为抛出异常），属于该作用域的**对象 (object)** 会依次被析构（即调用析构函数）。

标准库设施（容器、[智能指针](./memory/smart_pointers.md)）普遍采用 RAII 来管理动态资源。

利用 RAII，[上面](#资源泄露)的例子可以改写为以下形式：
```cpp
#include <cassert>
#include <iostream>
#include <stdexcept>

template <class T>
class Array {
  T* _a;
  const int _n;
 public: 
  explicit Array(int n) : _a(new T[n]), _n(n) { }
  ~Array() noexcept { delete[] _a; }
  int size() const noexcept { return _n; }
  int& operator[](int i) {
    if (i < 0 or i >= _n) {
      throw std::out_of_range("The given index is out of range.");
    }
    return _a[i];
  }
};

void Use(Array<int>& a) {
  for (int i = 0; i != a.size(); ++i) {
    if (i < 0 or i >= a.size()) {
      throw std::out_of_range("The given index is out of range.");
    }
    a[i] = i;
    std::cout << a[i] << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, const char* argv[]) {
  assert(argc > 1);
  int n = atoi(argv[1]);
  auto a = Array<int>(n);  // 创建对象时获取资源
  Use(a);  // 使用资源，可能会抛出异常
  // 无论是否抛出异常，离开作用域前都会析构对象，释放资源
}
```
