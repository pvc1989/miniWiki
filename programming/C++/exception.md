# 异常

## 运行期错误及处理策略

### 运行期错误
**运行期 (run-time)** 错误是指程序在运行时可能会遇到的非正常或非预想的状态。
这种错误往往由 **运行期** 所获得或产生的非法参数所导致，因此无法在 **编译期 (compile-time)** 被发现和排除。

对于简单的错误，通常可以在本地进行 **处理 (handle)**。
而对于下面这种更复杂、更一般的场景，就需要引入专门的 **错误处理策略 (error-handling strategy)**：
- **作者** 知道哪里会发生运行期错误，但不知道该如何去处理这种错误。
- **用户** 知道该如何处理运行期错误，但不知道这种错误会在哪里发生。

⚠️ 对于新系统，应当确定错误处理策略；对于现有系统，应当与现有的错误处理策略保持一致。

### 传统处理策略
在[异常机制](#异常机制)被引入 C++ 之前，业界已经形成了一些常用的错误处理策略。
这些策略可以看作是[异常机制](#异常机制)的替代项，但都带有各自的缺点。

#### 中止程序
调用 `exit()` 或 `abort()` 来中止（整个）程序。

主要缺点：用户可能无法承受系统崩溃所造成的影响。

#### 返回错误码
返回一个表示错误信息的整数（或枚举值）。

主要缺点：用户可能忘记检测错误码。

#### 设置状态变量
返回一个合法值，将某个用于记录状态的变量设为错误状态。

主要缺点：用户可能忘记检测状态变量。

#### 调用处理函数
调用处理函数的抽象接口，由用户提供具体实现。

主要缺点：系统行为依赖于用户提供的具体实现。

### 异常机制
C++ 语言提供的 **异常 (exception)** 机制只用来处理 **同步 (synchronous)** 事件（数组越界、读写错误等），而不能（直接）处理 **异步 (asynchronous)** 事件（键盘中断、电源故障等)。

#### 标准库异常类
任何可复制的对象都可以被用作异常。

标准库中用到的异常以 `std::exception`（定义在 `<exception>` 中）为共同基类。
除构造和析构函数外，它含有两个公共的方法成员：
- `operator=()` 用于异常对象的复制。
- `what()` 返回构造时传入的字符串。

标准库中用到的其他异常定义在 `<stdexcept>` 等头文件中。
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

#### 抛出 (`throw`)
如果一个操作可能发生运行期错误，则应当用 `throw` 语句 **抛出** 一个异常：
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

#### 捕获 (`catch`)
用户应当将可能抛出的异常的操作置于 `try` 代码块（用`{ }`表示边界）中，并紧随其后用一个或多个 `catch` 子句进行 **捕获**：
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
  } catch (std::out_of_range& e) {
    std::cerr << e.what() << std::endl;
  } catch (...) {
    throw;
  }
  for (int i = 0; i != anArray.size(); ++i) {
    std::cout << anArray.at(i) << ' ';
  }
  std::cout << std::endl;
}
```
这里的两条 `catch` 子句体现了两种典型的处理策略：
- 对于 `std::out_of_range` 类型的异常，用 `what()` 方法打印提示信息。
- 对于其他类型的异常，用 `throw;` 语句将其重新抛出。

## 异常安全
### 概念
#### 类不变量
每个类的定义中都隐含有一个 **类不变量 (class invariant)**，它表示这个类的对象在变化过程中所必须保持的性质。
对于每个对象，该不变量在构造函数运行后就被建立起来。
在析构函数运行前，所有访问该对象内部表示的操作都必须维护该不变量。

#### 广义不变量
不变量的概念可以推广到多个对象之间。
在下面的例子中，*`x.size() == y.size()` 始终为真* 就是一种不变量：
```cpp
struct Points {
  vector<int> x;
  vector<int> y;
};
```

#### 有效状态
- 对于一个对象，如果其类不变量没有被破坏，则称它处于有效状态。
- 对于含有不变量的多个对象，如果这种不变量没有被破坏，则称它们处于有效状态。

#### 异常保证
标准库的每一个操作都至少满足以下三种保证之一：
- 所有操作都满足 **基本保证 (basic gaurantee)**：类不变量得到维护，不会发生资源泄露。
在满足基本保证的基础上，
- 关键操作（例如 `std::vector::push_back()`）满足 **强保证 (strong gaurantee)**：如果操作失败，则不产生影响。
- 简单操作（例如 `std::vector::pop_back()`）满足 **无抛出保证 (nothrow gaurantee)**：该操作的实现不会抛出异常（应当用 `noexcept` 标识）。

**基本保证** 和 **强保证** 都要求：
- 自定义操作不会造成资源泄露。
- 容器操作所依赖的自定义操作（`operator=()`, `swap()`）不会使容器处于无效状态。
- 析构函数不会抛出异常。

