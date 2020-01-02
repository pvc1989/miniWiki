# 智能指针
C++11 在 `<memory>` 中以 ***类模板 (class template)*** 的形式提供了三种 ***智能指针 (smart pointers)***：[`std::unique_ptr`](#`std::unique_ptr`)、[`std::shared_ptr`](#`std::shared_ptr`)、 [`std::weak_ptr`](#`std::weak_ptr`)。

## 公共操作
### 默认初始化
默认初始化均 *接管或分享* `nullptr`：
```cpp
std::unique_ptr<T> uptr;
std::shared_ptr<T> sptr;
std::  weak_ptr<T> wptr;
```

### 用作判断条件
只支持 `std::unique_ptr` 和 `std::shared_ptr`：

```cpp
std::unique_ptr<T> uptr;
assert(!uptr);
```

### 解引用及访问成员
只支持 `std::unique_ptr` 和 `std::shared_ptr`：
```cpp
*p;      // 解引用，获得 p 所指对象的（左值）引用
p->mem;  // 等价于 (*p).mem
```

### 异常安全
即使在 *离开作用域* 或 *重置* 前抛出了 [***异常 (exception)***](../exception.md)，*智能指针* 也会确保资源被正确释放：
```cpp
void f() {
  auto sptr = std::make_shared<int>(42);
  // 中间代码可能抛出异常，并且没有被 f 捕获
  return;
}  // 离开作用域前，std::shared_ptr 负责释放资源
```
而用 *原始指针* 则有可能因 *忘记释放资源* 或 *忘记捕获异常* 而造成 *内存泄漏* ：
```cpp
void f() {
  auto ip = new int(42);
  // 中间代码可能抛出异常，并且没有被 f 捕获
  delete ip;  // 手动释放资源，但有可能因 忘记捕获异常 而运行不到这一行
}
```

### `swap()`
交换两个同一类型的 *智能指针* 所管理的 *原始指针* ：
```cpp
p.swap(q);
std::swap(p, q);
```

### `get()` ⚠️
返回 *智能指针* 所管理的 *原始指针* ：
```cpp
auto p = sptr.get();
```
⚠️ 该方法仅用于向 *只接受原始指针* 且 *不会释放资源* 的函数传递参数。

## `std::unique_ptr`

`std::unique_ptr<T>` 用于管理 *独占所有权* 的资源，具有以下优点：

1. 体积小：默认情况下，与 `T*` 大小相同。
2. 速度快：大多数操作（含解引用）与 `T*` 执行相同的指令。
3. 独占所有权：不能 ***拷贝 (copy)***，只能 ***移动 (move)***。

### 创建
自 C++14 起，推荐使用 `std::make_unique()` 函数来创建 `std::unique_ptr` 对象：
```cpp
auto uptr = std::make_unique<T>(args);
```
该函数依次完成三个任务：
1. 动态分配所需内存。
2. 用 `args` 初始化 `T` 类型的对象。
3. 返回指向该对象的 `std::unique_ptr`。

### 删除器
- *删除器类型* 是 *`std::unique_ptr<T>` 类型* 的一部分。
  - 每一个 *`std::unique_ptr<T>` 对象* 所拥有的 *删除器对象* 是在 ***编译期 (compile time)*** 绑定的，因此无法在 ***运行期 (run time)*** 更换。
  - 如果没有显式指定删除器，那么将采用 `std::default_delete<T>`。
- *删除器对象* 是 *`std::unique_ptr<T> `对象* 的一部分。
  - 如果删除器是 *函数指针* 或 *含有数据成员的函数对象*，则 `sizeof(std::unique_ptr<T>) > sizeof(T*)`。
  - 如果删除器是 *不含数据成员的函数对象*，例如 *无捕获的 lambda 表达式*，则 `sizeof(std::unique_ptr<T>) == sizeof(T*)`。

### `reset()`
`delete` 当前所管理的原始指针，然后接管传入的原始指针，含一次原始指针的赋值操作。

`std::unique_ptr<T>` 独占其所指对象的所有权，因此要确保

- 传入的 `T*` 不被其他 *智能指针* 管理。
- 传入的 `T*` 不会在其他地方被 `delete`。
```cpp
uptr.reset(ptr);      // 接管 原始指针 ptr
uptr.reset(nullptr);  // 接管 nullptr
uptr.reset();         // 同上
uptr = nullptr;       // 同上（不推荐）
```

### `release()`
让渡当前所管理的原始指针的所有权。
```cpp
auto p = uptr.release();
```
该方法至少含两次 *原始指针赋值* 操作：
```cpp
// 可能的实现：
pointer release() noexcept {
  auto p_temp = p_;  // 第一次 原始指针赋值
  p_ = nullptr;      // 第二次 原始指针赋值
  return p_temp;     // 通常由另一个智能指针接管
}
```

### 只能移动
典型用例：在函数中构造一个 `std::unique_ptr<T>` 并将其返回：

```cpp
template <class... Args>
unique_ptr<T> Create(Args&&... args) {
  auto uptr = make_unique<T>(std::forward<Args>(args)...);
  // ...
  return uptr;
}
```

这种函数被称为 [***工厂方法***](../../patterns/factory_method/README.md)。以 `std::unique_ptr<T>` 作为 *工厂方法* 的返回类型有如下好处：
- `std::unique_ptr<T>` 可以很容易地转为 `std::shared_ptr<T>`。
- 将原始指针赋值给 `std::unique_ptr<T>` 的错误在 *编译期* 就能被发现。

联合使用 `release()` 与 `reset()`，可以在两个 `std::unique_ptr<T>` 之间 ***传递所有权 (transfer ownership)*** ：
```cpp
auto p1 = std::make_unique<int>(16);    // p1 指向 16
std::unique_ptr<int> p2(p1.release());  // p1 为空，p2 指向 16
auto p3 = std::make_unique<int>(32);    // p1 为空，p2 指向 16，p3 指向 32
p2.reset(p3.release());                 // p1 为空，p2 指向 32，p3 为空
```

### 接管动态数组 ⚠️
类模板 `std::unique_ptr` 支持两种形式的模板类型参数：
- `std::unique_ptr<T>` 用于管理 *单个动态对象*。
- `std::unique_ptr<T[]>` 用于管理 *动态数组*。⚠️ 这种形式只应当用于 *接管 C-style API 返回的动态数组*。

与 `T*` 类似，可以用 `operator[]` 访问被 `std::unique_ptr<T[]>` 接管的数组的成员：
```cpp
#include <cstdlib>
#include <iostream>
#include <memory>

int main() {
  const int n = 10;
  std::unique_ptr<int[]> pa((int*) std::malloc(n * sizeof(int)));

  for (int i = 0; i < n; ++i) {
    pa[i] = (i == 0) ? 1 : i * pa[i-1];
  }

  for (int i = 0; i < n; ++i) {
    std::cout << i << ": " << pa[i] << '\n';
  }
}
```

## `std::shared_ptr`

### 创建
推荐使用 `std::make_shared<T>()` 函数来创建 `std::shared_ptr<T>` 对象：
```cpp
auto sptr = std::make_shared<T>(args);
```
该函数依次完成三个任务：
1. 动态分配所需内存。
2. 用 `args` 初始化 `T` 类型的对象。
3. 返回指向该对象的 `std::shared_ptr<T>`。

⚠️ 显式使用 `std::shared_ptr<T>` 的构造函数的场景：
- 已经存在另一个指向动态对象的（智能或原始）指针 `p`，并且希望它的所有权被 `std::shared_ptr<T>` 接管或分享。

```cpp
std::shared_ptr<T> sptr(p);     // sptr 接管或分享 p 所指对象
std::shared_ptr<T> sptr(p, d);  // sptr 接管或分享 p 所指对象, 并以 d 为删除器
```
具体语义取决于 `p` 的类型：

| `p` 的类型            | 语义                                           |
| :----------------------: | :-------------------------------------------: |
| `std::shared_ptr<T>` | `sptr` 分享 `p` 所指对象的所有权 |
| `std::unique_ptr<T>` | `sptr` 接管 `p` 所指对象，并令 `p` 指向 `nullptr` |
| 原始指针（必须是直接初始化）  | `sptr` 接管 `p` 所指对象 |

### 引用计数
尽管 C++ 标准没有规定 `std::shared_ptr` 的实现方式，但几乎所有实现都采用了 ***引用计数 (reference count) 方案***：
- 一个 `T*` 可以被多个 `std::shared_ptr<T>` 共享所有权，管理同一 `T*` 的 `std::shared_ptr<T>` 的个数称为它的 *引用计数*。
- 引用计数作为 ***控制块 (control block)*** 的一部分，需要存储在 *动态内存* 里，并通过 `std::shared_ptr<T>` 中的指针来访问。
- 所谓 *共享所有权* 其实是通过 *共享控制块* 来实现的。

```cpp
sptr.use_count();  // 获取 引用计数
sptr.unique();     // 判断 引用计数 是否为 1
```
这一方案存在以下性能缺陷：
- 空间开销：每个 `std::shared_ptr<T>` 至少含有两个指针成员，分别用于存储 *被管理对象* 与 *控制块*  的地址，因此 `std::shared_ptr<T>` 的大小至少是 `T*` 的 `2` 倍。
- 时间开销：为避免 ***数据竞争 (data racing)***，增减 *引用计数* 的操作必须是 ***原子的 (atomic)***。因此，隐含 *读写引用计数* 的操作（构造、析构、赋值）会比 *非原子* 操作消耗更多时间。

### 删除器
与 `std::unique_ptr<T>` 不同，
- *删除器类型* 不是 *`std::shared_ptr<T>` 类型* 的一部分。
  - 每个 *`std::shared_ptr<T>` 对象*  所绑定的 *删除器对象* 可以在 *运行期* 更换。
  - 如果没有显式指定删除器，那么将采用 `delete` 表达式。
- *删除器对象* 不是 *`std::shared_ptr<T>` 对象* 的一部分。
  - 删除器对象存储在 *控制块* 中，因此不会影响 `std::shared_ptr<T>` 的大小。
  - 如果删除器是 *函数指针* 或 *含有数据成员的函数对象*，则会作为 *控制块* 的一部分，存储在动态内存中。
  - 如果删除器是 *不含有数据成员的函数对象*，例如 *无捕获的 lambda 表达式*，则不会占据 *控制块* 的空间。

### 拷贝与移动
用一个 `std::shared_ptr<T>` 对另一个 `std::shared_ptr<T>` 进行 ***拷贝赋值 (copy-assign)*** 会改变二者的引用计数：
```cpp
p = q;  // p 的引用计数 - 1，q 的引用计数 + 1
```
同理，用一个 `std::shared_ptr<T>` ***拷贝构造 (copy-construct)*** 另一个 `std::shared_ptr<T>` 会增加前者的引用计数：
```cpp
auto p = q;  // q 的引用计数 + 1，p 的引用计数与之相同
```

 ***移动赋值 (move-assign)*** 与 ***移动构造 (move-construct)*** 不改变引用计数。

### `reset()`
如果当前 *引用计数* 为 `1`，则 `delete` 当前所管理的原始指针，然后接管传入的原始指针；
否则跳过 `delete` 操作。

```cpp
p.reset(q, d);  // 接管 *原始指针* q，并将 *删除器* 替换为 d
p.reset(q);     // 接管 *原始指针* q
p.reset();      // 接管 nullptr
```

### `shared_from_this()`
用 `this` 去创建 `std::shared_ptr<T>`，所得结果的引用计数为 `1`。
考虑以下情形：

```cpp
class Request {
 public:
  void Process();
 private:
  std::vector<std::shared_ptr<Request>> processed_requests_;
};
```
如果在 `Process()` 的实现中，用 `this` 创建了新的 `std::shared_ptr<Request>`：
```cpp
void Request::Process() {
  // ...
  processed_requests_.emplace_back(this);
  // ...
}
```
则有可能造成
- 一个 *非动态* 对象被一个 `std::shared_ptr<T>` 管理，或者
-  *一个* 动态对象被 *两个* 独立的 `std::shared_ptr<T>` 管理。

为避免以上情形，应当
- 对外：将 `Request` 的 *构造函数* 设为 `private`，改用 *工厂方法* 来创建 `std::shared_ptr<Request>`。
- 对内：借助于类模板 `std::enable_shared_from_this` 提供的 `shared_from_this()` 方法来获取 `std::shared_ptr<Request>`。
```cpp
#include <memory>
class Request: public std::enable_shared_from_this<Request> {
 public:
  void Process();
  // 工厂方法：
  template<typename... Args>
  static std::shared_ptr<Request> Create(Args&&... args);
 private:
  std::vector<std::shared_ptr<Request>> processed_requests_;
  //  构造函数 设为 private
  Request();
  // 其他成员方法 ...
};
void Request::Process() {
  // ...
  processed_requests_.emplace_back(shared_from_this());
  // ...
}
```

## `std::weak_ptr`
`std::weak_ptr` 必须与 `std::shared_ptr` 配合使用，并且不支持 *条件判断* 或 *解引用* 等常用的指针操作，因此它不是一种独立的智能指针。

### 创建
指向一个 `std::shared_ptr<T>` 所管理的对象，但不改变其引用计数：
```cpp
std::weak_ptr<T> wptr(sptr);
```

### 引用计数
获取引用计数的操作与 `std::shared_ptr<T>` 类似：
```cpp
wptr.use_count();  // 返回与之共享所有权的 std::shared_ptr<T> 的引用计数
wptr.expired();    // 等价于 wptr.use_count() == 0
```

如果引用计数不为零，通常希望执行 *解引用* 以获取所管理的对象。
但在 *判断引用计数是否为零* 与 *解引用* 这两步之间，所管理的对象有可能被其他 ***线程 (thread)*** 析构了，因此需要将两步合并为一个 *原子的* 操作：

```cpp
// 如果 expired() 返回 true, 则返回一个空的 std::shared_ptr<T>
// 否则, 返回一个与之共享所有权的 std::shared_ptr<T>, 引用计数 + 1
wptr.lock();
```

以上所说的 *引用计数* 均指 `std::shared_ptr<T>` 的个数。
除此之外，*控制块* 中还有一个 ***弱计数 (weak count)***，用于统计指向同一对象的 `std::weak_ptr<T>` 的数量。
因此，`std::weak_ptr<T>` 的构造、析构、赋值等操作都会 *读写弱计数*。
与 `std::shared_ptr<T>` 的 *引用计数* 类似：为避免 *数据竞争*，增减 *弱计数* 的操作必须是 *原子的*。
因此，隐含 *读写弱计数* 的操作（构造、析构、赋值）会比 *非原子* 操作消耗更多时间。

### 拷贝赋值
一个 `std::weak_ptr<T>` 或 `std::shared_ptr<T>` 可以 *拷贝赋值* 给另一个 `std::weak_ptr<T>`，但不改变 *引用计数* ：
```cpp
wptr = p;  // p 可以是 std::weak_ptr<T> 或 std::shared_ptr<T>
```

### `reset()`
只将自己所管理的 `T*` 设为 `nullptr`，不负责 *析构对象* 或 *释放内存* ：
```cpp
wptr.reset();
```

### 应用：缓存复杂操作
 *工厂方法* 返回 `std::shared_ptr<T>` 而非 `std::unique_ptr<T>`：
```cpp
std::shared_ptr<const Request> FastLoad(RequestId id) {
  static std::unordered_map<RequestId, std::weak_ptr<const Request>> cache;
  auto obj_ptr = cache[id].lock();
  if (!obj_ptr) {
    obj_ptr = RealLoad(id);
    cache[id] = obj_ptr;
  }
  return obj_ptr;
}
```

### 应用：实现 Observer 模式
[***Observer 模式***](../../patterns/observer/README.md) 要求：`Subject` 的状态发生变化时，应当通知所有的 `Observer`。
这一需求可以通过在 `Subject` 对象中维护一个存储 `std::weak_ptr<Observer>` 的容器来实现。

### 应用：避免 `std::shared_ptr` 成环
- 对于 ***图 (graph)*** 这种数据结构，只用 `std::shared_ptr<Node>` 有可能形成 ***环 (cycle)***。
  - 当环外不再有指向环内任一成员的 `std::shared_ptr<Node>`  时，环内的成员就成了 ***孤儿 (orphan)***，从而造成 *内存泄露*。
- 对于 ***树 (tree)*** 这种数据结构，`parent` 的生存期总是覆盖其 `child`，因此
  - `parent` 指向 `child` 的指针应当选用 `std::unique_ptr<Node>`。
  - `child` 指向 `parent` 的指针应当选用 `Node*`。
- 如果树的深度过大，例如长达 `100000000ul` 的 ***链表 (linked list)***，则有可能导致 `std::unique_ptr<Node>` 的析构函数递归爆栈。
  - 此时可以考虑用 ***循环 (iteration)*** 代替 ***递归 (recursion)*** 来实现析构。

## `make` 函数
尽量
- 用 `std::make_unique<T>()` 创建 `std::unique_ptr<T>`。
- 用 `std::make_shared<T>()` 创建 `std::shared_ptr<T>`。

### 节省资源
对于 `std::make_shared<T>()` 和 `std::allocate_shared<T>()`，除被管理的动态对象外，*控制块* 也需要动态分配内存。
用 `make` 函数可以节省 *存储空间* 和 *运行时间* ：
```cpp
std::shared_ptr<Object> sptr1(new Object);  // 分配 2 次
auto sptr2 = std::make_shared<Object>();    // 分配 1 次
```

### 异常安全
`make` 函数有助于减少代码重复（与 `auto` 配合可以少写一次 *类型名*）并确保 *异常安全*。
在如下语句中

```cpp
Process(std::unique_ptr<Request>(new Request), ComputePriority());
```
编译器只保证 *参数在被传入函数之前被取值*，因此实际的运行顺序可能是
```cpp
new Request
ComputePriority()  // 可能抛出异常
std::unique_ptr<Request>()
```
如果第二行抛出了异常，则由 `new` 获得的 `Request*` 来不及被 `std::unique_ptr<Request>` 接管，从而有可能发生泄漏。
用 `make` 函数就可以避免这种情况的发生：

```cpp
Process(std::make_unique<Request>(), ComputePriority());
```
实际的运行顺序只能是
```cpp
std::make_unique<Request>()
ComputePriority()  // 可能抛出异常
```
或
```cpp
ComputePriority()  // 可能抛出异常
std::make_unique<Request>()
```

在[不应或无法使用 `make` 函数的情况](#不应或无法使用的情形)下，一定要确保：
由 `new` 获得的动态内存 *在一条语句内* 被智能指针接管，并且在该语句内 *不做任何其他的事*。

### 不应或无法使用的情形
`make` 函数用 `()` 进行完美转发，因此无法直接使用 *列表初始化构造函数*。
一种解决办法是：先用 `auto` 创建一个 `std::initializer_list<T>` 对象，再将其传给 `make` 函数：

```cpp
auto init_list = { 10, 20 };
auto sptrv = std::make_shared<std::vector<int>>(init_list);
```

对于 `std::shared_ptr`，不应或无法使用 `make` 函数的情形还包括：
- 需要指定内存管理方案（分配器、删除器）的类。
- 系统内存紧张，对象体积巨大，且 `std::weak_ptr` 比相应的 `std::shared_ptr` 存活得更久。

## `pImpl` 模式

### 隔离依赖关系
假设在原始设计中，`Algorithm` 是一个含有 `Implementor` 类型数据成员的类：
```cpp
// algorithm.h
#include "implementor.h"

class Algorithm {
 public:
  Algorithm();
  // 其他成员方法 ...
 private:
  Implementor implementor_;
};
```
使用 `Algorithm` 的 `user.cpp` 必须引入 `algorithm.h`，这样会导致
- `user.cpp` 间接地引入了 `implementor.h`，从而会造成 *编译时间延长*。
- `implementor.h` 更新后，必须重新编译 `algorithm.cpp` 及 `user.cpp`。

所谓 `pImpl` 就是用 ***指向实现的指针 (Pointer to IMPLementation)*** 代替 *数据成员*：
- 将 `Algorithm` 对 `Implementor` 的依赖从 `algorithm.h` 移入 `algorithm.cpp`，从而将 `user.cpp` 与 `implementor.h` 隔离。
- `implementor.h` 更新后，只需重新编译 `algorithm.cpp`，而不必重新编译 `user.cpp`，但可能需要重新链接。

基于这一技术所设计的架构，完全符合 [***依赖倒置原则***](../../principles/README.md#DIP)，甚至用 C 语言也可以做到 ***面向对象编程 (Object Oriented Programming, OOP)***。

### 用 原始指针 实现
```cpp
// algorithm.h
class Algorithm {
 public:
  Algorithm();   // 需要 分配 资源：不是默认行为，需要显式声明
  ~Algorithm();  // 需要 释放 资源：不是默认行为，需要显式声明
  // 其他成员方法 ...
 private:
  struct Implementor;  // 仅声明，完整定义在 algorithm.cpp 中给出
  Implementor* pImpl_;
};
```
```cpp
// algorithm.cpp
#include "implementor.h"  // 定义 RealImplementor
#include "algorithm.h"
struct Algorithm::Implementor {
  RealImplementor implementor;
};
// 实现 构造 和 析构 函数:
Algorithm::Algorithm() : pImpl_(new Implementor) { }  // 分配
Algorithm::~Algorithm() { delete pImpl_; }            // 释放
```

### 用 `std::shared_ptr` 实现
```cpp
// algorithm.h
#include <memory>
class Algorithm {
 public:
  // 构造 和 析构 函数均采用默认版本
  // 其他成员方法 ...
 private:
  struct Implementor;  // 仅声明，完整定义在 algorithm.cpp 中给出
  std::shared_ptr<Implementor> pImpl_;  // 代替 Implementor*
};
```
```cpp
// algorithm.cpp
#include "implementor.h"  // 定义 RealImplementor
#include "algorithm.h"
struct Algorithm::Implementor {
  Implementor implementor;
};
```

### 用 `std::unique_ptr` 实现
```cpp
// algorithm.h
#include <memory>
class Algorithm {
 public:
  Algorithm();
  ~Algorithm();
  Algorithm(Algorithm&& rhs);
  Algorithm& operator=(Algorithm&& rhs);
  Algorithm(const Algorithm& rhs);
  Algorithm& operator=(const Algorithm& rhs);
  // 其他成员方法 ...
 private:
  struct Implementor;// 仅声明，完整定义在 algorithm.cpp 中给出
  std::unique_ptr<Implementor> pImpl_;  // 代替 Implementor* 
};
```
- 尽管希望使用 ***默认析构函数***，但还是要 *显式声明*，因为
  - 编译器在生成 *默认析构函数* 时，通常要求 `std::unique_ptr<Implementor>` 中的 `Implementor` 是完整类型。
  - 在 `pImpl` 模式中，`Implementor` 的定义只能在 `algorithm.cpp` 中给出，因此 `~Algorithm()` 只能在 `algorithm.cpp` 中实现。
  - 在 `algorithm.cpp` 中 *实现* 的方法，必须在 `algorithm.h` 中预先 *声明*。
- 尽管希望使用 ***默认移动操作***，但还是要 *显式声明*，因为
  - 显式声明 *析构函数*，会阻止编译器生成 *默认移动操作*。
  - *默认移动操作* 在捕获异常时需要调用 *默认析构函数*。
- ***拷贝操作***  需要 *显式声明*，因为
  - 编译器无法为含有 move-only 成员的类生成 *默认拷贝操作*。
  - *默认拷贝操作* 是 ***浅拷贝 (shallow copy)***，通常不符合拷贝语义。

```cpp
// algorithm.cpp
#include "algorithm.h"
#include "implementor.h"  // 定义 RealImplementor

struct Algorithm::Implementor {
  RealImplementor implementor;
};  // 至此，Implementor 已经是完整类型。

// 实现 构造、析构，可采用默认版本：
Algorithm::Algorithm() = default;
Algorithm::~Algorithm() = default;
// 实现 move 操作，可采用默认版本：
Algorithm::Algorithm(Algorithm&& rhs) = default;
Algorithm& Algorithm::operator=(Algorithm&& rhs) = default;
// 实现 copy 操作，不可采用默认版本：
Algorithm& Algorithm::operator=(const Algorithm& rhs) {
  // 拷贝所指对象，而非指针成员：
  pImpl_ = std::make_unique<Implementor>(*rhs.pImpl_);
  return *this;
}
Algorithm::Algorithm(const Algorithm& rhs)
    : pImpl_(std::make_unique<Implementor>(*rhs.pImpl_)) { }
```
