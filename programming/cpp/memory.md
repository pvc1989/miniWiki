# 动态内存管理

## 内存分类
一个「进程 (process)」的「虚拟内存空间 (virtual memory space)」分为以下几个部分：

| 分类 | 用途 |
| ---- | ---- |
| 静态 (static) 内存 | 全局对象；局部静态变量；类的静态数据成员 |
| 栈 (stack) 内存 | 非静态局部对象 |
| 堆 (heap) 内存 | 运行期动态分配的对象 |

这里的「堆内存」就是通常所说的「动态 (dynamic) 内存」。

需要用到动态内存的场合包括：
- 运行前不知道所需空间，例如：容器 (container)。
- 运行前不知道对象的具体类型，例如：多态 (polymorphism)。
- 运行时在多个对象之间共享数据。

## 原始指针 ⚠️
### `new`

#### 分配内存 + 构造对象
置于「类型名」之前的 `new` 运算符用于创建「单个动态对象」。
如果分配成功，则返回一个「指向动态对象的指针」，否则抛出异常：
```cpp
int* p = new int;
```
`new` 语句依次完成三个任务:
1. 动态分配所需内存；
2. 默认初始化对象；
3. 返回指向该对象的「原始 (raw) 指针」。

#### 值初始化
默认情况下，动态分配对象时采用的是「默认 (default) 初始化」。
若要进行「值 (value) 初始化」，需要在「类型名」后面紧跟 `()` 或 `{ }`，例如
```cpp
std::string* ps1 = new std::string;    // 默认初始化 为 空字符串
std::string* ps2 = new std::string();  // 值初始化 为 空字符串
int* pi1 = new int;    // 默认初始化 为 不确定值
int* pi2 = new int();  // 值初始化 为 0
```

#### 常值对象
动态分配的常值对象必须由「指向常量的指针」接管，并且在创建时被初始化：
```cpp
const int* pci = new const int(1024);
const std::string* pcs = new const std::string;
```
自 C++11 起，推荐用 `auto` 作为对象类型，编译器会推断出变量的类型：
```cpp
auto pi = new int();
auto ps = new std::string();
auto pci = new const int(1024);
auto pcs = new const std::string("hello");
```

#### 内存耗尽
内存空间在运行期有可能被耗尽，此时「分配内存」的任务无法完成。
- 在默认情况下，分配失败会抛出 `std::bad_alloc`。
- 如果在 `new` 与「类型名」之间插入 `(std::nothrow)`，则分配失败时不会抛出异常，而是以 `nullptr` 作为返回值。
- `std::bad_alloc` 及 `std::nothrow` 都定义在 `<new>` 中。
```cpp
#include <new>
int* p1 = new int;                 // 如果分配失败, 将抛出 std::bad_alloc
int* p2 = new (std::nothrow) int;  // 如果分配失败, 将返回 nullptr
```
⚠️ 使用 `new (std::nothrow)` 一定要检查返回是否为 `nullptr`。

### `delete`

#### 析构对象 + 释放内存
传递给 `delete` 的指针必须是「指向动态对象的指针」或 `nullptr`：
```cpp
delete p;     // 析构并释放 (单个) 动态对象
delete[] pa;  // 析构并释放 (整个) 动态数组
```

⚠️ 编译器无法判断一个指针所指的「对象是否是动态的」，也无法判断一个指针所指的「内存是否已经被释放」。

#### 内存泄漏
```cpp
Foo* factory(T arg) { return new Foo(arg); }
void use_factory(T arg) {
  Foo *p = factory(arg)  // factory 返回一个指向动态内存的指针
  // 使用 p
  delete p;  // 调用者负责将其释放
}
```
如果 `use_factory` 在返回前没有释放 `p` 所指向的动态内存，则 `use_factory` 的调用者将不再有机会将其释放，可用的动态内存空间将会变小。
这种现象被称为「内存泄漏 (memory leak)」。

#### 空悬指针
在执行完 `delete p;` 之后, `p` 将成为一个「空悬 (dangling) 指针」，对其进行
- 解引用，并进行
  - 读：返回无意义的值。
  - 写：有可能破坏数据。
- 二次 `delete`：会破坏内存空间。

为避免这些陷阱，应当
- 将 `delete p;` 尽可能放在 `p` 的作用域末端，或者
- 在 `delete p;` 后面紧跟 `p = nullptr;`。

即便如此，由于同一个动态对象有可能被多个指针所指向，还是会有危险：
```cpp
int* p(new int(42));
auto q = p;   // p 和 q 指向同一块动态内存
delete p;     // 释放
p = nullptr;  // p 不再指向该地址
              // q 仍然指向该地址, 对其进行 解引用 或 二次释放 都有可能造成破坏
```

## 智能指针
C++11 在 `<memory>` 中定义了三种智能指针：
- [`std::unique_ptr`](#`std::unique_ptr`)
- [`std::shared_ptr`](#`std::shared_ptr`)
- [`std::weak_ptr`](#`std::weak_ptr`)

### 公共操作
#### 与原始指针相同的操作
##### 默认初始化
默认初始化为 `nullptr`：
```cpp
std::unique_ptr<T> up;
std::shared_ptr<T> sp;
std::weak_ptr<T> wp;
```

##### 用作判断条件
非空为真，只支持 `std::unique_ptr` 和 `std::shared_ptr`。

##### 解引用及访问成员
只支持 `std::unique_ptr` 和 `std::shared_ptr`.
```cpp
*p;      // 解引用，获得 p 所指对象的（左值）引用
p->mem;  // 等价于 (*p).mem
```

#### 异常安全
即使在「离开作用域」或「重置」前抛出了异常，「智能指针」也会确保资源被正确释放：
```cpp
void f() {
  auto sp = std::make_shared<int>(42); 
  // 中间代码可能抛出异常，并且没有被 f 捕获
  return;  // std::shared_ptr 负责释放资源，即使异常没有被捕获
}
```
而用「原始指针」则有可能因「忘记 `delete`」或「忘记捕获异常」而造成「内存泄漏」：
```cpp
void f() {
  auto ip = new int(42);
  // 中间代码可能抛出异常，并且没有被 f 捕获
  delete ip;  // 手动释放资源，但有可能因「忘记捕获异常」而运行不到这一行，造成内存泄漏
}
```

#### `swap()`
交换两个同一类型的「智能指针」所管理的「原始指针」：
```cpp
p.swap(q);
std::swap(p, q);
```

#### `get()` ⚠️
返回「智能指针」所管理的「原始指针」：
```cpp
auto p = sp.get();
```
⚠️ 该方法仅用于向「只接受原始指针」且「不会 `delete` 被传入的指针」的函数传递参数。

### `std::unique_ptr`

`std::unique_ptr` 用于管理「独占所有权」的资源，具有以下优点：
1. 体积小：默认情况下，与「原始指针」大小相同。
2. 速度快：大多数操作（含解引用）与「原始指针」执行相同的指令。
3. 独占所有权：不能「拷贝 (copy)」或「赋值 (assign)」，只能「移动 (move)」。

#### 创建
自 C++14 起，推荐使用 `std::make_unique()` 函数来创建 `std::unique_ptr` 对象：
```cpp
auto up = std::make_unique<T>(args);
```
该函数依次完成三个任务：
1. 动态分配所需内存。
2. 用 `args` 初始化 `T` 类型的对象。
3. 返回指向该对象的 `std::unique_ptr`。

#### 删除器
「删除器的类型」是「`std::unique_ptr` 类型」的一部分。
每一个「`std::unique_ptr` 对象」所拥有的「删除器对象」是在「编译期」绑定的，因此无法在「运行期」更换。
如果没有显式指定删除器，那么将采用 `delete`。

删除器是 `std::unique_ptr` 对象的一部分：
- 如果删除器是「函数指针」或「含有数据成员的函数对象」，则相应的 `std::unique_ptr` 比「原始指针」大。
- 「不含有数据成员的函数对象」，例如「未捕获参数的 lambda 表达式」，则相应的 `std::unique_ptr` 与「原始指针」具有相同的大小。

#### `reset()`
`delete` 当前所管理的原始指针，然后接管传入的原始指针，含一次原始指针的赋值操作。

`std::unique_ptr` 独占其所指对象的所有权，因此要确保
- 传入的「原始指针」不被其他「智能指针」管理。
- 传入的「原始指针」不会在其他地方被 `delete`。
```cpp
// 声明：
void reset( pointer ptr = pointer() ) noexcept;
// 用例：
up.reset(q);        // 接管原始指针 q
up.reset(nullptr);  // 接管 nullptr
up.reset();         // 同上
up = nullptr;       // 同上（不推荐）
```

#### `release()`
让渡当前所管理的原始指针的所有权。
```cpp
// 声明:
pointer release() noexcept;
// 用例:
auto p = up.release();
```
该方法通过以下 `3` 步来实现其功能，至少含 `2` 次原始指针的赋值操作：
1. 用所管理的原始指针保存为临时指针（第 `1` 次）。
2. 将所管理的原始指针设为 `nullptr`（第 `2` 次）。
3. 返回第 `1` 步创建的临时指针，通常交给另一个智能指针接管。

#### 只能移动
特例：即将被销毁的 `std::unique_ptr` 可以被拷贝或赋值。
例如在函数（工厂方法）中构造一个 `std::unique_ptr` 并将其返回：
```cpp
unique_ptr<int> clone(int x) {
  auto up = make_unique<int>(x);
  // 使用 up
  return up;
}
```

`std::unique_ptr` 非常适合用作「工厂方法」的返回类型，原因如下：
- `std::unique_ptr` 可以很容易地转为 `std::shared_ptr`。
- 将原始指针赋值给 `std::unique_ptr` 的错误在「编译期」就能被发现。

联合使用 `release()` 与 `reset()`，可以在两个 `std::unique_ptr` 对象之间「传递所有权」：
```cpp
auto p1 = std::make_unique<int>(16);    // p1 指向 16
std::unique_ptr<int> p2(p1.release());  // p1 为空，p2 指向 16
auto p3 = std::make_unique<int>(32);    // p1 为空，p2 指向 16，p3 指向 32
p2.reset(p3.release());                 // p1 为空，p2 指向 32，p3 为空
```

#### 接管动态数组 ⚠️
`std::unique_ptr` 支持两种形式的类型参数：
- `std::unique_ptr<T>` 用于管理「单个动态对象」。
- `std::unique_ptr<T[]>` 用于管理「动态数组」。
⚠️ 第二种形式只应当用于：接管 C-style API 所返回的动态数组。

与「原始指针」类似，可以用 `[]` 访问数组成员：
```cpp
#include <iostream>
#include <memory>

int main() {
  const int size = 10;
  std::unique_ptr<int[]> upa(new int[size]);

  for (int i = 0; i < size; ++i) {
    upa[i] = (i == 0) ? 1 : i * upa[i-1];
  }

  for (int i = 0; i < size; ++i) {
    std::cout << i << ": " << upa[i] << '\n';
  }
}
```

### `std::shared_ptr`

#### 创建
推荐使用 `std::make_shared()` 函数来创建 `std::shared_ptr` 对象：
```cpp
auto sp = std::make_shared<T>(args);
```
该函数依次完成三个任务：
1. 动态分配所需内存。
2. 用 `args` 初始化 `T` 类型的对象。
3. 返回指向该对象的 `std::shared_ptr`。


⚠️ 只在一种情况下，才会显式使用 `std::shared_ptr` 的构造函数，那就是：
已经存在另一个指向「动态」对象的指针 `q`，并且希望它被 `std::shared_ptr` 接管。
```cpp
std::shared_ptr<T> sp(q);     // sp 指向 q 所指对象
std::shared_ptr<T> sp(q, d);  // sp 指向 q 所指对象, 并指定删除器
```
具体语义取决于 `q` 的类型:

| `q` 的类型                | 语义                                           |
| ------------------------ | --------------------------------------------- |
| 原始指针（必须是直接初始化）  | `sp` 接管 `q` 所指对象 |
| `std::shared_ptr<T>` 对象 | `sp` 分享 `q` 所指对象的所有权 |
| `std::unique_ptr<T>` 对象 | `sp` 接管 `q` 所指对象，并令 `q` 指向 `nullptr` |

#### 引用计数
尽管 C++ 标准没有规定 `std::shared_ptr` 的实现方式，但几乎所有的实现都采用了「引用计数 (reference count)」方案。
一个「原始指针」可以被多个 `std::shared_ptr` 共享所有权，管理同一「原始指针」的 `std::shared_ptr` 的个数称为它的「引用计数」：
```cpp
sp.use_count();  // 获取「引用计数」
sp.unique();     // 判断「引用计数」是否为 1
```
这一方案存在以下性能缺陷：
- 空间开销：引用计数作为「控制块 (control block)」，需要存储在「动态内存」里，并通过 `std::shared_ptr` 对象中的指针来访问。因此 `std::shared_ptr` 的大小至少是「原始指针」的 `2` 倍。
- 时间开销：为避免「数据竞争 (data racing)」，增减引用计数的操作必须是「原子的 (atomic)」。因此「读写引用计数」会比「非原子」操作消耗更多时间。

#### 删除器
与 `std::unique_ptr` 不同，「删除器的类型」不是「`std::shared_ptr` 类型」的一部分。
每一个「`std::shared_ptr` 对象」所绑定的「删除器对象」可以在「运行期」更换。

如果没有显式指定删除器，那么将采用 `delete`。
删除器不会影响 `std::shared_ptr` 对象的大小，但是会间接影响「控制块」的大小：
- 如果删除器是「函数指针」或「含有数据成员的函数对象」，则它将会作为「控制块」的一部分，存储在动态内存中。
- 如果删除器是「不含有数据成员的函数对象」，例如「未捕获参数的 lambda 表达式」，则它不会占据「控制块」的空间。

所谓「共享所有权」，正是通过「共享控制块」来实现的。

#### 拷贝与移动
用一个 `std::shared_ptr` 对另一个 `std::shared_ptr` 进行「拷贝赋值 (copy-assign)」会改变二者的引用计数：
```cpp
p = q;  // p 的引用计数 - 1，q 的引用计数 + 1
```
同理，用一个 `std::shared_ptr`「拷贝初始化 (copy-initialize)」另一个 `std::shared_ptr` 会增加前者的引用计数：
```cpp
auto p = q;  // q 的引用计数 + 1，p 的引用计数与之相同
```

「移动赋值 (move-assign)」与「移动初始化 (move-initialize)」不需要改变引用计数。

#### `reset()`
如果当前「引用计数」为 `1`，则 `delete` 当前所管理的原始指针，然后接管传入的原始指针；
否则跳过 `delete` 操作。
```cpp
p.reset(q, d);  // 接管「原始指针」q，并将「删除器」替换为 d
p.reset(q);     // 接管「原始指针」q
p.reset();      // 接管 nullptr
```

#### `shared_from_this()`
`this` 是「原始指针」，用它去创建 `std::shared_ptr`，所得结果的引用计数为 `1`。
考虑以下情形：
```cpp
class Request {
 public:
  void process();
 private:
  std::vector<std::shared_ptr<Request>> processed_requests_;
};
```
如果在 `process` 的实现中，用 `this` 创建了新的 `std::shared_ptr`：
```cpp
void Request::process() {
  // ...
  processed_requests_.emplace_back(this);
  // ...
}
```
则有可能造成
- 一个「非动态」对象被一个 `std::shared_ptr` 管理，或者
- 「一个」动态对象被「两个」独立的 `std::shared_ptr` 管理。
为避免以上情形，应当
- 对外：将 `Request` 的「构造函数」设为 `private`，改用「工厂方法」来创建 `std::shared_ptr`。
- 对内：借助于标准库模板类 `std::enable_shared_from_this` 提供的 `shared_from_this()` 来获取 `std::shared_ptr`。
```cpp
#include <memory>
class Request: public std::enable_shared_from_this<Request> {
 public:
  void process();
  // 工厂方法：
  template<typename... Ts>
  static std::shared_ptr<Request> create(Ts&&... params);
 private:
  std::vector<std::shared_ptr<Request>> processed_requests_;
  // 「构造函数」设为 private
  // ...
};
void Request::process() {
  // ...
  processed_requests_.emplace_back(shared_from_this());
  // ...
}
```

### (C++11) `std::weak_ptr`
`std::weak_ptr` 不支持`条件判断`或`解引用`等常用的指针操作, 因此不是一种独立的智能指针, 而必须与 `std::shared_ptr` 配合使用. 

#### 创建
指向一个 `std::shared_ptr` 所管理的对象, 但不改变其引用计数:
```cpp
std::weak_ptr<T> wp(sp);
```

#### 引用计数
获取引用计数的操作与 `std::shared_ptr` 类似:
```cpp
// 返回与之共享所有权的 std::shared_ptr 的引用计数:
w.use_count();
// 等价于 w.use_count() == 0:
w.expired();
```

如果引用计数不为零, 通常希望执行`解引用`以获取所管理的对象.
但在`判断引用计数是否为零`与`解引用`这两步之间, 所管理的对象有可能被其他线程析构了, 因此需要将两步合并为一个`原子`操作:
```cpp
// 如果 expired() 返回 true, 则返回一个空的 std::shared_ptr
// 否则, 返回一个与之共享所有权的 std::shared_ptr, 引用计数 + 1
w.lock();
```

以上所说的`引用计数`均指 `std::shared_ptr` 的个数.
除此之外, `控制块`中还有一个`弱引用计数 (weak count)`, 用于统计指向同一对象的 `std::weak_ptr` 的数量.
因此, `std::weak_ptr` 的创建, 析构, 赋值等操作都会读写`弱引用计数`.
为避免`数据竞争`, 读写引用计数操作都是`原子的`, 从而会比`非原子操作`消耗更多资源.

#### 拷贝与移动
一个 `std::weak_ptr` 或 `std::shared_ptr` 可以`拷贝赋值`给另一个 `std::weak_ptr`, 但不改变引用计数:
```cpp
w = p;  // p 可以是 std::weak_ptr 或 std::shared_ptr
```

#### `reset` --- 重设裸指针
只将自己所管理的裸指针设为 `nullptr`, 不负责析构对象或释放内存:
```cpp
w.reset();
```

#### 应用场景
##### 缓存复杂操作
工厂方法返回 `std::shared_ptr` 而非 `std::unique_ptr`
```cpp
std::shared_ptr<const Widget> fastLoadWidget(WidgetID id) {
  static std::unordered_map<WidgetID, std::weak_ptr<const Widget>> cache;
  auto objPtr = cache[id].lock();
  if (!objPtr) {
    objPtr = loadWidget(id);
    cache[id] = objPtr;
  }
  return objPtr;
}
```

##### 实现 Observer 模式
Observer 模式要求: `Subject` 的状态发生变化时, 应当通知所有的 `Observer`.
这一需求可以通过在 `Subject` 对象中维护一个存储 `std::weak_ptr<Observer>` 对象的容器来实现.

##### 避免 `std::shared_ptr` 成环
对于非树结构, 全部使用 `std::shared_ptr` 有可能形成环.
当环外的所有对象都不再指向环内的任何一个成员时, 环内的成员就成了孤儿, 从而造成内存泄露.

对于树结构, `parent` 的生存期总是大于其 `child`, 因此
- `parent` 指向 `child` 的指针应当选用 `std::unique_ptr`
- `child` 指向 `parent` 的指针可以选用`裸指针`

### (C++11) `std::make_shared` 与 (C++14) `std::make_unique`

#### 节省资源
对于 `std::make_shared` 和 `std::allocate_shared`, 用 `make` 函数可以节省存储空间和运行时间:
```cpp
std::shared_ptr<Widget> spw(new Widget);  // 分配 2 次
auto spw = std::make_shared<Widget>();    // 分配 1 次
```

#### 异常安全
`make` 函数有助于减少代码重复 (例如与 `auto` 配合可以少写一次类型), 并提高`异常安全`.
例如在如下代码中
```cpp
processWidget(std::unique_ptr<Widget>(new Widget), 
              computePriority());
```
编译器只能保证`参数在被传入函数之前被取值`, 因此实际的运行顺序可能是:
```cpp
new Widget
computePriority()
std::unique_ptr<Widget>()
```
如果第 2 行抛出了异常, 则由 `new` 获得的动态内存来不及被 `std::unique_ptr` 接管, 从而有可能发生泄漏.
用 `make` 函数就可以避免这种情况的发生:
```cpp
processWidget(std::make_unique<Widget>(), 
              computePriority());
```

在不应或无法使用 `make` 函数的情况下, 一定要确保由 `new` 获得的动态内存`在一条语句内`被智能指针接管, 并且在该语句内`不做任何其他的事`.

#### 列表初始化
`make` 函数用 `()` 进行完美转发, 因此无法直接使用对象的列表初始化构造函数.
一种解决办法是先用 `auto` 创建一个 `std::initializer_list` 对象, 再将其传入 `make` 函数:
```cpp
auto initList = { 10, 20 };
auto spv = std::make_shared<std::vector<int>>(initList);
```

#### 不使用 `make` 函数的情形
对于 `std::shared_ptr`, 不应或无法使用 `make` 函数的情形还包括: 
- 需要指定内存管理方案 (allocator, deleter) 的类.
- 系统内存紧张, 对象体积庞大, 且 `std::weak_ptr` 比相应的 `std::shared_ptr` 存活得更久.

### `pImpl` --- 指向实现的指针
#### 动机
假设 `Widget` 是一个含有数据成员的类.
在定义 `Widget` 之前, 必须在 `widget.h` 中引入定义成员类型的头文件:

```cpp
// widget.h
#include <vector>
#include "gadget.h"

class Widget {
 public:
  Widget();
  // ...
 private:
  std::vector<double> data;
  Gadget g1, g2, g3;
};
```
`Widget` 的`用户` 必须引入 `widget.h` --- 这样就间接地引入了 (它可能并不需要的) `<vector>` 和 `gadget.h`, 从而会造成`编译时间过长`以及`用户对实现的依赖` (这些头文件更新后, 需要重新编译 `Widget` 及其`用户`).

所谓 `pImpl` 就是用`指向实现的指针 (pointer to implementation)` 代替数据成员, 将 `Widget` 对成员数据类型的依赖从`头文件`移入`源文件`, 从而隔离`用户`与`实现`.

#### 用 裸指针 实现
```cpp
// widget.h
class Widget {
 public:
  Widget();   // 不是默认行为 (需要分配资源), 需要显式声明
  ~Widget();  // 不是默认行为 (需要释放资源), 需要显式声明
  // 其他成员方法 ...
 private:
  struct Impl;  // 封装数据成员的类型, 在 widget.h 仅作声明, 在 widget.cpp 中实现
  Impl* pImpl;  // 裸指针
};
```
```cpp
// widget.cpp
#include <vector>
#include "gadget.h"
#include "widget.h"

struct Widget::Impl {
  std::vector<double> data;
  Gadget g1, g2, g3;
};

// 实现构造和析构函数:
Widget::Widget() : pImpl(new Impl) { }  // 分配资源
Widget::~Widget() { delete pImpl; }     // 释放资源
```

#### 用 `std::shared_ptr` 实现
```cpp
// widget.h
#include <memory>
class Widget {
 public:
  // 构造和析构函数均采用默认版本
  // 其他成员方法 ...
 private:
  struct Impl;  // 封装数据成员的类型, 在 widget.h 仅作声明, 在 widget.cpp 中实现
  std::shared_ptr<Impl> pImpl;  // 代替裸指针
};
```
```cpp
// widget.cpp
#include <vector>
#include "gadget.h"
#include "widget.h"

struct Widget::Impl {
  std::vector<double> data;
  Gadget g1, g2, g3;
};
```

#### 用 `std::unique_ptr` 实现
```cpp
// widget.h
#include <memory>
class Widget {
 public:
  // 尽管希望使用 默认析构函数, 但还是要显式声明, 因为:
  // 编译器在生成 默认析构函数 时, 通常要求 std::unique_ptr<Impl> 中的 Impl 是完整类型.
  // 因此需要在 widget.h 中显式声明, 而将 (默认的) 实现移到 widget.cpp 中.
  Widget();
  ~Widget();
  // 尽管希望使用 默认移动操作, 但还是要显式声明, 因为:
  // 显式声明析构函数 会阻止编译器生成 默认移动操作, 并且
  // 默认移动操作内部 会在抛出异常时调用 默认析构函数.
  Widget(Widget&& rhs);
  Widget& operator=(Widget&& rhs);
  // 拷贝操作 需要显式声明, 因为:
  // 编译器不会为含有 move-only 成员 (std::unique_ptr) 的类生成 默认拷贝操作, 并且
  // 默认拷贝操作 是 浅拷贝, 通常不符合语义要求.
  Widget(const Widget& rhs);
  Widget& operator=(const Widget& rhs);
  // 其他成员方法 ...
 private:
  struct Impl;  // 封装数据成员的类型, 在 widget.h 仅作声明, 在 widget.cpp 中实现
  std::unique_ptr<Impl> pImpl;  // 代替裸指针
};
```
```cpp
// widget.cpp
#include <vector>
#include "gadget.h"
#include "widget.h"

struct Widget::Impl {
  std::vector<double> data;
  Gadget g1, g2, g3;
};
// 至此, Impl 已经是完整类型.

// 实现构造和析构函数, 采用默认版本:
Widget::Widget() = default;
Widget::~Widget() = default;
// 实现移动操作, 采用默认版本:
Widget::Widget(Widget&& rhs) = default;
Widget& Widget::operator=(Widget&& rhs) = default;
// 实现拷贝操作:
Widget& Widget::operator=(const Widget& rhs) {
  *pImpl = *rhs.pImpl;  // 深拷贝
  return *this;
}
Widget::Widget(const Widget& rhs)
    : pImpl(std::make_unique<Impl>(*rhs.pImpl)) { }
```

## 动态数组

### 直接管理动态数组（慎用）
大多数情况下应当优先选用标准库提供的`容器类`而不是`动态数组`.
如果要显式创建`动态数组`, 则需要在`类型名`后面紧跟数组`元素个数`.
如果分配成功, 则返回一个指向该数组第一个元素的指针, 否则抛出异常:
```cpp
int* pArray = new int[42];
```
数组`元素个数`是数组`类型`的一部分.
可以先定义`类型别名`, 然后就可以像普通类型一样创建动态对象:
```cpp
typedef int Array[42];
int* pArray = new Array;
```

### `std::allocator`
通常, `new` 会依次完成 <分配内存> 和 <构造对象> 两个操作.
对于动态数组, 后一个操作是多余的, 因为 `new` 所构造出来的对象会 <立即> 被其他参数所构造的对象覆盖.
标准库定义的 `std::allocator` 模板类可以将 <分配内存> 与 <构造对象> 两个操作分离:
```cpp
#include <memory>
std::allocator<T> a;      // 创建 allocator 对象
auto p = a.allocate(n);   // 分配整块内存, 不进行构造, 返回首元素地址
a.deallocate(p, n);       // 释放整块内存, p 指向首元素
a.construct(p, args);     // 构造对象, 存储于 p 所指向的位置, p 不必是首元素地址
a.destroy(p);             // 析构 p 所指向的对象
```

`<memory>` 提供了一组相应的算法, 用于在 (由 `std::allocator` 获得的) <未初始化的::uninitialized> 内存中填入对象:
```cpp
#include <memory>
// 从另一个容器中 copy 或 move (C++17):
std::uninitialized_copy(b, e, p);    // 返回 p+n
std::uninitialized_move(b, e, p);    // 返回 p+n
std::uninitialized_copy_n(b, n, p);  // 返回 p+n
std::uninitialized_move_n(b, n, p);  // 返回 p+n
// 在一段范围内用 t 进行构造:
std::uninitialized_fill(b, e, t);    // 返回 void
std::uninitialized_fill_n(b, n, t);  // 返回 b+n
```
