# 动态内存
一个`进程 (process)` 的`虚拟内存空间 (virtual memory space)` 分为以下几个部分:
| 类型 | 用途 |
| ---- | ---- |
| 静态 (static) 内存 | 全局对象; 局部静态变量; 类的静态数据成员 |
| 栈 (stack) 内存 | 非静态局部对象 |
| 堆 (heap) 内存 | 运行期动态分配的对象 |
这里的`堆内存`就是通常所说的`动态 (dynamic)` 内存.

需要用到动态内存的场合包括:
- 运行前不知道所需空间, 例如: 容器.
- 运行前不知道对象的具体类型, 例如: 多态 (polymorphism).
- 运行时在多个对象之间共享数据.

# 直接管理动态内存（慎用）
## `new` --- 分配内存并构造对象

### 值初始化
默认情况下, 动态分配对象时采用的是`默认初始化`.
若要进行`值初始化`, 需要在类型名后面紧跟 `()`, 例如
```cpp
std::string* ps1 = new std::string;   // 默认初始化 为 空字符串
std::string* ps = new std::string();  // 值初始化 为 空字符串
int* pi1 = new int;    // 默认初始化 为 不确定值
int* pi2 = new int();  // 值初始化 为 0
```

### 常值对象
动态分配的常值对象必须由`指向常量的指针`接管, 并且在创建时被初始化:
```cpp
const int* pci = new const int(1024);
const std::string* pcs = new const std::string;
```

### 内存耗尽
内存空间在运行期有可能被耗尽, 在此情况下, `new` (默认) 将抛出 `std::bad_alloc` 异常.
为阻止抛出异常, 可以在 `new` 与类型名之间插入 `(nothrow)`, 例如:
```cpp
#include <new>
int* p1 = new int;            // 如果分配失败, 将抛出 std::bad_alloc
int* p2 = new (nothrow) int;  // 如果分配失败, 将返回 nullptr
```

## `delete` --- 析构对象并释放内存

传递给 `delete` 的指针必须是`指向动态对象的指针`或 `nullptr`.

通常, 编译器无法判断一个指针所指的
- 对象是否是动态的
- 内存空间是否已经被释放

### 内存泄漏 (Memory Leak)
```cpp
Foo* factory(T arg) { return new Foo(arg); }
void use_factory(T arg) {
  Foo *p = factory(arg)  // factory 返回一个指向动态内存的指针
  // 使用 p
  delete p;  // 调用者负责将其释放
}
```
如果 `use_factory` 在返回前没有释放 `p` 所指向的动态内存, 则 `use_factory` 的调用者将不再有机会将其释放, 可用的动态内存空间将会变小 --- 这种现象被称为`内存泄漏`.

### 悬垂指针 (Dangling Pointer)
在执行完 `delete p;` 之后, `p` 将成为一个`悬垂的`指针, 对其进行
- 解引用, 并进行
  - 读: 会返回无意义的值
  - 写: (有可能) 会破坏有效数据
- 二次释放: 会破坏内存空间
为避免这些陷阱, 应当
- 将 `delete p;` 尽可能靠近 `p` 的作用域末端, 或者
- 在 `delete p;` 后面紧跟 `p = nullptr;`
即便如此, 由于同一个动态对象有可能被多个指针所指向, 还是有可能发生危险:
```cpp
int* p(new int(42));
auto q = p;   // p 和 q 指向同一块动态内存
delete p;     // 释放
p = nullptr;  // p 不再指向该地址
              // q 仍然指向该地址, 对其进行 解引用 或 二次释放 都有可能造成破坏
```

# 智能指针

## **`std::unique_ptr`**

`std::unique_ptr` 用于管理`独占所有权`的资源, 具有以下优点:

1. 体积小 --- 默认情况下, 与`裸指针`大小相同.
2. 速度快 --- 大多数操作 (含 `operator*()`) 执行与`裸指针`相同的指令.
3. `move`-only --- 确保独占所有权.



资源析构默认借助于 `operator delete()` 来完成, 也可以在创建时为其指定其他的 deleter. 如果被指定的 deleter 是:

- `函数指针` 或 `含有内部状态的函数对象`, 则 `std::unique_ptr` 的体积比裸指针大.
- `不含有内部状态的函数对象` (例如 无捕获的 lambda 表达式), 则 `std::unique_ptr` 的体积与裸指针相同.



`std::unique_ptr` 的类型参数支持两种形式:

- `std:: unique_ptr<T>` --- 用于单个对象.
- `std::unique_ptr<T[]>` --- 用于对象数组, 几乎只应当用于管理从 C-风格 API 所获得的动态内存.



`std::unique_ptr` 非常适合用作工厂方法的返回类型, 这是因为:

- `std::unique_ptr` 可以很容易地转为 `std::shared_ptr`.
- 将裸指针 (例如返回自 `new`) 赋值给 `std::unique_ptr` 的错误在编译期能够被发现.



## **`make`** 函数

`std::make_shared` 由 C++11 引入, 而 `std::make_unique` 则是由 C++14 引入.



`make` 函数有助于减少代码重复 (例如与 `auto` 配合可以少写一次类型), 并提高`异常安全性`, 例如在如下代码中

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

如果第 2 行抛出了异常, 则由 `new` 获得的动态内存来不及被 `std::unique_ptr` 接管, 从而有可能发生泄漏. 改用 `make` 函数则不会发生这种内存泄漏:

```cpp
processWidget(std::make_unique<Widget>(), 
              computePriority());
```

在无法使用 `make` 函数的情况下 (例如指定 deleter 或传入列表初始化参数), 一定要确保由 `new` 获得的动态内存`在一条语句内`被智能指针接管, 并且在该语句内`不做任何其他的事`.



对于 `std::make_shared` 和 `std::allocate_shared`, 用 `make` 函数可以节省空间和运行时间:

```cpp
std::shared_ptr<Widget> spw(new Widget);  // 2 memory allocations
auto spw = std::make_shared<Widget>();    // 1 memory allocation
```



`make` 函数用 `( )` 进行完美转发, 因此无法直接使用对象的列表初始化构造函数. 一种解决办法是先用 `auto` 创建一个 `std::initializer_list` 对象, 再将其传入 `make` 函数:

```cpp
auto initList = { 10, 20 };
auto spv = std::make_shared<std::vector<int>>(initList);
```



对于 `std::shared_ptr`, 不宜使用 `make` 函数的情形还包括: 

- 采用特制内存管理方案的类.
- 系统内存紧张, 对象体积庞大, 且 `std::weak_ptr` 比相应的 `std::shared_ptr` 存活得更久.