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

# 智能 (Smart) 指针
`智能指针`对普通指针进行了封装, 因此可以将普通指针形象地称为`裸 (raw) 指针`.

C++11 引入了三种智能指针: `std::unique_ptr`, `std::shared_ptr`, `std::weak_ptr`.

## 公共操作
### 与裸指针相同的操作
#### 默认初始化
默认初始化为空指针:
```cpp
shared_ptr<T> sp;
unique_ptr<T> up;
```

#### 用作判断条件

#### 解引用及访问成员
```cpp
*p;      // 解引用, 获得 p 所指对象的 (左值) 引用
p->mem;  // 等价于 (*p).mem
```

### `swap` --- 交换所管理的裸指针
`p` 和 `q` 必须是同一类型的智能指针:
```cpp
p.swap(q);
std::swap(p, q);
```

### `get` --- 获得所管理的裸指针 (慎用)
```cpp
p.get();
```
该方法仅用于向`只接受裸指针`并且`不会将传入的指针 delete 掉`的函数传递参数.

## (C++11) `std::unique_ptr`

`std::unique_ptr` 用于管理`独占所有权`的资源 (通常是内存), 具有以下优点:
1. 体积小 --- 默认情况下, 与`裸指针`大小相同.
2. 速度快 --- 大多数操作 (含解引用) 执行与`裸指针`相同的指令.
3. 只能`移动` --- 不能`拷贝`与`赋值`, 确保独占所有权.

### 创建
推荐使用 (C++14) `std::make_unique` 函数来创建 `std::unique_ptr` 对象:
```cpp
auto up = std::make_unique<T>(args);
```
该函数依次完成三个任务:
1. 动态分配所需内存
2. 用 `args` 初始化 `T` 类型的对象
3. 返回指向该对象的 `std::unique_ptr`

### 指定 deleter
`deleter 类型`是 `std::unique_ptr 类型`的一部分.
每一个 `std::unique_ptr 对象`所拥有的 `deleter 对象`是在`编译期`绑定的, 因此无法在`运行期`更换.

如果没有显式指定 deleter, 那么将采用 `delete`.
如果被指定的 deleter 是
- `函数指针` 或 `含有内部状态的函数对象`, 则 `std::unique_ptr` 的体积比裸指针大.
- `不含有内部状态的函数对象` (例如 无捕获的 lambda 表达式), 则 `std::unique_ptr` 的体积与裸指针相同.

### `reset` --- 重设裸指针
一个 `std::unique_ptr` 对象独占其所指对象的所有权, 因此重设裸指针总是会 `delete` 它之前所管理的裸指针:
```cpp
// 接管裸指针 q:
u.reset(q);
// 将自己设为空指针:
u.reset(nullptr);
u.reset();
u = nullptr;
```

### `release` --- 让渡所有权
```cpp
u.release();
```
该方法依次完成三个任务:
1. 返回其所管理的`裸指针` (通常由另一个智能指针接管)
2. 放弃对所指对象的所有权
3. 将自己设为空指针

### 不支持拷贝或赋值
特例: 即将被销毁的 `std::unique_ptr` 可以被拷贝或赋值.
例如在函数 (工厂方法) 中构造一个 `std::unique_ptr` 对象并将其返回:
```cpp
unique_ptr<int> clone(int x) {
  auto up = make_unique<int>(x);
  // ...
  return up;
}
```

`std::unique_ptr` 非常适合用作`工厂方法`的返回类型, 这是因为:
- `std::unique_ptr` 可以很容易地转为 `std::shared_ptr`.
- 将裸指针赋值给 `std::unique_ptr` 的错误在编译期能够被发现.

借助于 `release` 和 `reset` 可以移交管理权:
```cpp
auto p1 = make_unique<string>("hello");
// 此时, p1 指向 "hello"
unique_ptr<string> p2(p1.release());
// 此时, p1 为空, p2 指向 "hello"
auto p3 = make_unique<string>("world");
// 此时, p1 为空, p2 指向 "hello", p3 指向 "world"
p2.reset(p3.release());
// 此时, p1 为空, p2 指向 "world", p3 为空
```

### 接管动态数组
`std::unique_ptr` 支持两种形式的类型参数:
- `std::unique_ptr<T>` --- 用于管理单个动态对象.
- `std::unique_ptr<T[]>` --- 用于管理动态数组, 只应当用于接管从 C-风格 API 所获得的动态数组.

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