# 原始指针（慎用）

## `new`

### 分配内存 + 默认初始化

置于 *类型名* 之前的 ***`new` 运算符*** 用于创建 *单个动态对象*。
如果分配成功，则返回一个 *指向动态对象的指针*，否则 *抛出异常*：

```cpp
int* p = new int;
```

`new` 语句依次完成三个任务:

1. 动态 ***分配 (allocate)*** 所需内存；
2. 默认 ***初始化 (initialize)*** 对象；
3. 返回指向该对象的 ***原始 (raw) 指针***。

### 分配内存 + 值初始化

若要进行 ***值 (value) 初始化***，需要在 *类型名* 后面紧跟 `()` 或 `{}`，例如

```cpp
std::string* ps1 = new std::string;    // 默认初始化 为 空字符串
std::string* ps2 = new std::string();  // 值初始化 为 空字符串
int* pi1 = new int;    // 默认初始化 为 不确定值
int* pi2 = new int();  // 值初始化 为 0
```

### 常值对象

动态分配的常值对象必须由 *指向常量的指针* 接管，并且在创建时被初始化：

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

### 内存耗尽

内存空间在运行期有可能被耗尽，此时 *分配内存* 的任务无法完成。

- 在默认情况下，分配失败会抛出 `std::bad_alloc`。
- 如果在 `new` 与 *类型名* 之间插入 `(std::nothrow)`，则分配失败时不会抛出异常，而是以 `nullptr` 作为返回值。⚠️ 使用这种形式的 `new` 一定要记得检查返回值是否为 `nullptr`。
- `std::bad_alloc` 及 `std::nothrow` 都定义在 `<new>` 中。

```cpp
#include <new>
int* p1 = new int;                 // 如果分配失败, 将抛出 std::bad_alloc
int* p2 = new (std::nothrow) int;  // 如果分配失败, 将返回 nullptr
```

## `delete`

### 析构对象 + 释放内存

传递给 `delete` 的指针必须是 *指向动态对象的指针* 或 `nullptr`：

```cpp
delete p;     // 析构并释放 (单个) 动态对象
```

⚠️ 编译器无法判断一个指针所指的 *对象是否是动态的*，也无法判断一个指针所指的 *内存是否已经被释放*。

### 内存泄漏

```cpp
Foo* factory(T arg) { return new Foo(arg); }
void use_factory(T arg) {
  Foo *p = factory(arg)  // factory 返回一个指向动态内存的指针
  // 使用 p
  delete p;  // 调用者负责将其释放
}
```

如果 `use_factory` 在返回前没有释放 `p` 所指向的动态内存，则 `use_factory` 的调用者将不再有机会将其释放，可用的动态内存空间将会变小。这种现象被称为 ***内存泄漏 (memory leak)***。

### 空悬指针

执行完 `delete p` 之后，`p` 将成为一个 ***空悬 (dangling) 指针***，对其进行

- 解引用，并进行
  - 读：返回无意义的值。
  - 写：有可能破坏数据。
- 再次 `delete p`：会破坏内存空间。

为避免这些陷阱，应当

- 将 `delete p` 尽可能放在 `p` 的作用域末端，或者
- 在 `delete p` 后面紧跟 `p = nullptr`。

即便如此，由于同一个动态对象有可能被多个指针所指向，还是会有危险：

```cpp
auto q = p;   // p 和 q 指向同一块动态内存
delete p;     // 释放
p = nullptr;  // p 不再指向该地址
              // q 仍然指向该地址, 对其进行 解引用 或 再次释放 都有可能造成破坏
```

## 动态数组

### `new T[]` + `delete[]`

大多数情况下应当优先选用标准库提供的 *容器* 而不是 *动态数组*。
如果要显式创建动态数组，则需要在 *类型名称* 后面紧跟 *对象个数*。
如果分配成功，则返回 *指向数组第一个对象的指针*，否则 *抛出异常*：

```cpp
auto pa = new int[42];
delete[] pa;  // 析构并释放 (整个) 动态数组
```

*对象个数* 是数组 *类型* 的一部分。
可以先定义 *类型别名*，然后就可以像普通类型一样创建动态对象：

```cpp
typedef int Array[42];  // 等价于 using Array = int[42];
auto pa = new Array;
```

### 在给定位置构造对象

与单个对象类似，数组版本的 `new` 会依次完成 ***分配 (allocate) 内存*** 和 ***构造 (construct) 对象*** 两个操作。对于动态数组，通常希望将这两个操作拆分开，前者可以通过分量 `char` 数组或 [`std::malloc`](#`std::malloc`) 完成，后者可以通过 ***placement new*** 完成：

```cpp
int count = 5;
char* char_ptr = new char[count * sizeof(Foo)];  // 分配内存
Foo* foo_ptr = new (char_ptr) Foo(args);  // 构造 foo_ptr[0]
foo_ptr->~Foo();                          // 析构 foo_ptr[0]
new (foo_ptr + 3) Foo(args);  // 构造 foo_ptr[3]
(foo_ptr + 3)->~Foo();        // 析构 foo_ptr[3]
delete[] char_ptr;  // 释放整块内存
```

其中

- `char_ptr` 与 `foo_ptr` 的 *值* 相同，但 *类型* 不同，因此指针加减运算的 *步长* 也不同。
- `new (foo_ptr + 3) Foo(args)` 不能用 `foo_ptr[3] = Foo(args)` 代替，因为赋值操作会析构左侧对象，而 `foo_ptr[3]` 还没有被构造过。

### `std::allocator`

标准库定义的 `std::allocator` 类模板将以上操作（*分配*、*构造*、*析构*、*释放*）封装为其成员函数，使用时更加安全：

```cpp
#include <memory>
std::allocator<T> a;      // 创建 allocator 对象
T* p = a.allocate(n);     // 分配 整块内存，不进行 构造，返回首元地址
a.deallocate(p, n);       // 释放 整块内存，p 指向首元
a.construct(p, args);     // 构造 单个对象，p 不必是首元地址，存储于 p 所指向的位置
a.destroy(p);             // 析构 单个对象，p 不必是首元地址
```

### `std::malloc`

C 标准库的 `<stdlib.h>` 提供了一组动态内存管理函数，也可以用来管理动态数组：

```cpp
// 分配 total_size 个字节的内存，不做初始化：
void* malloc(std::size_t total_size);
// 分配 num * each 个字节的内存，并将所有字节初始化为 0：
void* calloc(std::size_t num, std::size_t each);
// 释放由 malloc | calloc 分配的内存：
void free(void* p);
```

在 C++ 中，它们被声明在命名空间 `std` 中：

```cpp
#include <cstdlib>
auto p = (int*) std::malloc(sizeof(int) * 100);
auto q = (int*) std::calloc(100, sizeof(int));
std::free(p);
std::free(q);
```

### `std::uninitialized_*`

`<memory>` 提供了一组名为 `std::uninitialized_*` 的算法，用于在由 `std::allocator` 或 `std::malloc` 获得的 ***未初始化的 (uninitialized) 内存*** 中填入对象：

```cpp
#include <memory>
// 从另一个容器中 copy 或 move (C++17)：
std::uninitialized_copy(b, e, p);    // 返回 p + n
std::uninitialized_move(b, e, p);    // 返回 p + n
std::uninitialized_copy_n(b, n, p);  // 返回 p + n
std::uninitialized_move_n(b, n, p);  // 返回 p + n
// 在一段范围内用 t 进行构造：
std::uninitialized_fill(b, e, t);    // 返回 void
std::uninitialized_fill_n(b, n, t);  // 返回 b + n
```