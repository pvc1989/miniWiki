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

## [原始指针 ⚠️](./raw_pointers.md#原始指针)

## [智能指针](./smart_pointers.md#智能指针)

## 动态数组

### 直接管理动态数组 ⚠️
大多数情况下应当优先选用标准库提供的「容器类」而不是「动态数组」。
如果要显式创建动态数组，则需要在「类型名」后面紧跟「对象个数」。
如果分配成功，则返回「指向数组第一个对象的指针」，否则抛出异常：

```cpp
auto pArray = new int[42];
```
数组「对象个数」是数组「类型」的一部分。
可以先定义「类型别名」，然后就可以像普通类型一样创建动态对象：
```cpp
typedef int Array[42];
auto pArray = new Array;
```

### `std::allocator`
通常，`new` 会依次完成「分配内存」和「构造对象」两个操作。
对于动态数组，后一个操作通常是多余的：`new` 所构造出来的对象通常会「立即」被其他参数所构造的对象覆盖。
标准库定义的 `std::allocator` 模板类可以将「分配内存」与「构造对象」两个操作分离：
```cpp
#include <memory>
std::allocator<T> a;      // 创建 allocator 对象
auto p = a.allocate(n);   // 「分配」整块内存，不进行「构造」，返回首元地址
a.deallocate(p, n);       // 「释放」整块内存，p 指向首元
a.construct(p, args);     // 「构造」单个对象，p 不必是首元地址，存储于 p 所指向的位置
a.destroy(p);             // 「析构」单个对象，p 不必是首元地址
```

`<memory>` 提供了一组相应的算法，用于在（由 `std::allocator` 获得的）「未初始化的 (uninitialized)」内存（块）中填入对象：
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
