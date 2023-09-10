---
title: 动态内存管理
---

## 内存分类
一个[**进程 (process)**](../../csapp/8_exceptional_control_flow.md) 的[**虚拟内存空间 (virtual memory space)**](../../csapp/9_virtual_memory.md) 分为以下几个部分：

| 分类 | 用途 |
| :--: | :--: |
| **静态 (static)** 内存 | 全局对象；局部静态变量；类的静态数据成员 |
| **栈 (stack)** 内存 | 非静态局部对象 |
| **堆 (heap)** 内存 | 运行期动态分配的对象 |

其中*堆内存*就是通常所说的**动态 (dynamic)** 内存。

需要用到动态内存的场合包括：
- 运行前不知道所需空间，例如**容器 (container)**。
- 运行前不知道对象的具体类型，例如**多态 (polymorphism)**。
- 运行时在多个对象之间共享数据。

## [原始指针](./memory/raw_pointers.md)

## [智能指针](./memory/smart_pointers.md)

## [内存检查](./memory/check.md)
