---
title: 拷贝控制
---

# 析构
## 析构函数
**析构函数 (destructor)** 是一种以 `~` 为前缀、以类名为后缀的成员函数。它的形参列表为空，没有返回类型，用于**析构 (destroy)** 对象。

```cpp
class Foo {
 public:
  ~Foo();  // destructor
};
```

## 成员析构的顺序
一个对象被析构时，先执行析构函数*函数体*中的语句，再隐式地*逐个析构*其（非静态）数据成员。
数据成员被析构的顺序与它们被构造的顺序相反，即：与它们在类的定义中出现的顺序相反。

## 合成的析构函数
如果析构函数没有被显式地声明，那么编译器会隐式地定义一个默认的版本，称为**合成的 (synthesized)** 析构函数。
C++11 允许显式地生成合成的析构函数，只需要在定义时在形参列表后紧跟 `= default;` 即可。

合成的析构函数，只会逐个析构数据成员，这意味着不会对[原始指针](../memory/raw_pointers.md)成员调用 `delete` 运算符。

# 拷贝
## 拷贝构造函数
**拷贝构造函数 (copy constructor)** 是一类特殊的[构造函数](./class.md#构造函数)：

- *第一个形参*必须是*对同类对象的引用*，并且几乎总是*对 `const` 对象的引用*。
- 如果还有*其他形参*，则这些形参都应有*默认实参值*。
- 在许多场合，拷贝构造函数需要被隐式地调用，因此通常不应设为 `explicit`。

```cpp
class Foo {
 public:
  Foo(const Foo&);  // copy constructor
};
```

## 拷贝赋值运算符
**拷贝赋值运算符 (copy assignment operator)** 是对[赋值运算符](./operator.md#赋值运算符)的重载，函数签名几乎总是如下形式：

- 唯一的（显式）形参的类型为*指向 `const` 对象的引用*。
- 返回类型为*指向非 `const` 对象的引用*。

```cpp
class Foo {
 public:
  Foo& operator=(const Foo&);  // copy assignment operator
};
```

## 删除的拷贝操作
有些类型的对象不应支持拷贝操作（例如 `std::iostream`）。
自 C++11 起，实现该语义只需将拷贝操作（[拷贝构造函数](#拷贝构造函数)和[拷贝赋值运算符](#拷贝赋值运算符)）标注为**删除的 (deleted)** ：

```cpp
class Foo {
 public:
  Foo(const Foo&) = delete;
  Foo& operator=(const Foo&) = delete;
};
```

## 合成的拷贝操作
合成的拷贝操作（[拷贝构造函数](#拷贝构造函数)和[拷贝赋值运算符](#拷贝赋值运算符)）会*逐个拷贝*数据成员。
这意味着只会对*内置指针*进行**浅拷贝 (shallow copy)**，即：只拷贝该指针的值（所指对象的地址），而不拷贝所指对象。

如果含有数组成员，则合成的拷贝操作会*逐个拷贝*成员数组的每一个元素。

如果一个类含有**无法拷贝的 (non-copyable)** 数据成员，则这个类本身也应当是*无法拷贝的*，此时合成的拷贝操作将是[删除的](#删除的拷贝操作)。

# 移动

## 右值
简单来讲：

- *可以*出现在赋值号左侧的表达式称为**左值表达式 (lvalue expression)**。左值表达式代表对象的**身份 (identity)**，可以取地址。
- *只能*出现在赋值号右侧的表达式称为**右值表达式 (rvalue expression)**。右值表达式代表对象的**值 (value)**，不可以取地址。

*类型名*后紧跟 `&&` 表示定义一个对该类型对象的**右值引用 (rvalue reference)**：

- 右值引用（通常）只能绑定到*即将被析构的对象*或更一般的**亡值表达式 (xvalue expression)** 上。
- 定义在 `<utility>` 中的函数模板 `std::move<>()` 可以将*左值表达式*变为*右值表达式*。
- *右值引用*本身是一个*表达式*，并且是*左值表达式*。

```cpp
int i = 42;
int& r = i;                  // ✅ 将 左值引用 绑定到 左值表达式
int& r2 = i * 42;            // ❌ 普通左值引用 无法绑定到 右值表达式
const int& r3 = i * 42;      // ✅ 指向常量的左值引用 可以绑定到 右值表达式
int&& rr = i;                // ❌ 右值引用 无法直接绑定到 左值表达式
int&& rr2 = i * 42;          // ✅ 将 右值引用 绑定到 右值表达式
int&& rr3 = std::move(rr2);  // ✅ std::move 将 左值表达式 变为 右值表达式
```

## 移动构造函数
**移动构造函数 (move constructor)** 是一类特殊的[构造函数](./class.md#构造函数):

- *第一个形参*必须是*对该类型对象的右值引用*。
- 如果还有*其他形参*，则这些形参都应有*默认实参值*。
- 必须确保析构**移动源对象 (moved-from object)** 是安全的。

```cpp
template <typename T>
class Vector {
 public:
  Vector(Vector&& rhs) noexcept  // 不抛出异常
      // 接管 移动源对象 的数据成员:
      : head_(rhs.head_), free_(rhs.free_), tail_(rhs.tail_) {
    rhs.head_ = rhs.free_ = rhs.tail_ = nullptr;  // 确保 析构 rhs 是安全的
  }
 private:
  T* head_;  // 指向 首元
  T* free_;  // 指向 第一个自由元
  T* tail_;  // 指向 过尾元
};
```

## 移动赋值运算符
**移动赋值运算符 (move assignment operator)** 是对[赋值运算符](./operator.md#赋值运算符)的重载，函数签名几乎总是如下形式：

- 唯一的（显式）形参的类型是*对该类型对象的右值引用*。
- 返回类型为*对非 `const` 对象的左值引用*。

```cpp
template <typename T>
class Vector {
 public:
  Vector& operator=(Vector&& rhs) noexcept {  // 不抛出异常
    if (this == &rhs) {
      // 自己给自己赋值，不做任何事
    } else {
      free();  // 析构 被赋值对象 中的元素，释放内存
      // 接管 移动源对象 的数据成员：
      head_ = rhs.head_;
      free_ = rhs.free_;
      tail_ = rhs.tail_;
      rhs.head_ = rhs.free_ = rhs.tail_ = nullptr;  // 确保 析构 rhs 是安全的
    }
    return *this;  // 返回 左值引用
  }
 private:
  void free();  // 析构元素，释放内存
  T* head_;  // 指向 首元
  T* free_;  // 指向 第一个自由元
  T* tail_;  // 指向 过尾元
};
```

## `noexcept`
移动操作（[移动构造函数](#移动构造函数)和[移动赋值运算符](#移动赋值运算符)）一般只涉及赋值操作，不需要分配[动态内存](../memory/README.md)，因此不会抛出异常，应当在*形参列表*与*函数体*之间用 `noexcept` 标注。

标准库容器类型（例如 `std::vector<T>`）在**重新分配 (reallocation)** 的过程中，需要将所存储的元素*逐个搬运*到新分配的内存空间里。
如果 `T` 的[移动构造函数](#移动构造函数)被标注为 `noexcept`，则容器会利用它来搬运元素；
否则，容器将不得不用 `T` 的[拷贝构造函数](#拷贝构造函数)来搬运元素。

