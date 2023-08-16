---
title: 创建类型
---

# 抽象数据类型
**类 (class)** 机制最基本的用途是在 `char`、`int`、`double` 等**内置类型 (built-in types)** 之外，创建新的**抽象数据类型 (Abstract Data Type, ADT)**。

这里的*抽象*体现在：
- 类的**使用者 (user)** 只需要了解并且只能访问类的**接口 (interface)**，这些*接口*
  - 包括*公共方法成员*和*公共数据成员*，以及二元[运算符](./operator.md)等*非成员接口函数*，
  - 通常在后缀为 `.h` 或 `.hpp` 的**头文件 (header)** 中以*源代码*形式给出**声明 (declaration)**。
- 类的**实现者 (implementor)** 负责提供类的**实现 (implementation)**：
  - 包括*私有方法成员*和*私有数据成员*，以及成员方法和非成员接口函数的**定义 (definition)**，
  - 通常在后缀为 `.cc` 或 `.cpp` 或 `.cxx` 的**源文件 (source file)** 中以*源代码*形式给出，也可以只提供编译生成的*目标文件*、*静态库*、*动态库*。

这样做的好处是：
- *使用者*代码只依赖于*接口*，而不依赖于*实现*：
  - *实现者*不需要将算法细节暴露给*使用者*，有助于保护知识产权。
  - *使用者*和*实现者*可以同时、独立地开发和测试。
  - *实现*发生变化时，不需要重新*编译*使用者的源代码，而只需将（重新编译实现代码）所得的目标文件*链接*进使用者的目标文件。
- 有助于减少不同类之间的依赖，允许各自独立变化。

# 访问控制
## 访问修饰符
一个类可以含有零个或多个**访问修饰符 (access specifier)**，每种访问修饰符出现的*次数*和*顺序*不限。
每个修饰符的作用范围起始于自己，终止于下一个修饰符或类的末尾。

| 访问修饰符 | 从类的外部 | 从类的内部 |
| :------: | :------: | :------: |
| `public` | 可以直接访问 | 可以直接访问 |
| `private` | （除[友元](#friend)外）无法直接访问 | 可以直接访问 |

## 默认访问级别
`class` 和 `struct` 都可以用来定义一个类。对于访问控制，二者的区别仅在于：

| 关键词  | 隐含的第 `0` 个访问修饰符 |
| :----: | :--------------------: |
| `struct` | `public`  |
| `class`  | `private` |

## `friend`<a href id="friend"></a>
定义一个类时，可以用 `friend` 将其他（可见的）类或函数声明为它的**友元**，从而允许这些友元访问其私有成员。
*友元声明*不是*函数声明*。
通常，将友元声明集中放在类定义的头部或尾部。

⚠️ 友元机制破坏了类的封装，因此要少用。

# 类型成员
一个类可以含有类型成员，可以是已知类型的**别名 (alias)**，也可以是定义在其内部的**嵌套类 (nested class)**。
类型成员必须在使用前被定义，因此通常将它们集中定义在类的头部。
类型成员与[数据成员](#数据成员)和[函数成员](#函数成员)遵循相同的[访问控制](#访问控制)规则：

```cpp
class Screen {
 public:
  typedef std::string::size_type Position;
 private:
  Position cursor_ = 0;
  Position height_ = 0;
  Position width_ = 0;
};
```

# 数据成员

# 函数成员
## 声明与定义
所有成员函数都必须在类的内部（通常位于*头文件*中）进行*声明*，但其*定义*可以放在类的外部（通常位于*源文件*中）。

## `this` 指针<a href id="this"></a>
除[静态成员函数](#static)外，所有成员函数都是通过隐式指针 `this` 来访问调用它的那个对象的。
```cpp
SalesData total;
total.isbn()
// 相当于
SalesData::isbn(&total)
```

## `const` 成员函数<a href id="const"></a>
默认情况下，`this` 是指向 non-`const` 对象的指针，这使得相应的成员函数无法被 `const` 对象调用。
如果要使 `this` 为指向 `const` 对象的指针，只需要在函数形参列表后面紧跟 `const` 关键词。

## `inline` 成员函数<a href id="inline"></a>
成员函数可以是**内联的 (inline)**：
- 定义在类的*内部*的成员函数是*隐式*内联的。
- 定义在类的*外部*的成员函数也可以是内联的，只需要在（位于同一头文件中的）函数*定义*前加上 `inline` 关键词。

## 其他接口函数
除了公共的方法成员，还可以在类的外部定义接口函数，最典型的是[重载为普通函数的运算符](./operator.md#通常重载为普通函数)。
如果需要在这些函数的*实现*中访问类的私有成员，则应将它们声明为 [`friend`](#friend)。

# `static` 成员<a href id="static"></a>
**静态 (static)** 成员由一个*类*的所有*对象*共享，因此不属于其中任何一个对象：

- 静态*数据*成员存储于所有对象的外部，不计入对象的大小。
- 静态*方法*成员独立于所有对象，形参列表不含 [`this` 指针](#this)。

## 访问静态成员
在类的外部，静态成员可以通过紧跟在*类名*后面的**作用域运算符 (scope operator) `::`** 来访问，也可以（像非静态成员一样）通过*对象*或指向该对象的*指针*或*引用*来访问。

在类的内部，静态成员可以被所属类的成员函数*直接*访问，不需要借助于作用域运算符。

## 定义静态成员
关键词 `static` 仅用于在类的内部*声明*静态成员，而不需要在类的外部*定义*静态成员时重复。

静态数据成员必须在类的*外部*进行*定义*和*初始化*。
与*非内联成员函数*类似，每个静态数据成员都只能被*定义一次*，因此应当将它们的定义放在同一个*源文件*中。

定义静态数据成员时，可以访问该类的*私有*成员。

## 类内初始化
通常，静态数据成员*不可以*在类的内部进行初始化，但有两个例外：

- *可以*为 `static const` *整型*数据成员指定类内初始值。
-  *必须*为 `static constexpr` 数据成员指定类内初始值。

用作类内初始值的表达式必须是 `constexpr`，被其初始化的静态数据成员也是 `constexpr`，可以用于任何需要 `constexpr` 的地方：
```cpp
// account.h
class Account {
 private:
  static constexpr int kLength = 30;  // kLength 是 constexpr
  double table[kLength];  // 数组长度必须是 constexpr
};
```
即使一个静态数据成员已经在类内被*初始化*，通常也应在类外给出*定义*。
如果其初始值已经在类内给定，则类外*不得*再给定初始值：

```cpp
// account.cpp
#include "account.h"
constexpr int Account::kLength;
```

## 特殊用法
静态数据成员的类型可以是它自己所属的那个类：
```cpp
class Point {
 private:
  static Point p1_;  // 正确: 静态数据成员 可以是 不完整类型
  Point* p2_;        // 正确: 指针成员 可以是 不完整类型
  Point  p3_;        // 错误: 非静态数据成员 必须是 完整类型
};
```
静态数据成员可以（在声明前）被用作默认实参：
```cpp
class Screen {
 public:
  Screen& clear(char c = kBackground);
 private:
  static const char kBackground;
};
```

# 构造函数
**构造函数 (constructor)** 是一种用于构造对象的特殊成员函数：以类名为函数名，没有返回类型。

在构造过程中，需要修改数据成员的值，因此构造函数不可以被声明为 [`const`](#const)。

## 默认构造函数
**默认构造函数 (default constructor)** 是指*形参列表为空*或*所有形参都有默认实参值*的特殊构造函数。

如果没有*显式地*定义任何构造函数，那么编译器会*隐式地*定义一个**合成的 (synthesized)** 默认构造函数。

自 C++11 起，允许（并且推荐）在形参列表后紧跟 `= default;` 以*显式地*生成该构造函数。

## 初始化列表
```cpp
class Point {
  double x_;
  double y_;
};
```
**初始化列表 (initializer list)** 位于*形参列表*与*函数体*之间，用于**值初始化 (value initialize)** 数据成员：

```cpp
// 推荐：在『初始化列表』中『初始化』数据成员
Point::Point(const double& x, const double& y)
    : x_(x), y_(y) {
}
```
*初始化列表*中成员按照它们在*类的定义*中出现的顺序依次进行构造。

没有出现在*初始化列表*中的数据成员会被**默认初始化 (default initialize)**，即调用[默认构造函数](#默认构造函数)，然后才会进入*函数体*：

```cpp
// 语义相同, 但『默认初始化』得到的值，立即被函数体内的『赋值』覆盖，浪费了计算资源
Point::Point(const double& x, const double& y) {
  x_ = x;
  y_ = y;
}
```

[`const`](#const) 成员、*引用*成员、没有[默认构造函数](#默认构造函数)的成员必须利用*初始化列表*进行初始化。

## 委托构造函数
**委托构造函数 (delegated constructor)** 在其[初始化列表](#初始化列表)中*调用*另一个构造函数，从而将构造任务*委托*给那个构造函数：

```cpp
class Point {
 public:
  Point(const double& x, const double& y)  // 双参数构造函数
      : x_(x), y_(y) {
  }
  Point()  // 默认构造函数
      : Point(0.0, 0.0) /* 委托给双参数构造函数 */ {
  }
 private:
  double x_;
  double y_;
};
```

## `explicit` 构造函数<a href id="explicit"></a>
默认情况下，只需要传入*一个实参*的构造函数定义了一种由*形参类型*到*当前类型*的隐式类型转换。
编译器只会进行一次这种转换。

如果需要（通常应该）禁止这种隐式类型转换，只需要在构造函数头部加上关键词 `explicit`：
```cpp
namespace std{
template <class T, class Allocator = std::allocator<T>>
class vector {
 public:
  // 禁止 std::size_type 到 std::vector 的隐式类型转换：
  explicit vector(std::size_type count);
};
}  // namespace std
```

