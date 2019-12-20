# 运算符重载

「运算符 (operator)」是一种特殊的函数：函数名总是以 `operator` 起始，后面紧跟某种特殊符号（如 `+`, `++`, `+=`）。
运算符可以是普通函数，也可以是类的方法成员（隐式地以 `this` 为第一个形参）。

运算符的种类和[优先级](https://en.cppreference.com/w/cpp/language/operator_precedence)都是由语言规范所确定的。
程序员只能对已有的运算符进行「重载 (overload)」，而不能创造新的运算符；在重载时，只能修改形参类型，而不能改变形参个数。

下面以 `Point` 为例，为其重载运算符：
```cpp
// point.h
#include <iostream>
class Point {
  friend bool operator==(const Point& lhs, const Point& rhs);
  friend bool operator<(const Point& lhs, const Point& rhs);
 public:
  Point(double x = 0.0, double y = 0.0) : _x(x), _y(y) { }
  // 赋值运算符
  Point& Point::operator=(double const (array&) [2]);
  // 下标运算符
  double& operator[](int i);
  const double& operator[](int i) const;
  // 类型转换运算符
  explicit operator bool() const;
  // 普通方法
  const double& x() const { return _x; }
  const double& y() const { return _y; }
 private:
  double _x;
  double _y;
};
// 读写运算符
std::ostream& operator<<(std::ostream& os, const Point& point);
std::istream& operator>>(std::istream& is, Point& point);
// 算术运算符
Point operator+(const Point& lhs, const Point& rhs);
// 关系运算符
bool operator==(const Point& lhs, const Point& rhs);
bool operator!=(const Point& lhs, const Point& rhs);
bool operator<(const Point& lhs, const Point& rhs);
```
```cpp
// point.cpp
#include "point.h"
constexpr Point Point::kOrigin;
```

## 必须重载为普通函数的运算符

### 读写运算符
「读写 (IO) 运算符」通常需要访问私有成员，因此通常需要声明为 `friend`。

输出运算符 `<<` 应当尽量减少对输出格式的修改（例如：不应添加换行符）：
```cpp
// point.cpp
#include "point.h"
#include <iostream>
std::ostream& operator<<(std::ostream& os, const Point& point) {
  os << '(' << point.x_ << ", " << point.y_ << ')';
  return os;
}
```

输入运算符 `>>` 必须处理（可能发生的）输入失败的情形：
```cpp
// point.cpp
#include "point.h"
#include <iostream>
std::istream& operator>>(std::istream& is, Point& point) {
  is >> point.x_ >> point._y;
  if (!is)
    point = Point();  // 输入失败，恢复到默认状态
  return is;
}
```

## 通常重载为普通函数的运算符
「对称的 (symmetric)」二元运算符通常应当重载为普通函数。

### 算术运算符
「算术 (arithmetic) 运算符」通常返回一个新的对象（或代理）。

对于定义了[算术运算符](#算术运算符)和相应的[复合赋值运算符](复合赋值运算符)的类，应当将算术运算委托给[复合赋值运算符](复合赋值运算符)，这样可以避免将非成员的[算术运算符](#算术运算符)声明为 `friend`：
```cpp
// point.cpp
#include "point.h"
Point operator+(const Point& lhs, const Point& rhs) {
  Point sum = lhs;
  sum += rhs;
  return sum;
}
```

### 关系运算符
「关系 (relational) 运算符」总是返回 `bool` 值。

`==` 关系应当是「传递的 (transitive)」，即： `a == b && b == c` 意味着 `a == c`。
类似的，`<`、`<=`、`>`、`>=` 也应当是 *传递的*。

如果定义了 `==`，则通常也应该定义 `!=`，并用其中一个实现另一个。

`<` 关系应当定义出一个「严格弱序 (strict weak order)」，并且与 `==` 及 `!=` 兼容，即：`a != b` 意味着 `a < b` 或 `b < a`。

```cpp
// point.cpp
#include "point.h"
bool operator==(const Point& lhs, const Point& rhs) {
  return lhs.x_ == rhs.x_ && lhs.y_ == rhs._y;
}
bool operator!=(const Point& lhs, const Point& rhs) {
  return !(lhs == rhs);
}
bool operator<(const Point& lhs, const Point& rhs) {
  return lhs.x_ < rhs.x_ || lhs.x_ == rhs.x_ && lhs.y_ < rhs._y;
}
```

### 位运算符

## 必须重载为方法成员的运算符

### 赋值运算符
「赋值 (assignment) 运算符」应当返回 *对「左端项 (Left Hand Side, LHS)」的引用*。

```cpp
// point.cpp
#include "point.h"
Point& Point::operator=(double const (array&) [2]) {
  x_ = array[0];
  y_ = array[1];
  return *this;
}
```
```cpp
// client.cpp
#include "point.h"
auto point = Point(1.0, 2.0);
double array[2] = { 0.1, 0.2 };
point = array;
```

### 下标运算符
「下标 (subscript) 运算符」通常应定义两个版本：

- 普通成员函数：只能用于 non-`const` 对象，返回 *对内部数据的 non-`const` 引用*。
- [`const` 成员函数](./class#`const`-成员函数)：可以用于任何对象，返回 *对内部数据的 `const` 引用*。

```cpp
#include <cassert>
double& Point::operator[](int i) {
  if (i == 0) return _x;
  if (i == 1) return _y;
  assert(false); 
}
const double& Point::operator[](int i) const {
  if (i == 0) return _x;
  if (i == 1) return _y;
  assert(false); 
}
```

### 函数调用运算符
「函数调用 (function call) 运算符」可以在同一个类中重载多次，相互之间以形参的类型或数量来区分。
支持 `operator()` 的对象被称为[函数对象](./function.md#函数对象)，其行为可以通过设置其内部「状态 (state)」的方式来「订制 (customize)」。

```cpp
// point.h
class PointBuilder {
 public:
  PointBuilder(double x, double y) : base_point_(x, y) { }
  Point operato()(double x, double y) {
    return Point(x + base_point_.x(), y + base_point_.y());
  }
 private:
  Point base_point_;
}
```
```cpp
// client.cpp
#include "point.h"
auto builder = PointBuilder(4.0, 3.0);
auto point = builder(-4.0, -3.0);
assert(point.x() == 0.0);
assert(point.y() == 0.0);
```

### 成员访问运算符
「成员访问 (member access) 运算符」`operator->` 通常与[解引用运算符](#解引用运算符) `operator*` 成对地重载，用于模拟指针的行为。

`operator->` 的返回类型，可以是一个指针，或者是一个支持 `operator->` 的对象（例如：迭代器）。

```cpp
// point.h
class PointHandle {
 public:
  PointHandle(double x = 0.0, double y = 0.0) : point_(new Point(x, y)) { }
  ~PointHandle() { delete point_; }
  Point* operator->() const { return point_; }
  Point& operator*() const { return *point_; }
 private:
  Point* point_;
};
```
```cpp
// client.cpp
#include "point.h"
auto point_handle = PointHandle(1.0, 2.0);
assert(point_handle->x() == (*point_handle)[0]);
```

### 类型转换运算符 ⚠️
「类型转换 (type cast) 运算符」的函数名为 `operator TargetType`，形参列表为空，没有返回类型，通常应为 [`const` 成员函数](./class#`const`-成员函数)：

```cpp
// point.cpp
#include "point.h"
Point::operator bool() const {
  return x_ != 0.0 || y_ != 0.0;
}
```

与只需要一个实参的 non-[`explicit` 构造函数](./class.md#`explicit`-构造函数)类似，non-`explicit` 类型转换运算符 *可能* 被 *隐式地* 用于类型转换，从而绕开类型检查机制。
⚠️ 应当尽量避免隐式类型转换。

`explicit` 类型转换运算符 *只能* 被 *显式地* 调用，
唯一的例外是 `operator bool()`，即使被声明为 `explicit` 也可以被 *隐式地* 调用：

```cpp
// client.cpp
auto origin = Point();
auto point = Point(1.0, 2.0);
assert(origin == false);
assert(point == true);
```

⚠️ 除 `explicit operator bool()` 之外，应当避免重载类型转换运算符。

## 通常重载为方法成员的运算符

### 复合赋值运算符

### 自增自减运算符
如果要自增运算符 `operator++`（或自减运算符 `operator--`），通常定义两个版本：
- 「前置 (prefix) 版本」：返回新值的引用。
- 「后置 (suffix) 版本」：返回旧值的副本（不含引用）。

```cpp
// point.h
class PointIterator {
 public:
  Point& operator++();    // 前置版本
  Point operator++(int);  // 后置版本，其中 int 为占位符
};
```

### 解引用运算符
「解引用 (dereference) 运算符」`operator*` 通常与[成员访问运算符](#成员访问运算符) `operator->` 成对地重载，用于模拟指针的行为，其返回类型必须是一个引用。
