# 设计原则

|    缩写     |              全称               |    中译名    |
| :---------: | :-----------------------------: | :----------: |
| [SRP](#SRP) | Single Resposibility Principle  | 单一责任原则 |
| [OCP](#OCP) |     Open--Closed Principle      | 开放封闭原则 |
| [LSP](#LSP) |  Liskov Substitution Principle  | 里氏替换原则 |
| [ISP](#ISP) | Interface Segregation Principle | 接口分离原则 |
| [DIP](#DIP) | Dependency Inversion Principle  | 依赖倒置原则 |

这五条原则在英文对话中常被称为 *the SOLID principles*，因为它们的首字母恰好构成英文单词 *SOLID*。在 ***面向对象设计 (Object Oriented Design, OOD)*** 中，它们是指导 ***类 (class)*** 的设计的基本原则。如果将 *类* 的含义推广为 *一组耦合的函数及数据*，则这些原则可以被用在更一般的 ***模块 (module)*** 设计上。

## SRP

SRP 的原始定义为：

> A class should have one, and only one, reason to change.

其中 *reason to change* 是指被称作 ***actor*** 的 *a group of people who require a change*，因此 SRP 可以重新表述为：

> A module should be responsible to one, and only one, actor.

违反 SRP 的类通常会有以下问题：
- 某个 actor 的修改影响其他 actor 的行为。
- 不同 actor 的修改在合并时发生冲突。

解决方案通常为：
- 将数据与函数分离，将函数按 actor 拆分为若干相互独立的小类。
- 如果上述拆分造成类的数量过多，可以用 [***Facade 模式***](../patterns/facade/README.md) 创建一个接口。

## OCP

该原则是 Bertrand Meyer (1988) 提出的：
> A software artifact should be open for extension, but closed for modification.

这里的 *software artifact* 可以是 ***类 (class)***、***模块 (module)***、***组件 (component)***。

遵循 OCP 的架构设计，通常会将 *系统* 按 *引起变化原因* 划分为若干 *组件*，并使组件之间的 *依赖* 符合 [DIP](#DIP)，从而实现 *高层业务逻辑* 不受 *低层实现细节* 变化的影响。

## LSP

> Subtypes must be substitutable for their base types.

该原则源自于 Barbara Liskov [1988] 对 ***子类 (subtype)*** 所作的定义：

> If for each object `o1` of type `S` there is an object `o2` of type `T` such that for all programs `P` defined in terms of `T`, the behavior of `P` is unchanged when `o1` is substituted for `o2` then `S` is a *subtype* of `T`.

这里的 *object* 在 C++ 中应当理解为 ***指针 (pointer)*** 或 ***引用 (reference)***。

### 语言实现机制
在 C++、Java、Python 等主流面向对象语言中，LSP 主要是通过 ***虚函数 (virtual function)*** 机制来实现的。

假设需要这样一个几何库：
- 所有几何类型都是 `Shape` 的派生类。
- 所有几何类型都要实现一个返回当前几何对象面积的 `GetArea()` 成员。
- 某个应用需要将 `GetArea()` 成员封装为一个 *非成员接口函数*，符合 LSP 的设计应当允许以下的简单实现：

```cpp
inline double GetArea(const Shape& shape) { return shape.GetArea(); }
```

下面给出三种设计方案：
- [原始设计 ⚠️](#原始设计-⚠️)
- [基于 RTTI 的设计 ⚠️](#基于-RTTI-的设计-⚠️)
- [基于虚函数设计](#基于虚函数的设计)

### 原始设计 ⚠️
不了解虚函数的新手可能会给出如下设计：
```cpp
struct Shape {
  double GetArea() const { return -1; }
};
struct Point : public Shape {
  double x;
  double y;
  Point(double a, double b) : x(a), y(b) { }
  double GetArea() const { return 0; }
};
class LineSegment : public Shape {
  Point head_;
  Point tail_;
 public:
  LineSegment(const Point& head, const Point& tail)
      : head_(head), tail_(tail) { }
  double GetArea() const { return 0; }
};
class Circle : public Shape {
  Point center_;
  double radius_;
 public:
  Circle(const Point& point, double radius)
      : center_(point), radius_(radius) { }
  double GetArea() const { return 3.1415926 * radius_ * radius_; }
};
```
该设计违反了 LSP：将 *派生类对象* 传递给接收 *基类引用* 的非成员接口函数，实际调用的总是 *基类的 `GetArea()` 成员*，得到的返回值总是 `-1`，从而无法通过下面的单元测试：
```cpp
#include <cassert>
int main() {
  auto p = Point(0, 0);
  auto q = Point(1, 0);
  assert(p.GetArea() == GetArea(p));
  auto ls = LineSegment(p, q);
  assert(ls.GetArea() == GetArea(ls));
  auto c = Circle(p, 2);
  assert(c.GetArea() == GetArea(c));
}
```

### 基于 RTTI 的设计 ⚠️
如果不使用[虚函数](#基于虚函数的设计)机制，往往会引入 ***运行期类型识别 (Run-Time Type Identification, RTTI)*** 或其他类似的机制。
一种实现方式：在基类中定义一个枚举成员 `type`，在派生类的构造函数中对其进行初始化，用于表示当前几何对象的类型。

```cpp
struct Shape {
  enum class Type { Shape, Point, LineSegment, Circle };
  const Type type;
  Shape(Type t) : type(t) { }
  double GetArea() const { return -1; }
};
struct Point : public Shape {
  double x;
  double y;
  Point(double a, double b)
      : Shape(Shape::Type::Point), x(a), y(b) { }
  double GetArea() const { return 0; }
};
class LineSegment : public Shape {
  Point head_;
  Point tail_;
 public:
  LineSegment(const Point& head, const Point& tail)
      : Shape(Shape::Type::LineSegment), head_(head), tail_(tail) { }
  double GetArea() const { return 1; }
};
class Circle : public Shape {
  Point center_;
  double radius_;
 public:
  Circle(const Point& point, double radius)
      : Shape(Shape::Type::Circle), center_(point), radius_(radius) { }
  double GetArea() const { return 3.1415926 * radius_ * radius_; }
};
```
在非成员接口函数的实现中，利用对象的动态类型信息，将任务转发到相应的成员函数：
```cpp
inline double GetArea(const Shape& shape) {
  switch (shape.type) {
    case Shape::Type::Point:
      return static_cast<const Point&>(shape).GetArea();
    case Shape::Type::LineSegment:
  	  return static_cast<const LineSegment&>(shape).GetArea();
    case Shape::Type::Circle:
  	 	return static_cast<const Circle&>(shape).GetArea();
    default:
      return shape.GetArea();
  }
}
```
该设计存在以下缺陷：
- 浪费资源：`type` 在每个对象中都要占据存储空间，对于需要生成大量几何对象的应用，这样的开销是不可忽视的。
- 违反 [OCP](#OCP)：引入新的派生类（例如 `Rectangle`）会迫使基类重新定义其枚举类型成员，并迫使非成员接口函数引入新的 `case`。
- 违反 [DIP](#DIP)：非成员接口函数依赖于所有具体的派生类。

### 基于虚函数的设计
如果将 *基类的 `GetArea()` 成员* 声明为虚函数，那么非成员接口函数中的 `shape.GetArea()` 将会在 ***运行期 (run-time)*** 自动转发到相应 *派生类的 `GetArea()` 成员*，从而有效地解决上述问题：
```cpp
struct Shape {
  virtual ~Shape() = default;
  virtual double GetArea() const { return -1; }
};
struct Point : public Shape {
  double x;
  double y;
  Point(double a, double b) : x(a), y(b) { }
  double GetArea() const override { return 0; }
};
class LineSegment : public Shape {
  Point head_;
  Point tail_;
 public:
  LineSegment(const Point& head, const Point& tail)
    	: head_(head), tail_(tail) { }
  double GetArea() const override { return 1; }
};
class Circle : public Shape {
  Point center_;
  double radius_;
 public:
  Circle(const Point& point, double radius)
    	: center_(point), radius_(radius) { }
  double GetArea() const { return 3.1415926 * radius_ * radius_; }
};
```

### 正方形 v. 长方形 ⚠️
派生机制不应过度使用，例如以下著名案例：
- 在几何意义上，所有的 `Square` 都是 `Rectangle`。
- 在程序设计中，将 `Square` 定义为 `Rectangle` 的派生类是一种糟糕的设计。

```cpp
#include <cassert>
struct Point {
  double x;
  double y;
  Point(double a, double b) : x(a), y(b) { }
};
class Rectangle {
  Point left_bottom_;
  double height_;
  double width_;
 public:
  Rectangle(Point& point, double height, double width)
      : left_bottom_(point), height_(height), width_(width) { }
  virtual ~Rectangle() = default;
  virtual void SetHeight(double height) { height_ = height; }
  virtual void SetWidth(double width) { width_ = width; }
  double GetArea() const { return height_ * width_; }
};
class Square : public Rectangle {
 public:
  Square(Point& point, double length) : Rectangle(point, length, length) { }
  virtual void SetHeight(double length) {
    Rectangle::SetHeight(length);
    Rectangle::SetWidth(length);
  }
  virtual void SetWidth(double length) {
    Rectangle::SetHeight(length);
    Rectangle::SetWidth(length);
  }
};
void Test(Rectangle& r, double height, double width) {
  r.SetHeight(height);
  r.SetWidth(width);
  assert(r.GetArea() == height * width);
}
int main() {
  auto p = Point(0, 0);
  auto r = Rectangle(p, 3, 5);
  Test(r, 3, 5);  // passed
  auto s = Square(p, 4);
  Test(s, 3, 5);  // failed
}
```

该设计存在以下缺陷：

- 浪费资源：`Square` 的 `height_` 总是等于 `width_`，不需要像 `Rectangle` 那样存储为两个独立成员。
- 违反 LSP：所有 `Rectangle` 都应当能通过单元测试 `Test(Rectangle&, int, int)`，但当 `height != width` 时 `Square` 却无法通过该测试。因此，从行为上看，a `Square` is *NOT* a `Rectangle`。

## ISP

> Clients should not be forced to depend on methods that they do not use.

## DIP

> High-level modules should not depend on low-level modules. Both should depend on abstractions.
> Abstractions should not depend on details. Details should depend on abstractions.

### 倒置的含义
*依赖倒置* 首先是指 *源代码依赖关系*（通常表现为 `#include` 或 `import` 语句）与 *程序控制流* 的倒置。

*依赖关系* 的倒置通常也意味着 *接口所有权* 的倒置：
接口代表一种服务，其所有权应当归属于服务的 *使用者（高层策略模块）* 而非 *提供者（底层实现模块）*。

下面的示例体现了 *依赖关系* 和 *接口所有权* 的双重倒置：

![](./inverted_dependency.svg)

- 在高层模块 `Application` 中，`TaskScheduler` 的 `addTask(), popTask()` 方法用到了 `PriorityQueue` 接口所提供的 `push(), top(), pop()` 服务；而实现这些服务所用到的算法，并不需要暴露给 `TaskScheduler`。
- 在中层模块 `Algorithm` 中，`BinaryHeap` 借助于 `Vector` 接口的 `at()` 服务，实现了 `PriorityQueue` 接口；而由 `Vector` 隐式提供的 `resize()` 服务，并不需要暴露给 `BinaryHeap`。
- 在底层模块 `DataStructure` 中，`DynamicArray` 借助于更底层的（通常由操作系统提供的）动态内存管理服务，实现了 `Vector` 接口的 `resize()` 服务。

### 语言实现机制

接口可以显式地出现在源代码中，例如：
- 在 Java 中，接口通常表现为 `interface` 或 `abstract class`。
- 在 C++ 中，接口可以表现为含有 *纯虚函数* 的 `class`。
- 在 C++20 中，接口可以表现为 [`concept`](https://en.cppreference.com/w/cpp/language/constraints)。
- 在 Python 3.4+ 中，接口可以表现为借助于标准库模块 [`abc`](https://docs.python.org/3/library/abc.html) 定义的 *抽象类*。

接口也可以仅仅作为一种抽象的 *概念*，例如：
- 在 C++ 中，接口可以表现为对 *模板类型形参* 的隐式约束，凡是满足该约束的 *类型* 都可以被视作是该接口的一个实现。
- 在 Python 等动态语言中，接口可以表现为对 *函数形参类型* 的隐式约束，凡是满足该约束的 *类型* 都可以被视作是该接口的一个实现。

### 面向对象设计 v. 面向对象语言
使用面向对象语言 (C++/Java/Python) 进行编程 *不等于* 面向对象编程。
一段程序是否是面向对象的，取决于程序中的依赖关系是否是倒置的，而与所使用的编程语言无关。
这种依赖关系的倒置是通过 ***多态 (polymorphism)*** 来实现的，多态既可以是 *静态的（编译期绑定）*，也可以是 *动态的（运行期绑定）*。
面向对象语言通过一定的语法机制，让多态变得更容易、更简洁、更安全。

