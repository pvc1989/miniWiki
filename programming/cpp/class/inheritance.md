# 继承与多态

## 基类 v. 派生类

| 名称 | 同义词 | 任务 |
| ---- | ---- | ---- |
| 基 (base) 类 | 超 (super) 类 | 定义自己和派生类『共有的 (common)』成员 |
| 派生 (derived) 类 | 子 (sub) 类 | 定义自己『特有的 (specific)』成员 |

### 声明 v. 定义
一个类只有被 *定义* 过（而不仅仅被 *声明* 过）才能被用作基类：
```cpp
class Base;  // 声明
class Derived : public Base { };  // ❌ 用作基类的 Base 还没有被定义
```

*派生列表* 只出现在派生类的 *定义* 中，而不出现在派生类的 *声明* 中：

```cpp
class Derived : public Base;  // ❌ 派生列表 只出现在 定义 中
class Derived;                // ✅ 派生列表 不出现在 声明 中
```

### 构造函数
派生类的[构造函数](./class.md#构造函数)必须用 *基类的构造函数* 来初始化派生类对象的 *基类部分*：
```cpp
class Point {
  double x_;
  double y_;
 public:
  Point(double x, double y) : x_(x), y_(y) { }
};
class Particle : public Point {
  double mass_;
 public:
  Particle(double x, double y, double mass)
      : Point(x, y) /* 用『基类构造函数』初始化『基类部分』*/, mass_(mass) { }
};
```

### `static` 数据成员
基类中的[静态数据成员](./class#`static`-成员)被继承体系中的所有派生类共享，既可以通过基类来访问，也可以通过派生类访问（前提是该基类成员对派生类可见）。

### `friend` 不被继承
基类与派生类的 `friend` 是相互独立的，必须为每个类独立声明 `friend`。

### 禁止继承
类名后面紧跟 `final` 关键词，表示该类不可以被用作基类：
```cpp
class NoDerived final { };
```

## 继承级别

### 访问级别
除了 [`public` 和 `private`](./class.md#访问控制)，基类还可以用 `protected` 来实现更精细的访问控制：

| 修饰符      | 派生类的成员或派生类的 `friend` | 其他对象 |
| ----------- | ---- | ---- |
| `public`    | 可访问 | 可访问 |
| `protected` | 可访问（只能通过 *派生类对象* 来访问） | 不可访问 |
| `private`   | 不可访问 | 不可访问 |

```cpp
class Base {
 protected:
  int protected_;  // 基类的 protected 成员
};
class Derived : public Base {
  friend void visit(Derived& d);
  friend void visit(Base& b);
  int private_;  // 派生类的 private 成员
};
// ✅ 可以通过『派生类对象』访问『基类的 protected 成员』:
void visit(Derived &d) { d.private_ = d.protected_ = 0; }
// ❌ 不能通过『基类对象』访问『基类的 protected 成员』:
visit(Base &b) { b.protected_ = 0; }
```

⚠️ 尽管在派生类中可以直接访问基类的 `protected` 成员，还是应该尽量使用基类的 `public` 接口。

### 继承保护级别
『派生访问修饰符 (derivation access specifier)』用于规定『继承保护级别 (inheritance protection level)』，即规定派生类的『使用者 (client)』（包括 *下一级派生类*）对 *继承自基类的成员* 的访问权限：

- 基类的 `private` 成员无法被派生类访问，不受继承级别的影响。
- 基类的 `public` 和 `protected` 成员可以被派生类访问，受继承级别的影响：
  - `public` 继承：基类的 `public`/`protected` 成员 → 派生类的 `public`/`protected` 成员。
  - `protected` 继承：基类的 `public`/`protected` 成员 → 派生类的 `protected` 成员。
  - `private` 继承：基类的 `public`/`protected` 成员 → 派生类的 `private` 成员。

### 改变访问级别
在派生类中，可以用 `using` 改变基类 `public`/`protected` 成员的访问级别：
```cpp
class Base {
 public:
  std::size_t size() const { return n; }
 protected:
  std::size_t n;
};
// private 继承，基类的 public 和 protected 成员 → 派生类的 private 成员：
class Derived : private Base {  
 public:
  using Base::size;  // 提升为 派生类的 public 成员
 protected:
  using Base::n;     // 提升为 派生类的 protected 成员
};
```

### 默认继承级别
`class` 与 `struct` 的区别仅仅体现在[默认访问级别](./class.md#默认访问级别)和默认继承级别上：

| 关键词  | 默认继承级别 | 默认访问级别 |
| ------ | ----------- | ----------- |
| `struct` | `public` | `public` |
| `class` | `private` | `private` |

为提高代码可读性，应当 *显式* 写出访问级别和继承级别。

## 自动类型转换

### 静态类型与动态类型
- 『静态 (static) 类型』是指一个变量在 *声明* 时所使用的类型，或者一个表达式的计算结果的类型，在『编译期 (compile-time)』就可以确定。
- 『动态 (dynamic) 类型』是指一个变量或表达式所代表的内存中的实际对象的类型，可能要到『运行期 (run-time)』才能够确定。

只有 *指针* 或 *引用* 的动态类型 *可能* 与静态类型不同。

### 派生类到基类的转换
一个派生类对象的非静态数据成员可以分为两类： 
- 派生类自己定义的非静态数据成员。
- 继承自基类的非静态数据成员。

可以用指向 *基类对象* 的指针（包括智能指针）或引用来指向 *派生类对象*。
这种转换仅适用于指针或引用，并且这种转换的『可访问性 (accessibility)』与基类的 `public` 成员相同。

用容器管理 *派生类对象* 时，几乎总是以 *基类指针* 为容器元素类型，然后将 *派生类指针* 存储到的容器中：
```cpp
auto vec = std::vector<std::shared_ptr<Base>>();
auto pd = std::make_shared<Derived>(/* ... */);
vec.push_back(pd);  // std::shared_ptr<Derived> 自动转换为 std::shared_ptr<Base>
```

*基类对象* 可以用 *派生类对象* 来 *初始化* 或 *赋值*。
此时，派生类对象中的 *基类部分* 将被『拷贝 (copy)』或『移动 (move)』，而派生类自己定义的非静态数据成员将被忽略。

### 基类到派生类的转换
一个基类指针或引用，可能指向一个 *基类对象*，也可能指向一个 *派生类对象*，
因此，不存在基类到派生类的自动转换：

```cpp
Base b;
Derived* pd = &b;  // ❌
Derived& rd = b;   // ❌
```
即使一个 *基类指针或引用* 的确指向一个 *派生类对象*，这种自动转换也不存在：
```cpp
Derived d;
Base* pb = &d;     // ✅ 可以从 Derived* 转换到 Base*
Derived* pd = pb;  // ❌ 无法从 Base* 转换到 Derived*
```

尽管不存在 *自动* 转换，但还是可以 *手动* 实现从基类到派生类的转换：
- 一般情况下，可以用 `static_cast<Derived>()` 来执行『静态类型转换 (static cast)』，该转换将在 *编译期* 发生。
- 如果基类定义了[虚函数](#虚函数)，则可以用 `dynamic_cast<Derived>()` 来执行 *基类指针或引用* 到 *派生类指针或引用* 的『动态类型转换 (dynamic cast)』，该转换将在 *运行期* 发生。
- ⚠️ 尽管存在以上语言支持，但其目的是为了兼容 C-style API。在 Modern C++ 中使用这些机制往往意味着设计存在缺陷。

## 虚函数
### 声明与定义
如果基类期望某个成员函数被派生类『重写 (override)』，则需要在 *声明的前端* 加上关键词 `virtual`。
关键词 `virtual` 只在类的内部 *声明* 虚函数时使用，在类的外部 *定义* 虚函数时用不到 `virtual`。
如果一个成员函数在基类中被声明为虚的，那么它在派生类中也将是虚的（即使在派生类中没有用 `virtual` 声明）。

虚函数在基类和派生类中必须具有相同的 *函数签名*，即 *形参类型* 和 *返回类型* 必须相同。
由于存在[动态绑定](#动态绑定)机制，对于返回类型允许存在一个例外：
如果虚函数在基类中的返回类型是 *基类的指针或引用*，那么它在派生类中的返回类型可以是 *派生类的指针或引用*。

```cpp
class Shape {
 public:
  virtual ~Shape() { std::cout << "~Shape()" << std::endl; }
  virtual double GetArea() const { return 0.0; };
};
class Rectangle : public Shape {
 public:
  Rectangle(const double& length, const double& width)
      : length_(length), width_(width) { }
  ~Rectangle() { std::cout << "~Rectangle()" << std::endl; }    
  virtual double GetArea() const override {
    return length_ * width_; 
  }
 private:
  double length_;
  double width_;
};
class Circle : public Shape {
 public:
  Circle(const double& x, const double& y, const double& r)
      : x_(x), y_(y), r_(r) { }
  ~Circle() { std::cout << "~Circle()" << std::endl; }    
  virtual double GetArea() const override { 
    return r_ * r_ * 3.141592653589793; 
  }
 private:
  double x_;
  double y_;
  double r_;
};
```

### 纯虚函数与抽象基类
编译器通常无法判断一个虚函数是否会被用到，因此必须为每一个虚函数给出 *定义*，而不仅仅是 *声明*。
作为例外，纯虚函数表示 *抽象操作*，其 *定义* 不需要在基类中给出，而是可以延迟到派生类中。
 *声明* 纯虚函数只需要在声明的末尾加上 `= 0`。

含有纯虚函数的类称为 *抽象基类*，表示派生类必须实现的 *抽象接口*。
抽象基类无法创建对象。

### 动态绑定
『动态绑定 (dynamic binding)』是指：
如果一个 *基类 (`Base`) 的指针或引用* 实际指向的是一个 *派生类 (`Derived`) 对象*，
那么 *用 `Base` 的指针或引用调用虚函数*，将会在 *运行期* 被解析为 *调用 `Derived` 实现的版本*。
```cpp
int main() {
  auto pShape = std::make_unique<Shape>();
  std::cout << pShape->GetArea() << std::endl;  // 调用 Shape::GetArea()
  auto pRectangle = std::make_unique<Rectangle>(4.0, 3.0);
  pShape.reset(pRectangle.release());  // 用指向 Shape 的指针接管 Rectangle 对象
  std::cout << pShape->GetArea() << std::endl;  // 调用 Rectangle::GetArea()
  auto pCircle = std::make_unique<Circle>(0.0, 0.0, 1.0);
  pShape.reset(pCircle.release());  // 用指向 Shape 的指针接管 Circle 对象
  std::cout << pShape->GetArea() << std::endl;  // 调用 Circle::GetArea()
}
```
输出
```cpp
0
~Shape()  // 第一次 reset(), 此时管理的是 Shape 对象, 因此调用 ~Shape()
12
~Rectangle()  // 第二次 reset(), 此时管理的是 Rectangle 对象, 因此调用 ~Rectangle()
~Shape()      // ~Rectangle() 会调用 ~Shape()
3.14159
~Circle()  // pShape 离开作用域, 此时管理的是 Circle 对象, 因此调用 ~Circle()
~Shape()   // ~Circle() 会调用 ~Shape()
```

### 虚析构函数
为支持[动态资源管理](../memory/README.md)，基类的析构函数应当被声明为 `virtual`，这样编译器才能够知道应该调用哪个版本的析构函数。

### 绕开动态绑定 ⚠️
可以用『作用域 (scope) 运算符』显式指定虚函数版本：
```cpp
int main() {
  auto pShape = std::make_unique<Shape>();
  auto pRectangle = std::make_unique<Rectangle>(4.0, 3.0);
  pShape.reset(pRectangle.release());  // 用指向 Shape 的指针接管 Rectangle 对象
  std::cout << pShape->GetArea() << std::endl;  // 调用 Rectangle::GetArea()
  std::cout << pShape->Shape::GetArea() << std::endl;  // 调用 Shape::GetArea()
}
```
输出
```cpp
~Shape()  // 调用 reset(), 此时管理的是 Shape 对象, 因此调用 ~Shape()
12  // 调用 Rectangle::GetArea()
0   // 调用 Shape::GetArea()
~Rectangle()  // pShape 离开作用域, 此时管理的是 Rectangle 对象, 因此调用 ~Rectangle()
~Shape()      // ~Rectangle() 会调用 ~Shape()
```

### `override`
在派生类中定义一个与基类中的虚函数 *同名* 但 *形参列表不同* 的成员函数是合法的。
编译器会将其视作与虚函数无关的一个新的成员，这属于函数『重载 (overload)』。

自 C++11 起，在形参列表（包括形参列表后的 `const` 修饰符）后加上关键词 `override`，
可以让编译器检查被『重写 (override)』的虚函数是否与基类的版本具有相同的形参列表。

### `final`
自 C++11 起，在基类中的虚函数形参列表（包括形参列表后的 `const` 修饰符）后加上关键词 `final`，
表示禁止派生类『重写 (override)』该虚函数：
```cpp
class Shape {
 public:
  virtual ~Shape() = default;
  virtual void noOverride() const final { }
};
class Rectangle : public Shape {
 public:
  virtual void noOverride() const { }  // ❌ 禁止重写 final 函数
};
```

### 默认实参 ⚠️
如果用 *基类的指针或引用* 调用虚函数，将会使用 *基类版本* 的默认实参。
因此，派生类版本的虚函数应当使用与基类版本相同的默认实参。
