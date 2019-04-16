# 继承与动态绑定

## 基类 v. 派生类

| 名称 | 同义词 | 任务 |
| ---- | ---- | ---- |
| 基类 (base class) | 超类 (superclass) | 定义自己和派生类 **共有的 (common)** 成员 |
| 派生类 (derived class) | 子类 (subclass) | 定义自己 **特有的 (specific)** 成员 |

### 声明 v. 定义
一个类只有被 **定义** 过（而不仅仅被 **声明** 过）才能被用作基类：
```cpp
class Base;  // 声明
class Derived : public Base { };  // ❌ 用作基类的 Base 还没有被定义
```

派生列表只出现在派生类的 **定义** 中，而不出现在派生类的 **声明** 中：
```cpp
class Derived : public Base;  // ❌ 派生列表 只出现在 定义 中
class Derived;                // ✔️ 派生列表 不出现在 声明 中
```

### [构造函数](./class.md#构造函数)
派生类的构造函数必须用 **基类的构造函数** 来初始化派生类对象的 **基类部分**：
```cpp
class Point {
  double _x;
  double _y;
 public:
  Point(double x, double y) : _x(x), _y(y) { }
};
class Particle : public Point {
  double _mass;
 public:
  Particle(double x, double y, double mass)
      : Point(x, y) /* 用 基类构造函数 初始化 基类部分 */, _mass(mass) { }
};
```

### 静态数据成员
基类中的静态数据成员被继承体系中的所有派生类共享，既可以通过基类来访问，也可以通过派生类访问（前提是该基类成员对派生类可见）。

### 禁止继承
类名后面紧跟 `final` 关键词，表示该类不可以被用作基类：
```cpp
class NoDerived final { };
```

## 继承级别

### 成员访问修饰符

| 修饰符      | 派生类的成员或友元 | 其他对象 |
| ----------- | ---- | ---- |
| `public`    | 可访问 | 可访问 |
| `protected` | 可访问 | 不可访问 |
| `private`   | 不可访问 | 不可访问 |

派生类的成员或友元, 只能通过`派生类对象`来访问基类的 `protected` 成员.
派生类对象本身, 对于基类的 `protected` 成员, 没有特殊的权限.

```cpp
class Base {
 protected:
  int _protected_i;  // 基类的 protected 成员
};
class Derived : public Base {
  friend void visit(Derived& d);
  friend void visit(Base& b);
  int _private_i;  // 派生类的 private 成员
};
// ✔️ 可以通过 派生类对象 访问 基类的 protected 成员:
void visit(Derived &d) { d._private_i = d._protected_i = 0; }
// ❌ 只能通过 派生类对象 访问 基类的 protected 成员:
clobber(Base &b) { b._protected_i = 0; }
```
在派生类中, 尽管基类的 `public` 和 `protected` 成员可以被直接访问, 还是应该`使用基类的公共接口`来访问基类成员.

### 派生访问修饰符
派生访问修饰符用于规定`派生类的客户` (包括`下一级派生类`) 对`继承自基类的成员`的访问权限:
- 基类的 `private` 成员无法被派生类访问, 因此不受继承级别的影响.
- 基类的 `public` 和 `protected` 成员可以被派生类访问, 因此受继承级别的影响:
  - `public` 继承:
    - 基类的 `public` 成员 → 派生类的 `public` 成员
    - 基类的 `protected` 成员 → 派生类的 `protected` 成员
  - `protected` 继承: 基类的 `public` 和 `protected` 成员 → 派生类的 `protected` 成员
  - `private` 继承: 基类的 `public` 和 `protected` 成员 → 派生类的 `private` 成员

### 改变访问级别
基类的 `public` 和 `protected` 成员在派生类中可以用 `using 声明`改变其访问级别:
```cpp
class Base {
 public:
  std::size_t size() const { return n; }
 protected:
  std::size_t n;
};
// private 继承, 基类的 public 和 protected 成员 → 派生类的 private 成员:
class Derived : private Base {  
 public:
  using Base::size;  // 提升为 派生类的 public 成员
 protected:
  using Base::n;     // 提升为 派生类的 protected 成员
};
```

### 默认的继承级别

| 关键词  | 默认的继承级别 |
| ------ | ---------------------- |
| `struct` | `public` |
| `class` | `private` |

为提高代码可读性, 应当`显式`写出继承级别.

### 友元关系不被继承
基类的友元与派生类的友元是相互独立的, 必须为每个类独立声明各自的友元.

## 自动类型转换

### 静态类型与动态类型
`静态类型`是指一个变量在声明时所使用的类型, 或者一个表达式的计算结果的类型, 在`编译期`就可以确定.

`动态类型`是指一个变量或表达式所代表的内存中的实际对象的类型, 可能要到`运行期`才能够确定.

对于指针或引用, 动态类型`可能`与静态类型`不同`; 
对于其他表达式, 动态类型`总是`与静态类型`相同`.

### 派生类到基类
一个派生类对象的非静态数据成员可以分为两类: 
- 派生类自己定义的非静态数据成员
- 继承自基类的非静态数据成员

#### 派生类指针或引用 到 基类指针或引用
可以用`指向基类对象`的指针 (包括智能指针) 或引用来`指向派生类对象`.
这种自动类型转换仅适用于指针或引用, 并且其可访问性与基类的 `public` 成员相同.

用容器管理继承体系的对象时, 几乎总是以`基类指针` (包括智能指针) 为元素类型, 然后将`派生类指针`存储到的容器中:
```cpp
std::vector<std::shared_ptr<Base>> vec;
auto pd = std::make_shared<Derived>(/* ... */);
vec.push_back(pd);  // std::shared_ptr<Derived> 自动转换为 std::shared_ptr<Base>
```

#### 派生类对象 到 基类对象
`基类对象`可以用`派生类对象`来**初始化**或`赋值`.
此时, 派生类对象中的`基类部分`将被 copy 或 move, 而`派生类特有的部分`将被忽略.

### 基类到派生类
#### 不存在自动转换
一个基类指针或引用, 可能指向一个`基类对象`, 也可能指向一个`派生类对象`.
因此没有基类到派生类的自动转换:
```cpp
Base b;
Derived* pd = &b;  // 错误
Derived& rd = b;   // 错误
```
即使一个`基类指针或引用`的确指向一个`派生类对象`, 这种自动转换也不存在:
```cpp
Derived d;
Base* pb = &d;     // ✔️ 可以从 Derived* 转换到 Base*
Derived* pd = pb;  // ❌ 无法从 Base* 转换到 Derived*
```

#### 强制类型转换 (慎用)
如果基类定义了虚函数, 则可以用 `std::dynamic_cast` 来执行`基类指针或引用`到`派生类指针或引用`的动态类型转换, 该转换将在`运行期`发生.

一般情况下, 可以用 `std::static_cast` 来执行静态类型转换, 该转换将在`编译期`发生.

## 虚函数与动态绑定
### 声明与定义
如果基类期望某个成员函数被派生类`重写 (override)`, 则需要在`声明的前端`用关键词 `virtual` 进行修饰.
关键词 `virtual` 只在类的内部对虚函数进行 **声明** 时使用.
如果一个成员函数在基类中被声明为虚的, 那么它在派生类中也将是虚的 (即使没有用关键词 `virtual` 进行声明).

虚函数在基类和派生类中必须具有相同的`函数签名`, 即`形参类型`和`返回类型`必须相同.
由于存在动态绑定机制, 对于返回类型允许存在一个例外: 如果虚函数在基类中的返回类型是`指向基类的指针`, 那么它在派生类中的返回类型可以是`指向派生类的指针`.

```cpp
class Shape {
 public:
  virtual ~Shape() { std::cout << "~Shape()" << std::endl; }
  virtual double getArea() const { return 0.0; };
};
class Rectangle : public Shape {
 public:
  Rectangle(const double& length, const double& width)
      : _length(length), _width(width) { }
  ~Rectangle() { std::cout << "~Rectangle()" << std::endl; }    
  virtual double getArea() const override {
    return _length * _width; 
  }
 private:
  double _length;
  double _width;
};
class Circle : public Shape {
 public:
  Circle(const double& x, const double& y, const double& r)
      : _x(x), _y(y), _r(r) { }
  ~Circle() { std::cout << "~Circle()" << std::endl; }    
  virtual double getArea() const override { 
    return _r * _r * 3.141592653589793; 
  }
 private:
  double _x;
  double _y;
  double _r;
};
```

#### 纯虚函数与抽象基类
通常, 编译器无法判断一个虚函数是否会被用到, 因此必须为每一个虚函数给出 **定义** , 而不仅仅是 **声明** .
作为例外, 纯虚函数表示`抽象操作`, 其 **定义** 不需要在基类中给出, 而是延迟到派生类中.
 **声明** 纯虚函数只需要在声明的末尾加上 `= 0`.

含有纯虚函数的类称为`抽象基类`, 表示派生类的`抽象接口`.
抽象基类无法创建对象.

### 动态绑定 (Dynamic Binding)
`动态绑定`是指: 如果一个`指向基类的指针或引用`实际指向的是一个`派生类对象`, 那么`通过它调用虚函数`将会在`运行期`被解析为`调用相应的派生类版本`.
```cpp
int main() {
  auto pShape = std::make_unique<Shape>();
  std::cout << pShape->getArea() << std::endl;  // 调用 Shape::getArea()
  auto pRectangle = std::make_unique<Rectangle>(4.0, 3.0);
  pShape.reset(pRectangle.release());  // 用指向 Shape 的指针接管 Rectangle 对象
  std::cout << pShape->getArea() << std::endl;  // 调用 Rectangle::getArea()
  auto pCircle = std::make_unique<Circle>(0.0, 0.0, 1.0);
  pShape.reset(pCircle.release());  // 用指向 Shape 的指针接管 Circle 对象
  std::cout << pShape->getArea() << std::endl;  // 调用 Circle::getArea()
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

#### 基类的析构函数
为支持动态资源管理, 基类应将`析构函数`声明为虚的, 这样编译器才能够知道应该调用哪个版本的 `delete`.

#### 绕开动态绑定机制
可以用 `::` 显式指定虚函数版本:
```cpp
int main() {
  auto pShape = std::make_unique<Shape>();
  auto pRectangle = std::make_unique<Rectangle>(4.0, 3.0);
  pShape.reset(pRectangle.release());  // 用指向 Shape 的指针接管 Rectangle 对象
  std::cout << pShape->getArea() << std::endl;  // 调用 Rectangle::getArea()
  std::cout << pShape->Shape::getArea() << std::endl;  // 调用 Shape::getArea()
}
```
输出
```cpp
~Shape()  // 调用 reset(), 此时管理的是 Shape 对象, 因此调用 ~Shape()
12  // 调用 Rectangle::getArea()
0   // 调用 Shape::getArea()
~Rectangle()  // pShape 离开作用域, 此时管理的是 Rectangle 对象, 因此调用 ~Rectangle()
~Shape()      // ~Rectangle() 会调用 ~Shape()
```

### (C++11) `override` --- 检查函数签名
在派生类中定义一个与基类中的虚函数`同名`但`形参列表不同`的成员函数是合法的.
编译器会将其视作与虚函数无关的一个新的成员.

为了避免在重写虚函数时无意中写错形参列表, 可以用在虚函数的形参列表 (包括 `const`) 后面紧跟关键词 `override`.
如果在派生类中被标注为 `override` 的虚函数与基类版本的虚函数具有不同的形参列表, 编译器将会报错.

### (C++11) `final` --- 禁止重写
如果要禁止派生类对一个虚函数进行重写, 可以在基类中的虚函数形参列表 (包括 `const`) 后面紧跟关键词 `final`:
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

### 默认实参 (少用)
如果通过`指向基类`的指针或引用调用虚函数, 将会使用`基类版本`的默认实参.
因此, 派生类版本的虚函数应当使用与基类版本相同的默认实参.
