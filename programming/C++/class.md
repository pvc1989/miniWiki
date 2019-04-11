# 类

## 数据抽象

### 抽象数据类型
**类 (Class)** 机制主要用来定义新的 **抽象数据类型 (Abstract Data Type)**。

这里的 **抽象** 体现在：
- 类的 **客户 (client)** 只需要了解并且只能访问类的 **接口 (interface)**：
  - 包括 **公共方法成员** 和 **公共数据成员**，以及二元运算符等 **非成员接口函数**，
  - 通常在 **头文件 (header)** 中以 **源代码** 形式给出 **声明 (declaration)**。
- 类的 **设计者 (designer)** 负责提供类的 **实现 (implementation)**：
  - 包括 **私有方法成员** 和 **私有数据成员**，以及成员方法和非成员接口函数的 **定义 (definition)**，
  - 通常在 **源文件 (source)** 中以 **源代码** 形式给出，也可以只提供编译生成的 **目标文件/静态库/动态库**。

这样做的好处是：
- client 只依赖于 interface，而不依赖于 implementation：
  - designer 不需要将算法细节暴露给 client，有助于保护知识产权。
  - client 和 implementation 可以同时、独立地开发和测试。
  - implementation 发生变化时，不需要重新 **编译 client**，而只需要重新 **编译 implementation**，并将新的目标文件 **链接进 client**。
- 有助于减少不同类之间的依赖，允许各自独立变化。

### 访问控制
#### 访问修饰符 (Access Specifiers)
一个类可以含有零个或多个访问修饰符，每种访问修饰符出现的 **次数** 和 **顺序** 不限。
每个修饰符的作用范围起始于自己，终止于下一个修饰符或类的末尾。

| 访问修饰符 | 从类的外部 | 从类的内部 |
| -------- | -------- | -------- |
| `public` | 可以直接访问 | 可以直接访问 |
| `private` | (除友元外) 无法直接访问 | 可以直接访问 |

#### `class` v. `struct`
这两个关键词都可以用来定义一个类，对于访问控制，二者的区别仅在于：

| 关键词  | 隐含的第 `0` 个访问修饰符 |
| ------ | ---------------------- |
| `struct` | `public`  |
| `class`  | `private` |

#### `friend`（少用）
定义一个类时，可以用 `friend` 将其他（可见的）类或函数声明为它的 **友元**，从而允许这些友元访问其私有成员。
**友元声明** 不是 **函数声明**。
通常，将友元声明集中放在类定义的头部或尾部。

⚠️ 友元机制破坏了类的封装，因此要少用。

### 类型成员
一个类可以含有类型成员，可以是已知类型的 **别名**，也可以是定义在其内部的 **嵌套类 (nested class)**。
类型成员必须在使用前被定义，因此通常将它们集中定义在类的头部。
类型成员与[数据成员](#数据成员)和[函数成员](#函数成员)遵循相同的[访问控制](#访问控制)规则：
```cpp
class Screen {
 public:
  typedef std::string::size_type Position;
 private:
  Position _cursor = 0;
  Position _height = 0;
  Position _width = 0;
};
```

### 数据成员

### 函数成员
#### 声明 v. 定义
所有成员函数都必须在类的内部（通常位于 **头文件** 中）进行 **声明**，但其 **定义** 可以放在类的外部（通常位于 **源文件** 中）。

#### `this` 指针
除[静态成员函数](#`static`-成员)外，所有成员函数都是通过隐式指针 `this` 来访问调用它的那个对象的。
```cpp
SalesData total;
total.isbn()
// s相当于
SalesData::isbn(&total)
```

#### `const` 成员函数
默认情况下，`this` 是指向 non-`const` 对象的指针，这使得相应的成员函数无法被 `const` 对象调用。
如果要使 `this` 为指向 `const` 对象的指针，只需要在函数形参列表后面紧跟 `const` 关键词。

#### `inline` 成员函数
成员函数可以是 **内联的 (inline)**：
- 定义在类的 **内部** 的成员函数是 **隐式** 内联的。
- 定义在类的 **外部** 的成员函数也可以是内联的，只需要在（位于同一头文件中的）函数 **定义** 前加上 `inline` 关键词。

### `static` 成员
**静态的 (static)** 成员由一个 **类** 的所有 **对象** 共享，而不属于其中任何一个对象：
- 静态 **数据** 成员存储于所有对象的外部，不计入对象的大小。
- 静态 **函数** 成员独立于所有对象，形参列表中没有隐含 [`this` 指针](#`this`-指针)。

#### 访问静态成员
在类的外部，静态成员可以通过紧跟在 **类名** 后面的作用域运算符 `::` 来访问，也可以（像非静态成员一样）通过 **对象** 或指向该对象的 **指针** 或 **引用** 来访问。

在类的内部，静态成员可以被所属类的成员函数直接访问，而不需要借助于作用域运算符。

#### 定义静态成员
关键词  `static` 仅用于在类的内部 **声明** 静态成员，而不需要在类的外部 **定义** 静态成员时重复。

静态数据成员必须在类的 **外部** 进行 **定义** 和 **初始化**。
与 **非内联成员函数** 类似，每个静态数据成员都只能被定义一次，因此应当将它们的定义放在同一个 **源文件** 中。

定义静态数据成员时，可以访问该类的 **私有** 成员。

#### (C++11) 类内初始化
通常，静态数据成员 **不可以** 在类的内部进行初始化，但有两个例外：
- **可以** 为 `static const` **整型** 数据成员指定类内初始值。
- **必须** 为 `static constexpr` 数据成员指定类内初始值。

用作类内初始值的表达式必须是 `constexpr`，被其初始化的静态数据成员也是 `constexpr`，可以用于任何需要 `constexpr` 的地方：
```cpp
// account.h
class Account {
 private:
  static constexpr int kLength = 30;  // kLength 是 constexpr
  double table[kLength];  // 数组长度必须是 constexpr
};
```
即使一个静态数据成员已经在类内被初始化，通常也应在类外给出定义。
如果其初始值已经在类内给定，则类外不得再给定初始值：
```cpp
// account.cpp
constexpr int Account::kLength;
```

#### 特殊用法
静态数据成员的类型可以是它自己所属的那个类：
```cpp
class Point {
 private:
  static Point _p1;  // 正确: 静态数据成员 可以是 不完整类型
  Point* _p2;        // 正确: 指针成员 可以是 不完整类型
  Point  _p3;        // 错误: 非静态数据成员 必须是 完整类型
};
```
静态数据成员可以（在声明前）被用作默认实参：
```cpp
class Screen {
 public:
  Screen& clear(char = kBackground);
 private:
  static const char kBackground;
};
```

### 构造函数 (Constructor)
构造函数是一种`以类名作为函数名`的特殊成员函数, 用于构造该类的对象.
在构造过程中, 需要修改数据成员的值, 因此构造函数不可以被声明为 `const`.

与一般的函数不同, 构造函数`没有返回类型`.

#### 默认构造函数
`不接受实参`的构造函数称为`默认 (default)` 构造函数.
默认构造函数可以是`形参列表为空`的构造函数, 也可以是`所有形参都有默认实参值`的构造函数.

##### 合成的默认构造函数
如果没有`显式地`定义任何构造函数, 那么编译器会`隐式地`定义一个`合成的 (synthesized)` 默认构造函数.
C++11 允许程序员显式地使用编译器合成的版本, 只需要在定义时在 (空) 形参列表后紧跟 `= default;` 即可.

#### 初始化列表
```cpp
class Point {
  double _x;
  double _y;
};
```
`初始化列表`位于 `:` 与 `{` 之间, 用于对数据成员进行`值初始化`:
```cpp
// 推荐: 在 初始化列表 中进行 值初始化
Point::Point(const double& x, const double& y) : _x(x), _y(y) { }
```
如果某个数据成员没有在`初始化列表`中被初始化, 则会被`默认初始化`, 然后才会被函数体内的赋值语句修改:
```
// 语义相同, 但 默认初始化 过程浪费了计算资源
Point::Point(const double& x, const double& y) {
  _x = x;
  _y = y;
}
```

如果某个成员的值在初始化之后无法被修改, 则必须在`初始化列表`中对其进行初始化:
- 该成员是 `const`
- 该成员是`引用`类型
- 该成员无法被`默认初始化` (即: 没有默认构造函数)

##### 成员构造的顺序
成员构造的顺序不是它们出现在`初始化列表`中的顺序, 而是它们出现在`类的定义`中的顺序.

##### 委托构造函数
一个构造函数只需要在`初始化列表`中`调用`另一个构造函数, 就可以将构造任务`委托`给另一个构造函数, :
```cpp
class Point {
 public:
  Point(const double& x, const double& y) : _x(x), _y(y) { }
  Point() : Point(0.0, 0.0) { }
 private:
  double _x;
  double _y;
};
```

#### 显式 (`explicit`) 构造函数
默认情况下, 只需要传入`一个实参`的构造函数`隐式地`定义了一种由`形参类型`到该`类类型`的转换.
编译器只会进行一次隐式类型转换.

如果要禁止这种隐式类型转换, 只需要在构造函数的声明`前`, 用关键词 `explicit` 进行修饰.
例如:
```cpp
namespace std{
template <
  class T, 
  class Allocator = std::allocator<T>
> class vector {
 public:
  explicit vector(std::size_type count);  // 禁止用于隐式类型转换
  // ...
};
}
```

## 运算符重载

### 常用运算符

### 函数调用运算符

### 类型转换

## 拷贝控制

### 析构 (Destroy)
#### 析构函数 (Destructor)
析构函数是一种`以 ~ 为前缀的类名作为函数名`的特殊成员函数, 用于析构 (销毁) 该类的对象.
析构函数`没有返回类型`, 并且`形参列表为空`.
```cpp
class Foo {
 public:
  ~Foo();  // destructor
  // ...
};
```

#### 成员析构的顺序
一个对象被析构时, 先执行其所属类型的析构函数体中的语句, 再隐式地析构其 (非静态) 数据成员.
数据成员被析构的顺序与它们被构造的顺序`相反`, 即: 与它们在类的定义中出现的顺序相反.

#### 合成的析构函数
如果析构函数没有被`显式地声明`, 那么编译器会`隐式地定义`一个默认的版本, 称为`合成的析构函数`.
C++11 允许程序员显式地使用编译器合成的版本, 只需要在定义时在 (空) 形参列表后紧跟 `= default;` 即可.

合成的析构函数, 只会`逐个析构`数据成员 --- 这意味着: 不会对`裸指针`成员调用 `delete`.

### 拷贝 (Copy)
#### 拷贝构造函数 (Copy Constructor)
拷贝构造函数是一类特殊的构造函数:
- `第一个形参`为`指向该类型对象的引用`, 并且几乎总是`指向常量的引用`.
- `其余形参`均有`默认实参值`.
```cpp
class Foo {
 public:
  Foo(const Foo&);
  // ...
};
```
在许多场合, 拷贝构造函数会被`隐式地`调用, 因此通常不应设为 `explicit`.

#### 拷贝赋值运算符 (Copy Assignment Operator)
拷贝赋值运算符是对成员函数 `operator=` 的重载, 函数签名几乎总是如下形式:
- 唯一的 (显式) 形参的类型为: 指向`常值`对象的引用.
- 返回类型为: 指向`非常值`对象的引用.
```cpp
class Foo {
 public:
  Foo& operator=(const Foo&);
  // ...
};
```

#### (C++11) 删除的拷贝操作
有些类型的对象不应支持拷贝操作 (例如 `std::iostream`).
为了实现该语义, (在 C++11 下) 只需在声明`拷贝构造函数`和`拷贝赋值运算符`时, 将它们标注为`删除的 (deleted)`:
```cpp
class Foo {
 public:
  Foo(const Foo&) = delete;
  Foo& operator=(const Foo&) = delete;
  // ...
};
```

#### 合成的拷贝操作
合成的拷贝操作 (`拷贝构造函数`和`拷贝赋值运算符`) 会`逐个拷贝`数据成员 --- 这意味着: 只会对`裸指针`进行`浅拷贝`.

如果含有数组成员, 则合成的拷贝操作会`逐个拷贝成员数组的每一个元素`.

如果一个类含有`无法拷贝的`数据成员, 则这个类本身也应当是`无法拷贝的`, 因此合成的拷贝操作将是`删除的`.

### 移动 (Move)

#### 右值
类型名后紧跟 `&&` 表示定义一个指向该类型对象的`右值引用`:
- `右值引用` (通常) 只能绑定到`即将被析构的对象`上.
- 定义在 `<utility>` 中的库函数 `std::move` 可以将`左值表达式`变为`右值表达式`.

通常, `左值表达式`代表对象的`身份 (identity)`, 而`右值表达式`代表对象的`值 (value)`.
`右值引用`作为一个表达式是一个`左值表达式`:
```cpp
int i = 42;
int& r = i;              // 正确: 将 左值引用 绑定到 左值表达式
int& r2 = i * 42;        // 错误: 普通左值引用 无法绑定到 右值表达式
const int& r3 = i * 42;  // 正确: 指向常量的左值引用 可以绑定到 右值表达式
int&& rr = i;                // 错误: 右值引用 无法直接绑定到 左值表达式
int&& rr2 = i * 42;          // 正确: 将 右值引用 绑定到 右值表达式
int&& rr3 = std::move(rr2);  // 正确: std::move 将 左值表达式 变为 右值表达式
```

#### 移动构造函数 (Move Constructor)

移动构造函数是一类特殊的构造函数:
- `第一个形参`为`指向该类型对象的右值引用`.
- `其余形参`均有`默认实参值`.
- 必须确保析构`移动源 (moved-from) 对象`是安全的.
```cpp
template <typename T>
class Vector {
 public:
  Vector(Vector&& rhs) noexcept  // 不抛出异常
      // 接管 移动源对象 的数据成员:
      : _head(rhs._head), _free(rhs._free), _tail(rhs._tail) {
    rhs._head = rhs._free = rhs._tail = nullptr;  // 确保 析构 rhs 是安全的
  }
  // ...
 private:
  T* _head;  // 指向 首元
  T* _free;  // 指向 第一个自由元
  T* _tail;  // 指向 过尾元
};
```

#### 移动赋值运算符 (Move Assignment Operator)
```cpp
template <typename T>
class Vector {
 public:
  Vector& operator=(Vector&& rhs) noexcept {  // 不抛出异常
    if (this == &rhs) {  // 自己给自己赋值, 不做任何事
      ;
    } else {
      free();  // 析构 被赋值对象 中的元素, 释放内存
      // 接管 移动源对象 的数据成员:
      _head = rhs._head;
      _free = rhs._free;
      _tail = rhs._tail;
      rhs._head = rhs._free = rhs._tail = nullptr;  // 确保 析构 rhs 是安全的
    }
    return *this;  // 返回 左值引用
  }
 private:
  void free();  // 析构元素, 释放内存
  T* _head;  // 指向 首元
  T* _free;  // 指向 第一个自由元
  T* _tail;  // 指向 过尾元
```

#### 异常与容器
容器类型 (例如 `std::vector<T>`) 在`重新分配 (reallocation)` 的过程中, 需要将所存储的元素`逐个搬运`到新分配的内存空间里.
对于类类型 (class type) `T`, 这个搬运动作是利用 `T` 的`拷贝构造函数`来完成的.
如果 `T` 的`移动构造函数`被标注为`不会抛出异常`, 则容器会利用 `T` 的`移动构造函数`来搬运元素.

因此, `移动构造函数`和`移动赋值运算符`应当用不会抛出异常的方法来实现, 并且在 **声明** 和 **定义** 时都用 `noexcept` 进行标注.

## 继承与动态绑定

### 基类与派生类

| 名称 | 同义词 | 任务 |
| ---- | ---- | ---- |
| 基类 (base class) | 父类 (superclass) | 定义自己和派生类`共有的 (common)` 成员 |
| 派生类 (derived class) | 子类 (subclass) | 定义自己`特有的 (specific)` 成员 |

#### 声明与定义
一个类只有被 **定义** 过 (而不仅仅被 **声明** 过), 才能被用作基类
```cpp
class Base;
class Derived : public Base { ... };  // 错误: 用作基类的 Quote 还没有被定义
```

派生列表只出现在派生类的 **定义** 中, 而不出现在派生类的 **声明** 中:
```cpp
class Derived : public Base;  // 错误: 派生列表 只出现在 派生类定义 中
class Derived;                // 正确: 派生列表 不出现在 派生类声明 中
```

#### 构造函数
`派生类构造函数`必须用`基类构造函数`来初始化`派生类对象的基类部分`:
```cpp
class Quote {
  std::string _title;
  double price;
 public:
  Quote(const std::string& title, double price)
      : _title(title), _price(price) { }
  // ...
};
class BulkQuote : public Quote {
  std::size_t _amount;
  double _discount;
 public:
  BulkQuote(const std::string& title, double price, 
            std::size_t amount, double discount)
      : Quote(title, price),  // 用 基类构造函数 来初始化 派生类对象的基类部分
        _amount(amount), _discount(discount) { }
  // ...
};
```

#### 静态数据成员
基类中定义的静态数据成员被继承体系中的所有派生类共享.

#### (C++11) `final` --- 禁止继承
类名后面紧跟 `final` 关键词, 表示该类不可以被用作基类:
```cpp
class NoDerived final { /* ... */ };
```

### 继承级别

#### 成员访问修饰符

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
// 正确: 可以通过 派生类对象 访问 基类的 protected 成员:
void visit(Derived &d) { d._private_i = d._protected_i = 0; }
// 错误: 只能通过 派生类对象 访问 基类的 protected 成员:
clobber(Base &b) { b._protected_i = 0; }
```
在派生类中, 尽管基类的 `public` 和 `protected` 成员可以被直接访问, 还是应该`使用基类的公共接口`来访问基类成员.

#### 派生访问修饰符
派生访问修饰符用于规定`派生类的客户` (包括`下一级派生类`) 对`继承自基类的成员`的访问权限:
- 基类的 `private` 成员无法被派生类访问, 因此不受继承级别的影响.
- 基类的 `public` 和 `protected` 成员可以被派生类访问, 因此受继承级别的影响:
  - `public` 继承:
    - 基类的 `public` 成员 → 派生类的 `public` 成员
    - 基类的 `protected` 成员 → 派生类的 `protected` 成员
  - `protected` 继承: 基类的 `public` 和 `protected` 成员 → 派生类的 `protected` 成员
  - `private` 继承: 基类的 `public` 和 `protected` 成员 → 派生类的 `private` 成员

#### 改变访问级别
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

#### 默认的继承级别

| 关键词  | 默认的继承级别 |
| ------ | ---------------------- |
| `struct` | `public` |
| `class` | `private` |

为提高代码可读性, 应当`显式`写出继承级别.

#### 友元关系不被继承
基类的友元与派生类的友元是相互独立的, 必须为每个类独立声明各自的友元.

### 自动类型转换

#### 静态类型与动态类型
`静态类型`是指一个变量在声明时所使用的类型, 或者一个表达式的计算结果的类型, 在`编译期`就可以确定.

`动态类型`是指一个变量或表达式所代表的内存中的实际对象的类型, 可能要到`运行期`才能够确定.

对于指针或引用, 动态类型`可能`与静态类型`不同`; 
对于其他表达式, 动态类型`总是`与静态类型`相同`.

#### 派生类到基类
一个派生类对象的非静态数据成员可以分为两类: 
- 派生类自己定义的非静态数据成员
- 继承自基类的非静态数据成员

##### 派生类指针或引用 到 基类指针或引用
可以用`指向基类对象`的指针 (包括智能指针) 或引用来`指向派生类对象`.
这种自动类型转换仅适用于指针或引用, 并且其可访问性与基类的 `public` 成员相同.

用容器管理继承体系的对象时, 几乎总是以`基类指针` (包括智能指针) 为元素类型, 然后将`派生类指针`存储到的容器中:
```cpp
std::vector<std::shared_ptr<Base>> vec;
auto pd = std::make_shared<Derived>(/* ... */);
vec.push_back(pd);  // std::shared_ptr<Derived> 自动转换为 std::shared_ptr<Base>
```

##### 派生类对象 到 基类对象
`基类对象`可以用`派生类对象`来`初始化`或`赋值`.
此时, 派生类对象中的`基类部分`将被 copy 或 move, 而`派生类特有的部分`将被忽略.

#### 基类到派生类
##### 不存在自动转换
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
Base* pb = &d;     // 正确: 可以从 Derived* 转换到 Base*
Derived* pd = pb;  // 错误: 无法从 Base* 转换到 Derived*
```

##### 强制类型转换 (慎用)
如果基类定义了虚函数, 则可以用 `std::dynamic_cast` 来执行`基类指针或引用`到`派生类指针或引用`的动态类型转换, 该转换将在`运行期`发生.

一般情况下, 可以用 `std::static_cast` 来执行静态类型转换, 该转换将在`编译期`发生.

### 虚函数与动态绑定
#### 声明与定义
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

##### 纯虚函数与抽象基类
通常, 编译器无法判断一个虚函数是否会被用到, 因此必须为每一个虚函数给出 **定义** , 而不仅仅是 **声明** .
作为例外, 纯虚函数表示`抽象操作`, 其 **定义** 不需要在基类中给出, 而是延迟到派生类中.
 **声明** 纯虚函数只需要在声明的末尾加上 `= 0`.

含有纯虚函数的类称为`抽象基类`, 表示派生类的`抽象接口`.
抽象基类无法创建对象.

#### 动态绑定 (Dynamic Binding)
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

##### 基类的析构函数
为支持动态资源管理, 基类应将`析构函数`声明为虚的, 这样编译器才能够知道应该调用哪个版本的 `delete`.

##### 绕开动态绑定机制
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

#### (C++11) `override` --- 检查函数签名
在派生类中定义一个与基类中的虚函数`同名`但`形参列表不同`的成员函数是合法的.
编译器会将其视作与虚函数无关的一个新的成员.

为了避免在重写虚函数时无意中写错形参列表, 可以用在虚函数的形参列表 (包括 `const`) 后面紧跟关键词 `override`.
如果在派生类中被标注为 `override` 的虚函数与基类版本的虚函数具有不同的形参列表, 编译器将会报错.

#### (C++11) `final` --- 禁止重写
如果要禁止派生类对一个虚函数进行重写, 可以在基类中的虚函数形参列表 (包括 `const`) 后面紧跟关键词 `final`:
```cpp
class Shape {
 public:
  virtual ~Shape() = default;
  virtual void noOverride() const final { }
};
class Rectangle : public Shape {
 public:
  virtual void noOverride() const { }  // 错误: 禁止重写 final 函数
};
```

#### 默认实参 (少用)
如果通过`指向基类`的指针或引用调用虚函数, 将会使用`基类版本`的默认实参.
因此, 派生类版本的虚函数应当使用与基类版本相同的默认实参.
