# 拷贝控制

## 析构
### 析构函数
**析构函数 (destructor)** 是一种特殊的成员函数，它以 **类名** 加前缀 `~` 为函数名，形参列表为空，没有返回类型，用于 **析构 (destroy)** 对象。

```cpp
class Foo {
 public:
  ~Foo();  // destructor
};
```

### 成员析构的顺序
一个对象被析构时，先执行其所属类型的析构函数的 **函数体** 中的语句，再隐式地 **逐个析构** 其（非静态）数据成员。
数据成员被析构的顺序与它们被构造的顺序相反，即：与它们在类的定义中出现的顺序相反。

### 合成的析构函数
如果析构函数没有被显式地声明，那么编译器会隐式地定义一个默认的版本，称为 **合成的 (synthesized)** 析构函数。
C++11 允许显式地生成合成的析构函数，只需要在定义时在形参列表后紧跟 `= default;` 即可。

合成的析构函数，只会逐个析构数据成员，这意味着不会对 **内置指针 (built-in pointer)** 成员调用 `delete` 运算符。

## 拷贝 (Copy)
### 拷贝构造函数 (Copy Constructor)
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

### 拷贝赋值运算符 (Copy Assignment Operator)
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

### (C++11) 删除的拷贝操作
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

### 合成的拷贝操作
合成的拷贝操作 (`拷贝构造函数`和`拷贝赋值运算符`) 会`逐个拷贝`数据成员 --- 这意味着: 只会对`裸指针`进行`浅拷贝`.

如果含有数组成员, 则合成的拷贝操作会`逐个拷贝成员数组的每一个元素`.

如果一个类含有`无法拷贝的`数据成员, 则这个类本身也应当是`无法拷贝的`, 因此合成的拷贝操作将是`删除的`.

## 移动 (Move)

### 右值
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

### 移动构造函数 (Move Constructor)

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

### 移动赋值运算符 (Move Assignment Operator)
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

### 异常与容器
容器类型 (例如 `std::vector<T>`) 在`重新分配 (reallocation)` 的过程中, 需要将所存储的元素`逐个搬运`到新分配的内存空间里.
对于类类型 (class type) `T`, 这个搬运动作是利用 `T` 的`拷贝构造函数`来完成的.
如果 `T` 的`移动构造函数`被标注为`不会抛出异常`, 则容器会利用 `T` 的`移动构造函数`来搬运元素.

因此, `移动构造函数`和`移动赋值运算符`应当用不会抛出异常的方法来实现, 并且在 **声明** 和 **定义** 时都用 `noexcept` 进行标注.

