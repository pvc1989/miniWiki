# 概述
以程序`实体` [entity] (例如 C++ 中的`类`或`函数`) 为运算对象的编程方式称为`元编程` [metaprogramming].

在 C++ 中, 元编程主要通过以下语言机制来实现:
- `template` --- 在编译期生成`类型`或`函数`.
- `constexpr` --- 在编译期完成一些`值`的计算.

`元编程`这种编程`技巧` [technique] (侧重于类型和值的计算) 是实现`泛型编程`这种编程`范式` [paradigm] (侧重于算法的抽象声明) 的技术基础.

⚠️过度使用元编程会使得代码可读性变差, 编译时间变长, 测试和维护难度加大.

# 类型函数
在 C++ 中, 类型函数不是普通[函数](./function.md), 而是借助于[模板类](./generic.md)实现的编译期类型运算机制.

## 移除或添加引用
### `std::remove_reference`
定义在 `<type_traits>` 中的模板类 `std::remove_reference` 用于 remove 类型实参的 reference:
```cpp
namespace std{
template <class T> struct remove_reference      { typedef T type; };
template <class T> struct remove_reference<T&>  { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };
}
```
使用时, 它以类型实参为输入, 以类型成员为输出.
受语法所限, 模板类的类型成员必须通过 `typename` 来访问, 这使得代码变得冗长:
```cpp
#include <type_traits>

int main() {
  typename std::remove_reference<int  >::type x{0};  // 等价于 int x{0};
  typename std::remove_reference<int& >::type y{0};  // 等价于 int y{0};
  typename std::remove_reference<int&&>::type z{0};  // 等价于 int z{0};
}
```
#### 别名 --- 省略 `::type` 和 `typename`
C++14 为它提供了以 `_t` 为后缀的别名:
```cpp
namespace std{
template <class T>
using remove_reference_t = typename remove_reference<T>::type;
}
```
这样在使用时就可以省略 `::type` 和 `typename`, 从而使代码变得简洁:
```cpp
#include <type_traits>

int main() {
  std::remove_reference_t<int  > x{0};  // 等价于 int x{0};
  std::remove_reference_t<int& > y{0};  // 等价于 int y{0};
  std::remove_reference_t<int&&> z{0};  // 等价于 int z{0};
}
```

#### `std::move` 的实现
定义在 `<utility>` 中的函数 `std::move()` 用于将实参强制转换为右值引用.
一种可能的实现方式为:
```cpp
#include <type_traits>  // std::remove_reference_t

namespace std{
template <class T>
remove_reference_t<T>&& move(T&& t) {
  // T 可能被推断为引用, 先用 remove_reference_t 将其去除
  // 再用 static_cast 将所得类型强制转换为 右值引用
  return static_cast<remove_reference_t<T>&&>(t);
}
}
```

#### `std::forward` 的实现
定义在 `<utility>` 中的模板函数 `std::forward<T>()` 用于完美转发实参 --- 保留所有类型 (含引用) 信息.
一种可能的实现方式为:
```cpp
#include <type_traits>  // std::remove_reference_t

namespace std{
template <class T>
T&& forward(remove_reference_t<T>&  t) { return static_cast<T&&>(t); }
template <class T>
T&& forward(remove_reference_t<T>&& t) { return static_cast<T&&>(t); }
}
```

## 谓词 --- `std::is_*`
谓词 [predicate] 是指返回值类型为 `bool` 的函数.
在 C++ 模板元编程中, 它是通过含有 `bool` 型成员的模板类来实现的.

### `std::is_empty`
定义在`<type_traits>` 中的模板类 `std::is_empty` 用于判断一个类的对象是否不占存储空间:
```cpp
namespace std {
template <class T> struct is_empty;
}
```
它含有一个 `bool` 型静态数据成员 `value`, 可以用作`返回值`.
#### 别名 --- 省略 `::value`
C++17 为它提供了以 `_v` 为后缀的别名:
```cpp
namespace std {
template <class T>
inline constexpr bool is_empty_v = is_empty<T>::value;
}
```
这样在使用时就可以省略 `::value`, 从而使代码变得简洁:
```cpp
#include <iostream>
#include <type_traits>

// 对象不占存储空间的类:
struct HasNothing { };
struct HasStaticDataMember { static int m; };
struct HasNonVirtualMethod { void pass(); };

// 对象占据存储空间的类:
struct HasNonStaticDataMember { int m; };
struct HasVirtualMethod { virtual void pass(); };

template <class T>
void print() {
  std::cout << (std::is_empty_v<T> ? true : false) << ' ';
}

int main() {
  print<HasNothing>();
  print<HasStaticDataMember>();
  print<HasNonVirtualMethod>();
  print<HasNonStaticDataMember>();
  print<HasVirtualMethod>();
}
```
运行结果:
```cpp
1 1 1 0 0
```

# 控制结构

## 选择
### `?:` --- 从两个值中选取一个
选择运算符根据`条件`在两个`值` [value] 之间进行选择.

### `std::conditional` --- 从两个类型中选取一个
定义在`<type_traits>` 中的模板类 `std::conditional` 根据`条件`在两个`类型` [type] 之间进行选择:
```cpp
#include <iostream>
#include <type_traits>
 
int main() {
  using T = std::conditional<true, int, double>::type;  // C++11
  using F = std::conditional_t<false, int, double>;     // C++14
  static_assert(std::is_same_v<T, int>);     // C++17
  static_assert(std::is_same_v<F, double>);  // C++17
}
```
一种可能的实现:
```cpp
namespace std {
// 通用版本, 用于 B == true 的情形:
template <bool B, class T, class F>
struct conditional { typedef T type; };  

// 特利版本, 用于 B == false 的情形:
template <class T, class F>
struct conditional<false, T, F> { typedef F type; };

// C++14 引入的别名:
template <bool B, class T, class F>
using conditional_t = typename conditional<B, T, F>::type;
}
```

### 从多个类型中选取一个
目前, 标准库没有提供类似于 `std::conditional` 的从`多个`类型中选取一个的方法.
如果将来有这样的方法被补充进标准库中, 那么大致应当有如下用例:
```cpp
#include <iostream>
#include <type_traits>
 
int main() {
  using T2 = std::select<2, int, long, float, double>::type;  // 仿 C++11
  using T3 = std::select_t<3, int, long, float, double>;      // 仿 C++14
  static_assert(std::is_same_v<T2, float>);   // C++17
  static_assert(std::is_same_v<T3, double>);  // C++17
}
```
`std::select` 的实现需要借助于模板特化和递归:
```cpp
namespace std{
// 通用版本, 禁止实例化
template<unsigned N, typename... Cases>
struct select;

// 特化版本 (N > 0):
template <unsigned N, typename T, typename... Cases>
struct select<N, T, Cases...> {
  using type = typename select<N-1, Cases...>::type;
};

// 特化版本 (N == 0):
template <typename T, typename... Cases>
struct select<0, T, Cases...> {
  using type = T; 
};

// 标准库风格的类型别名:
template<unsigned N, typename... Cases>
using select_t = typename select<N, Cases...>::type;
}
```


# (C++11) 可变参数模板 (Variadic Templates)
模板形参数量可变的模板函数或模板类称为`可变参数模板`.

## 形参包 (Parameter Pack)
数量可变的一组 (`模板`或`函数`) 形参称为`形参包`.
```cpp
// Args 是一个 模板形参包, 可含有 零或多个 模板形参:
template <typename T, typename... Args>
// rest 是一个 函数形参包, 可含有 零或多个 函数形参:
void foo(const T& t, const Args&... rest);
```
在如下调用中
```cpp
int i = 0;
double d = 3.14;
string s = "how now brown cow";
foo(i, s, 42, d);  // 形参包中含有 3 个形参
foo(s, 42, "hi");  // 形参包中含有 2 个形参
foo(d, s);         // 形参包中含有 1 个形参
foo("hi");         // 形参包中含有 0 个形参
编译器会实例化出以下 4 个版本的 `foo`:
```cpp
void foo(const int&, const string&, const int&, const double&);
void foo(const string&, const int&, const char(&)[3]);
void foo(const double&, const string&);
void foo(const char(&)[3]);
```

### `sizeof...` 运算符
形参包中的形参个数可以由 `sizeof...` 运算符获得:
```cpp
template<typename... Args>
void g(Args... args) {
  cout << sizeof...(Args) << endl;
  cout << sizeof...(args) << endl;
}
```
该表达式是一个 `constexpr`, 因此不会对实参求值.

## 递归的模板函数
可变参数模板函数通常是`递归的`. 作为`递归终止条件`的 (模板) 函数必须在`可变参数`的版本之前声明:
```cpp
#include <iostream>

// 作为递归终止条件的版本
template<typename T>
std::ostream& print(std::ostream& os, const T& t) {
  return os << t;
}
// 可变参数的版本
template <typename T, typename... Args>
std::ostream& print(std::ostream& os, const T& t, const Args&... rest) {
  os << t << ", ";
  return print(os, rest...);
}

int main() {
  print(std::cout, "hello", "world");
  return 0;
}
```

## 包的展开
位于包右侧的 `...` 表示对其按相应的`模式 (pattern)` 进行展开:
```cpp
template <typename T, typename... Args>
std::ostream& print(std::ostream& os, const T& t, const Args&... rest) {
  os << t << ", ";
  return print(os, rest...);
}
```
在这里, `const Args&...` 对函数形参包 `Args` 按模式 `const Args&` 进行展开, `rest...` 对函数实参包 `rest` 按模式 `rest` 进行展开.

需要展开的包可以具有更加复杂的模式:
```cpp
// 对 print 的每一个实参调用 debug_rep
template <typename... Args>
std::ostream& errorMsg(std::ostream& os, const Args&... rest) {
  // 相当于 print(os, debug_rep(a1), ..., debug_rep(an))
  return print(os, debug_rep(rest)...);
}
```

## 包的转发
之前定义过的 `print` 在末尾不换行.
若要添加一个末尾换行的版本, 可以基于 `print` 定义一个 `println`:
```cpp
#include <iostream>
#include <utility>

// 作为递归终止条件的版本
template<typename T>
std::ostream& print(std::ostream& os, const T& t) {
  return os << t;
}
// 可变参数的版本
template <typename T, typename... Args>
std::ostream& print(std::ostream& os, const T& t, const Args&... rest) {
  os << t << ", ";
  return print(os, rest...);
}
// 末尾换行的版本
template <typename... Args>
void println(std::ostream& os, Args&&... args) {
  print(os, std::forward<Args>(args)...);
  print(os, '\n');
}

int main() {
  println(std::cout, "hello", "world");
  return 0;
}
```
在这里, `std::forward<Args>(args)...` 中的`模板实参包 Args` 和`函数实参包 args` 将被同时展开:
```cpp
std::forward<T1>(t1), ..., std::forward<Tn>(tn)
```
