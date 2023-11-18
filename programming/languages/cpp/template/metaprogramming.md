---
title: 元编程
---

**元编程 (metaprogramming)**：以*类*或*函数*等程序**实体 (entity)** 为运算对象的编程方式。

在 C++ 中，元编程主要通过以下语言机制来实现：
- 利用 `template` 生成*类*或*函数*。
- 利用 `constexpr` 完成一些*编译期常量*的计算。

[元编程](#元编程)这种编程**技巧 (technique)**（强调编译期计算）为实现[泛型编程](../generic.md)这种编程**范式 (paradigm)**（强调算法和数据类型的抽象）提供了技术支持。

⚠️ 过度使用元编程会使得代码可读性差、编译时间长、测试难度大。

# 类型函数
类型函数不是[普通函数](../function.md)，而是借助于[类模板](../generic.md)实现的（以*类型*或*编译期常量*为运算对象的）编译期运算机制。

## `std::remove_reference`<a href id="remove_reference"></a>
定义在 `<type_traits>` 中的类模板 `std::remove_reference` 用于**移除 (remove)** 类型实参的**引用 (reference)**：
```cpp
namespace std{
template <class T> struct remove_reference      { typedef T type; };
template <class T> struct remove_reference<T&>  { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };
}  // namespace std
```
使用时，它以*类型实参*为输入，以*类型成员*为输出。
在 C++14 以前，*类模板的类型成员*必须通过 `typename` 来访问，这使得代码变得冗长：
```cpp
#include <type_traits>
int main() {
  typename std::remove_reference<int  >::type x{0};  // 等价于 int x{0};
  typename std::remove_reference<int& >::type y{0};  // 等价于 int y{0};
  typename std::remove_reference<int&&>::type z{0};  // 等价于 int z{0};
}
```
自 C++14 起，标准库为它提供了以 `_t` 为后缀的**别名 (alias)**：
```cpp
namespace std{
template <class T>
using remove_reference_t = typename remove_reference<T>::type;
}  // namespace std
```
这样就可以省略 `::type` 和 `typename`，使代码变得简洁：
```cpp
#include <type_traits>
int main() {
  std::remove_reference_t<int  > x{0};  // 等价于 int x{0};
  std::remove_reference_t<int& > y{0};  // 等价于 int y{0};
  std::remove_reference_t<int&&> z{0};  // 等价于 int z{0};
}
```

## `std::move` 的实现
定义在 `<utility>` 中的函数模板 `std::move` 用于*将实参强制转换为右值引用*。
借助于 [`std::remove_reference`](#remove_reference) 可以给出它的一种实现：

```cpp
#include <type_traits>  // std::remove_reference_t
namespace std{
template <class T>
remove_reference_t<T>&& move(T&& t) {
  // T 可能含有引用属性，先用 remove_reference_t 将其去除，
  // 再用 static_cast 将所得类型强制转换为『右值引用』：
  return static_cast<remove_reference_t<T>&&>(t);
}
}  // namespace std
```

## `std::forward` 的实现
定义在 `<utility>` 中的函数模板 `std::forward` 用于*完美转发实参*，即*保留函数实参的所有类型信息（含引用属性）*。

```cpp
#include <utility>

template<class T>
void foo_wrapper(T&& argu/* always lvalue */) {
  foo(std::forward<T>(argu)); // Forward as lvalue or as rvalue, depending on T
}
```

借助于 [`std::remove_reference`](#remove_reference) 和[引用折叠](./type_deduction.md#collapse)机制可以给出它的一种实现：

```cpp
#include <type_traits>  // std::remove_reference_t
namespace std{
// 如果 T 为 int& ，则 remove_reference_t<T>&  及 T&& 均为 int& ：
template <class T>
T&& forward(remove_reference_t<T>&  t) {
  return static_cast<T&&>(t);
}
// 如果 T 为 int&&，则 remove_reference_t<T>&& 及 T&& 均为 int&&：
template <class T>
T&& forward(remove_reference_t<T>&& t) {
  return static_cast<T&&>(t); 
}
}  // namespace std
```

## 编译期谓词
定义在 [`<type_traits>`](https://en.cppreference.com/w/cpp/header/type_traits) 中的**编译期谓词 (compile-time predicate)** 都是类模板。
它们都含有一个  `static bool value` 成员，可以用于对*类型实参*作*编译期判断*。
自 C++17 起，标准库为它们的 `value` 成员提供了以 `_v` 为后缀的**别名 (alias)**，可以用于简化代码。

例如 `std::is_empty` 用于判断一个*类的对象是否不占存储空间*：
```cpp
namespace std {
// is_empty 的声明：
template <class T> struct is_empty;
// C++17 引入的别名：
template <class T>
inline constexpr bool is_empty_v = is_empty<T>::value;
}  // namespace std
```
用例：
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
// 输出判断结果：
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
运行结果：
```shell
1 1 1 0 0
```

# 选择
## 从两个值中选取一个
*编译期表达式 `c ? v1 : v2`* 根据 `c` 的值（`true` 或 `false`），从 `v1` 与 `v2` 中选取一个，作为该表达式的值。

## 从两个类型中选取一个
定义在 `<type_traits>` 中的类模板 [`std::conditional`](https://en.cppreference.com/w/cpp/types/conditional) 根据*第一个（`bool` 型）模板实参的值*，从*后两个（类型）模板实参*中选取一个：
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
一种可能的实现：
```cpp
namespace std {
// 通用版本，用于 B == true  的情形：
template <bool B, class T, class F>
struct conditional { typedef T type; };  
// 特化版本，用于 B == false 的情形：
template <class T, class F>
struct conditional<false, T, F> { typedef F type; };
// C++14 引入的别名：
template <bool B, class T, class F>
using conditional_t = typename conditional<B, T, F>::type;
}  // namespace std
```

## 从多个类型中选取一个

目前 (C++17)，标准库没有提供从*多个类型*中选取一个的方法。
如果将来有这样的方法（暂且命名为 `std::select`）被补充进标准库中，那么它大致应当支持如下用法：
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
在这里，`std::select` 以*第一个模板实参*为序号，从后面的模板实参列表中选出对应的类型。
它的实现需要用到类模板的[特例化](../generic.md#specialization)和[递归](#递归)以及[变参模板](#变参模板)等机制：

```cpp
namespace std{
// 通用版本，禁止实例化
template<unsigned N, typename... Cases>
struct select;
// 特化版本 (N > 0)：
template <unsigned N, typename T, typename... Cases>
struct select<N, T, Cases...> {
  using type = typename select<N-1, Cases...>::type;
};
// 特化版本 (N == 0)：
template <typename T, typename... Cases>
struct select<0, T, Cases...> {
  using type = T; 
};
// 标准库风格的类型别名:
template<unsigned N, typename... Cases>
using select_t = typename select<N, Cases...>::type;
}  // namespace std
```

## 条件数据成员 (C++20)

```cpp
#include <type_traits>

struct Empty { };

template <bool C>
struct A {
  int *pi;  // 8
  std::conditional_t<C, double, Empty> x;  // C ? 8 : 1
};

template <bool C>
struct B {
  int *pi;  // 8
  [[no_unique_address]] std::conditional_t<C, double, Empty> x;  // C ? 8 : 0
};

int main() {
  static_assert(sizeof(Empty) == 1);
  static_assert(sizeof(A<true>) == 8 + 8);
  static_assert(sizeof(B<true>) == 8 + 8);
  static_assert(sizeof(A<false>) == 8 + 1 + 7/* padding */);
  static_assert(sizeof(B<false>) == 8 + 0);
}
```

See [Conditional Members](https://brevzin.github.io/c++/2021/11/21/conditional-members) and [`no_unique_address`](https://en.cppreference.com/w/cpp/language/attributes/no_unique_address) for details.

# 递归

元编程（编译期计算）中没有**变量 (variable)** 的概念，也没有**循环 (loop)** 机制，因此算法中用到的**迭代 (iteration)** 语义都必须通过**递归 (recursion)** 来实现的。

## 普通函数的递归
```cpp
constexpr int factorial(int i) {
  return i < 2 ? 1 : i * factorial(i-1);
}
int main() {
  static_assert(factorial(0) == 1);
  static_assert(factorial(1) == 1);
  static_assert(factorial(2) == 2);
  static_assert(factorial(3) == 6);
}
```

## 函数模板的递归
```cpp
// 通用版本：
template <int I>
constexpr int factorial() {
  return I * factorial<I-1>();
}
// 特化版本，用作『递归基』：
template <>
constexpr int factorial<0>() {
  return 1;
}
// 测试：
int main() {
  static_assert(factorial<0>() == 1);
  static_assert(factorial<1>() == 1);
  static_assert(factorial<2>() == 2);
  static_assert(factorial<3>() == 6);
}
```

## 类模板的递归
```cpp
// 通用版本：
template <int I>
struct factorial {
  static constexpr int value = I * factorial<I-1>::value;
};
// 特化版本，用作『递归基』：
template <>
struct factorial<0> {
  static constexpr int value = 1;
};
// 仿 C++17 别名：
template <int I>
inline constexpr int factorial_v = factorial<I>::value;
// 测试：
int main() {
  static_assert(factorial<0>::value == 1);
  static_assert(factorial<1>::value == 1);
  static_assert(factorial<2>::value == 2);
  static_assert(factorial<3>::value == 6);
  static_assert(factorial_v<0> == 1);
  static_assert(factorial_v<1> == 1);
  static_assert(factorial_v<2> == 2);
  static_assert(factorial_v<3> == 6);
}
```

# 变参模板

C++11 引入了**变参模板 (variadic template)**，这种模板可以含有**模板形参包 (template parameter pack)**，它是一种可以接受零个或多个**模板实参 (template argument)** 的特殊的**模板形参 (template parameter)**。

## 形参包
```cpp
// Args 是一个『模板形参包』，可以接受零个或多个『模板实参』：
template <typename T, typename... Args>
// rest 是一个『函数形参包』，可以接受零个或多个『函数实参』：
void foo(const T& t, const Args&... rest);
```
在如下调用中
```cpp
int i = 0;
double d = 3.14;
string s = "how now brown cow";
foo(i, s, 42, d);  // 接受 3 个实参
foo(s, 42, "hi");  // 接受 2 个实参
foo(d, s);         // 接受 1 个实参
foo("hi");         // 接受 0 个实参
```
编译器会生成以下 `4` 个版本的实例：
```cpp
void foo(const int&, const string&, const int&, const double&);
void foo(const string&, const int&, const char(&)[3]);
void foo(const double&, const string&);
void foo(const char(&)[3]);
```

## 包的大小
形参包的大小可以由 `sizeof...` 运算符获得：
```cpp
template<typename... Args>
void g(Args... args) {
  cout << sizeof...(Args) << endl;
  cout << sizeof...(args) << endl;
}
```
该表达式是 `constexpr`，因此不会对实参求值。

## 递归的变参模板
*变参函数模板*通常是**递归的 (recursive)**。
作为*递归基*的*非变参*版本必须在*变参*版本之前给出**声明 (declaration)**：

```cpp
#include <iostream>
// 『非变参』版本，用作『递归基』
template<typename T>
std::ostream& print(std::ostream& os, const T& t) {
  return os << t;
}
// 『变参』版本，置于『非变参』版本之后：
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
位于*形参包*右侧的 `...` 表示对这个*包*按相应的**模式 (pattern)** 作展开。在上面的*变参*版本中：
- `const Args&...` 对类型形参包 `Args` 按模式 `const Args&` 作展开。
- `rest...` 对函数形参包 `rest` 按模式 `rest` 作展开。

需要展开的形参包可以具有更加复杂的模式：
```cpp
// 对 print 的每一个实参调用 Debug
template <typename... Args>
std::ostream& printDebug(std::ostream& os, const Args&... rest) {
  // 相当于 print(os, Debug(a1), ..., Debug(an))
  return print(os, Debug(rest)...);
}
```

## 包的转发
上面定义的 `print()` 在末尾不执行换行。
若要添加一个末尾换行的版本，可以基于 `print()` 定义一个 `println()`：

```cpp
#include <iostream>
#include <utility>
// 『非变参』版本，用作『递归基』
template<typename T>
std::ostream& print(std::ostream& os, const T& t) {
  return os << t;
}
// 『变参』版本
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
在这里，`std::forward<Args>(args)...` 中的*模板实参包 `Args`* 和*函数实参包 `args`* 将同时被展开，相当于：
```cpp
std::forward<T1>(t1), ..., std::forward<Tn>(tn)
```
