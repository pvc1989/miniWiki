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
