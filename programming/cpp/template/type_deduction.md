# 模板类型推断

## 形参类型 v. 实参类型

考虑如下 *函数模板* 和*调用语句*：
```cpp
template <typename T> void func(ParaType parameter) { /* ... */ }
ArguType argument;  // argument 的类型为 ArguType
func(argument);     // 根据 ArguType 推断 ParaType
```
其中
- `T` 为留待编译器推断的 ***模板形参 (template parameter)***。
- `parameter` 为 ***函数形参 (function parameter)***，`ParaType` 是它的类型。`ParaType` 既可以是 `T`，也可以是  `T*`、`T&`、`T&&` 等 *复合类型*，还可以是 `std::vector<T>`、`std::set<T>` 等 *容器类型*。
- `argument` 为 ***函数实参 (function argument)***，`ArguType` 是它的类型。`argument` 既可以是 ***右值 (rvalue) 表达式***（例如 `1 + 1`），也可以是 ***左值 (lvalue) 表达式***（例如由变量名构成的表达式，变量本身可以是任何类型）。

编译器通过比较 `ParaType` 与 `ArguType` 来推断 `T`：
> 基本推断规则：
> 1. 忽略 `ArguType` 的 RCV 属性，其中 R、C、V 分别表示：左值或右值引用 (reference)、顶层 `const`、顶层 `volatile`。
> 2. 将上一步所得类型与 `ParaType` 比较，以所需修饰符最少的类型作为 `T`。

为方便讨论，这里先定义一组变量，变量名的含义在注释中用大写字母给出：
```cpp
      int            i =  0;  //                    Int
const int           ci =  i;  //              Const Int
      int &         ri =  i;  //       Ref to       Int
const int &        rci =  i;  //       Ref to Const Int
      int &&       rri =  0;  // Right Ref to       Int
      int *         pi = &i;  //       Ptr to       Int
const int *        pci = &i;  //       Ptr to Const Int
      int * const  cpi = &i;  // Const Ptr to       Int
const int * const cpci = &i;  // Const Ptr to Const Int
```

## `ParaType` 不是指针或引用
这条情况对应于 ***传值 (pass-by-value) 调用*** ：
函数内部所使用的对象是 `argument` 的 *独立副本*，因此 `argument` 的 *顶层 `const` 属性* 及 *顶层 `volatile`* 属性对这个 *独立副本* 没有影响。

### `ParaType = T`
推断过程及结果如下：

| `argument` | `ArguType`    | 忽略 R | 忽略 CV | `T`        |
| ----: | ------------ | ------------- | -------------- | ------------ |
| `0`   | `int`        |              |                  | `int`        |
| `i`  | `int`        |              |                  | `int`        |
| `ci`  | `int const`  |              | `int`        | `int`        |
| `ri`  | `int &`      | `int`      |           | `int`        |
| `rci` | `int const &` | `int const` | `int`        | `int`        |
| `rri` | `int &&`     | `int`       |           | `int`        |
| `pi`  | `int *`      |         |          | `int *`      |
| `pci` | `int const *` |             |                  | `int const *` |
| `cpi` | `int * const` |  | `int *`      | `int *`      |
| `cpci` | `int const * const` | | `int const *` | `int *` |

### `ParaType = const T`
（这里的 `const T` 也可以写成 `T const`）

推断过程及 `T` 的推断结果与上一种情形相同，只是 `ParaType` 多出一个顶层 `const`。

## `ParaType` 为指针
此时 `ArguType` 必须是指针（或对指针的引用）。 

### `ParaType = T *`
此时，底层 `const` 属性（被指对象的 `const` 属性）会被推断为 `T` 的一部分：

| `argument` | `ArguType`    | 忽略 RCV | `T *`         | `T`         |
| ----: | ------------ | --------------- | ------------ | ----------- |
| `pi` | `int *`      |           | `int *`      | `int`       |
| `pci` | `int const *` |     | `int const *` | `int const` |
| `cpi` | `int * const` | `int *`       | `int *`      | `int`       |
| `cpci` | `int const * const` | `int const *` | `int const *` | `int const` |

### `ParaType = T * const`
推断过程及 `T` 的推断结果与 `ParaType = T *` 的情形相同，只是 `ParaType` 多出一个 *顶层 `const` 属性*。

### `ParaType = T const *`
（这里的 `T const *` 也可以写成 `const T *`）

此时，`ArguType` 的 *底层 `const` 属性* 已体现在 `ParaType` 中，因此 `T` 的推断结果不含这个 *底层 `const` 属性*：

| `argument` | `ArguType`    | 忽略 RCV | `T const *`   | `T`   |
| ----: | ------------ | --------------- | -------------- | ----- |
| `pi` | `int *`      |          | `int const *` | `int` |
| `pci` | `int const *` |          |               | `int` |
| `cpi` | `int * const` | `int *`       | `int const *` | `int` |
| `cpci` | `int const * const` | `int const *` | `int const *` | `int` |

### `ParaType = T const * const`
（这里的 `T const * const` 有可以写成 `const T * const`）

推断过程及 `T` 的推断结果与 `ParaType = T const *` 或 `ParaType = const T *` 的的情形相同，只是 `ParaType` 多出一个 *顶层 `const` 属性*。

## `ParaType` 为引用

此时 `ArguType` 可以是任意类型。 

### `ParaType = T &`
此时 `argument` 必须是 *左值表达式*。如果它含有 *顶层 `const` 属性* 或 *底层 `const` 属性*，则会被推断为 `T` 的一部分：

| `argument` | `ArguType`    | `ParaType` | `T`          |
| ----: | ------------ | ------------- | ------------ |
| `i`   | `int`        | `int &`       | `int`        |
| `ci`  | `int const`  | `int const &` | `int const`  |
| `pi`  | `int *`      | `int * &`     | `int *`      |
| `pci` | `int const *` | `int const * &` | `int const *` |
| `cpi` | `int * const` | `int * const &` | `int * const` |
| `cpci` | `int const * const` | `int const * const &` | `int const * const` |

最后两行是 *推断结果含顶层 `const` 属性* 的例子：
- `argument` 是带有 *顶层 `const` 属性* 的指针：这个指针 *本身* 是 `const`。
- `parameter` 是 *对 `const` 指针的引用*，这个 `const` 成为了 *底层 `const` 属性*。
- 因此 `T` 的推断结果中含有 `argument` 的 *顶层 `const` 属性*。

### `ParaType = T const &`
（这里的 `T const &` 也可以写成 `const T &`）

此时 `argument` 可以是任意表达式。如果它含有 *底层 `const` 属性*，则会被推断为 `T` 的一部分，而 *顶层 `const` 属性* 则会被忽略：

| `argument` | `ArguType`    | `T const &`          | `T`          |
| ----- | ------------ | ------------------- | ------------ |
| `0`   | `int`        | `int const &`        | `int`        |
| `i`   | `int`        | `int const &`        | `int`        |
| `ci`  | `int const`  | `int const &`        | `int`      |
| `pi`  | `int *`       | `int * const &`       | `int *`       |
| `pci` | `int const *` | `int const * const &` | `int const *` |
| `cpi` | `int * const` | `int * const &`       | `int *`     |

### `ParaType = T &&`
[Scott Meyers](https://www.aristeia.com/) 在其所著 [***Effective Modern C++***](http://shop.oreilly.com/product/0636920033707.do) 中，将形如 `T &&` 并且 `T` 需要被推断的引用（例如 `ParaType = T &&` 或 `auto &&`）称为 ***万能  (universal) 引用***。

形如 `T &&` 的 *万能引用* 中的待定类型按以下规则推断：

> 1. 如果 `argument` 是 *左值表达式*，则 `T` 为 *左值引用*，否则 `T` 不含引用。
> 2. 如果推断结果出现了多重引用，则按 *引用折叠* 规则处理。

***引用折叠 (reference collapsing)*** 规则：

> 假设 `X` 是不含引用的类型，则
> - `X && &&` 折叠为 `X &&`。
> - 其他情形（`X & &&` 或 `X && &` 或 `X & &`）均折叠为 `X &`。

根据以上规则，`argument` 可以是任意类型：

| `argument` | `ArguType`    | 表达式类型 | `T`           |
| ----: | ------------ | :-----------: | ----- |
| `0`   | `int`        | R  | `int`         |
| `i`   | `int`        | L | `int &`        |
| `ci`  | `int const`  | L | `int const &`  |
| `ri` | `int &` | L | `int &` |
| `rci` | `int const &` | L | `int const &` |
| `rri` | `int &&` | L | `int &` |
| `pi`  | `int *`       | L | `int * &`       |
| `pci` | `int const *` | L | `int const * &` |
| `cpi` | `int * const` | L | `int * const &` |
| `cpci` | `int const * const` | L | `int const * const &` |
| `std::move(i)` | `int &&` | R | `int` |

万能引用几乎总是与 `std::forward<T>()` 配合使用，以达到 ***完美转发 (perfect forward)*** 函数实参的目的。
这里的 *完美* 是指：避免不必要的拷贝或移动，并且保留函数实参的所有类型信息（包括 RCV 属性）。
它的实现需要借助于 [***模板元编程 (template metaprogramming)***](./metaprogramming.md#`std::forward<T>()`-的实现) 技术。
典型应用场景为 *向构造函数完美转发实参*：

```cpp
#include <utility>
#include <vector>
template <class T>
std::vector<T> build(T&& x) {  // T&& 是一个 ***万能引用*** 
  auto v =  std::vector<T>(std::forward<T>(x));
  // decorate v
  return v;
}
```

### 数组或函数
如果 `argument` 是 *数组* 或 *函数*（或对它们的引用），则 `ParaType` 必须含引用，否则 `argument` 会 ***退化 (decay)*** 为指针：
```cpp
template <typename T, typename U>
void f(T, U&) { /* ... */ }
// argument 为数组：
const char book[] = "C++ Primer";  // book 的类型为 const char[11]
f(book, book);  // 推断结果为 void f(const char*, const char(&)[11])
// argument 为函数：
int g(double);
f(g, g);  // 推断结果为 void f(int(*)(double), int(&)(double))
```

# 其他类型推断
## `auto` 类型推断
### 一般情况：与模板类型推断相同
自 C++11 起，可以将 `auto` 用作 *变量类型*。如果使用得当，可以大大简化代码。
一般情况下，[*`auto` 类型推断*](#`auto`-类型推断) 与 [*模板类型推断*](#模板类型推断) 具有相同的规则。
在这些场合，`auto` 实际上就是模板类型形参 `T`，而其他元素有如下对应关系：

| `auto` 语句          | `ParaType`   | `parameter` | `argument` | `ArguType` |
| -------------------- | ------------- | ------- | ----- | --------- |
| `auto i = 0;`        | `auto`        | `i`     | `0`   | `int`     |
| `const auto & j = 1;` | `const auto &` | `j`     | `1` | `int`     |
| `auto&& k = 2;`      | `auto &&`     | `k`     | `2`  | `int`     |

自 C++14 起，还可以将 `auto` 用作 *函数返回类型* 或 *lambda 形参类型*：
```cpp
auto func(int* p) {
  return *p;  // *p 的类型是 int &，因此 auto 被推断为 int
}
auto is_positive = [](const auto& x) { return x > 0; };
is_positive(3.14);  // auto 被推断为 double
is_positive(-256);  // auto 被推断为 int
```

### 唯一例外：列表初始化 ⚠️
对于 `int`，有以下四种 *几乎* 等价的初始化方式，得到的都是 `int` 变量：
```cpp
int a = 1;
int b(2);
int c = {3};
int d{4};
```
但如果换作 `auto`：
```cpp
auto a = 1;
auto b(2);
auto c = {3};
auto d{4};
```
后两种方式得到的是 *只含一个元素的 `std::initializer_list<int>`*。
⚠️ 这是 *唯一* 一处 [*`auto` 类型推断*](#`auto`-类型推断) 不同于 [*模板类型推断*](#模板类型推断) 的地方。

二者的区别在下面的例子中体现得更为明显:

```cpp
#include <initializer_list>  // 不可省略
auto x = {1, 2, 3};  // x 为含有 3 个元素的 std::initializer_list<int>
//  等价的函数模板定义：
template <typename T>
int f(T parameter) { return sizeof(parameter); }
//  正确的函数模板定义：
template <typename T>
int g(std::initializer_list<T> parameter) { return sizeof(parameter); }
// 测试：
int main() {
  f(x);          // T 被推断为 std::initializer_list<int>
  f({1, 2, 3});  // 类型推断失败
  g(x);          // T 被推断为 std::initializer_list<int>
  g({1, 2, 3});  // T 被推断为 int
}
```

##  `decltype` 类型推断
`decltype` 是一种 ***修饰符 (specifier)***，它作用在表达式 `expr` 上得到其类型 `ExprType`：

- 一般情况下，`ExprType` 是 `expr` 的类型（含 RCV 属性）。
- 如果 `expr` 是一个 *左值表达式* 但不是 *变量名*，则 `ExprType` 还需附加一个 *左值引用*。
- 如果 `expr` 是一个 *只含变量名* 的 *左值表达式*，则 `ExprType` 不附加额外的 *左值引用*。

```cpp
#include <type_traits>  // std::is_same
using std::is_same_v;   // C++17

int i = 0;
static_assert(is_same_v<decltype( i ), int >);  // 只含变量名
static_assert(is_same_v<decltype((i)), int&>);  // 左值表达式

int&& rri = 0;
static_assert(is_same_v<decltype( rri ), int&&>);  // 只含变量名
static_assert(is_same_v<decltype((rri)), int& >);  // 左值表达式 + 引用折叠

void f(const int& x) {
  static_assert(is_same_v<decltype(x), const int&>);  // 保留 RCV 属性
}
auto* pf = f;
auto& rf = f;
static_assert(is_same_v<decltype( f), void   (const int&)>);  // 函数
static_assert(is_same_v<decltype(pf), void(*)(const int&)>);  // 函数指针
static_assert(is_same_v<decltype(rf), void(&)(const int&)>);  // 对函数的引用

int a[] = {1, 2, 3};
auto* pa = a;
auto& ra = a;
static_assert(is_same_v<decltype( a),   int   [3]>);  // 数组
static_assert(is_same_v<decltype(ra),   int(&)[3]>);  // 对数组的引用
static_assert(is_same_v<decltype(pa),   int *>);  // 指向数组首元的指针
static_assert(is_same_v<decltype(a[0]), int &>);  // 对数组首元的引用
```

## 函数返回类型推断
考虑如下函数
```cpp
ReturnType func(ParaType parameter) {
  return expr;  // expr 的类型为 ExprType
}
```

### C++11 后置返回类型
如果希望以 `ExprType` 作为 `ReturnType`，则只需要以 `auto` 作为 `ReturnType`，并在 *函数形参列表* 与 *函数体* 之间插入 `-> decltype(expr)`：
```cpp
auto func(ParaType parameter) -> decltype(expr) {
  return expr;  // expr 的类型为 ExprType
}
```
这里的 `auto` 只是一个占位符，实际推断工作是由 `decltype` 来完成的，因此需遵循 [*`decltype` 类型推断*](#`decltype`-类型推断) 规则。
按此规则：如果 `expr` 是一个 *左值表达式* 但不是 *变量名*，则 `decltype` 会为 `ExprType` 附加一个 *左值引用*，即以 `ReturnType = ExprType &`。

如果希望去掉返回值的引用属性（无论是 `ExprType` 本身所含有的，还是 `decltype` 附加上的），则需借助 [`std::remove_reference`](./metaprogramming.md#`std::remove_reference`)：
```cpp
#include <type_traits>  // std::remove_reference
auto func(ParaType parameter)
    -> typename std::remove_reference<decltype(expr)>::type {
  return expr;  // expr 的类型为 ExprType
}
```

### C++14 前置返回类型
C++14 允许以 `auto` 作为返回类型：
```cpp
auto func(ParaType parameter) {
  return expr;  // expr 的类型为 ExprType
}
```
这里的 `auto` 承担了类型推断任务，因此需遵循 [*`auto` 类型推断*](#`auto`-类型推断) 规则。
按此规则：`expr` 的 RCV 属性会丢失。

也可以将 `decltype(auto)` 用作 `ReturnType`，此时需遵循 [*`decltype` 类型推断*](#`decltype`-类型推断) 规则：
```cpp
decltype(auto) func(ParaType parameter) {
  return expr;
}
```
⚠️ 以 `decltype(auto)` 作为 `ReturnType` 需避免返回 *对局部变量的引用*：
```cpp
decltype(auto) func() {
  auto v = std::vector<int>{1, 2, 3};
  return v[0];  // 危险行为：返回对 局部变量 的引用
}
```
