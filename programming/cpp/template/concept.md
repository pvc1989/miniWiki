---
title: 概念
---

# 类型限制

## 使用 `concept`

类型未加限制的版本，类型错误到『实例化时 (instantiation time)』才会发现：

```cpp
template<typename Seq, typename Val>
Val sum(Seq s, Val v) {
  for (const auto& x : s)
    v += x;
  return v;
}
```

假设已定义 `Sequence` 及 `Value` 两个 `concept`s，可定义类型受限的版本：

```cpp
template<Sequence Seq, Value Val>
Val sum(Seq s, Val v) {
  for (const auto& x : s)
    v += x;
  return v;
}
```

## `requires` 分句

`template <Concept Type>` 等价于 `template<typename Type> requires Concept<Type>`，其中 `requires` 开启「需求分句 (`requires` clause)」，其后的 `Concept<Type>` 为编译期谓词（`Type` 满足 `Concept` 则为 `true`）。

```cpp
template<typename Seq, typename Val>
	requires Sequence<Seq> && Value<Val>
Val sum(Seq s, Val v) {
  for (const auto& x : s)
    v += x;
  return v;
}
```

进一步要求 `Seq` 的元素的类型可以与 `Val` 类型进行算术运算：

```cpp
template<Sequence Seq, Value Val>
  requires Arithmetic<range_value_t<Seq>, Val>
Val sum(Seq s, Val n);
# 或更简洁的
template<Sequence Seq, Arithmetic<range_value_t<Seq>> Val>
Val sum(Seq s, Val n);
```

## `requires` 表达式

```cpp
template<forward_iterator Iter>
  requires requires(Iter p, int i) { p[i]; p+i; }  // 额外的需求
void advance(Iter p, int n) {
  p += n;  // ⚠️ 满足上述需求的 Iter，未必支持 +=
}
```

其中第二个 `requires` 开启「需求表达式 (`requires` expression)」。后者为编译期谓词（`{}` 中的代码合法则其值为 `true`），相当于泛型编程的汇编代码，只因出现在底层代码（如 `concept`s 的定义）中。

## 定义 `concept`

定义概念：

```cpp
template<typename B>
concept Boolean = requires(B x, B y) {
  { x = true };
  { x = false };
  { x = (x == y) };
  { x = (x != y) };
  { x = !x };
  { x = (x = y) };
};

template<typename T>
concept EqualityComparable = requires (T a, T b) {
  { a == b } -> Boolean;  // -> 之后必须跟某个 concept
  { a != b } -> Boolean;
};
```

显式判断：

```cpp
static_assert(EqualityComparable<int>);  // 通过编译

struct S { int a; };
static_assert(EqualityComparable<S>);  // 编译期报错
```

隐式判断：

```cpp
template<EqualityComparable T>
bool cmp(T a, T b) {
  return a < b;  // ⚠️ 未在 EqualityComparable 中检查
}

bool b0 = cmp(cout, cerr);  // 未通过 EqualityComparable 检查
bool b1 = cmp(2, 3);        // OK
bool b2 = cmp(2+3i, 3+4i);  // 通过 EqualityComparable 检查，但在实例化时报错
```

补全开头的例子：

```cpp
#include <ranges>
#include <iterator>
template<typename S>
concept Sequence = requires (S a) {
  typename range_value_t<S>;  // S 必须有 value type
  typename iterator_t<S>;     // S 必须有 iterator type

  requires input_iterator<iterator_t<S>>;  // S 的 iterator 必须可读
  requires same_as<range_value_t<S>, iter_value_t<S>>;  // 类型必须一致

  { a.begin() } -> same_as<iterator_t<S>>;  // S 必须有返回 iterator 的 begin()
  { a.end() } -> same_as<iterator_t<S>>;

};
template<typename T, typename U = T>
concept Numeric = requires(T x, U y) {
  x+y; x-y; x*y; x/y; x+=y; x-=y; x*=y; x/=y; x=x; x=0;
};
template<typename T, typename U = T>
concept Arithmetic = Numeric<T, U> && Numeric<U, T>;
```

💡 建议用形容词命名概念。

## 限制 `auto`

限制函数形参：

```cpp
auto twice(Arithmetic auto x) { return x+x; } // 只支持算术类型
auto thrice(auto x) { return x+x+x; }         // 支持任意可 + 类型

auto x1 = twice(7);   // x1 == 14
auto s = string("Hello ");
auto x2 = twice(s);   // string 不满足 Arithmetic
auto x3 = thrice(s);  // x3 == "Hello Hello Hello "
```

限制变量类型：

```cpp
Channel open_channel(string);

auto ch1 = open_channel("foo");              // ch1 为 Channel 变量
Arithmetic auto ch2 = open_channel("foo");   // Channel 不满足 Arithmetic
ChannelLike auto ch3 = open_channel("foo");  // Channel 满足 ChannelLike
```

限制返回类型：

```cpp
Numeric auto some_function(int x) {
    // ...
    return fct(x);    // an error unless fct(x) returns a Numeric
}
```

