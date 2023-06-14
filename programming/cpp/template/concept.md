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

# 标准库概念

## `<concepts>`

## Range

[Range](https://en.cppreference.com/w/cpp/ranges/range) 是对容器概念的推广，可以由「起始迭代器 + END」来定义，其中 END 可以是：

- 终止迭代器，如 `{ vec.begin(), vec.end() }`
- 个数，如 `{ vec.begin(), vec.size() }`
- 终止条件，如 `{ vec.begin(), [](int x){ return x % 2; } }`

标准库在命名空间 `std::ranges` 中定义了一些常用的 range `concept`s：

|                       Range `concept`s                       |                             说明                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| [`ranges::range`](https://en.cppreference.com/w/cpp/ranges/range) |                   提供「起始迭代器 + END」                   |
| [`ranges::borrowed_range`](https://en.cppreference.com/w/cpp/ranges/borrowed_range) |                   迭代器可返回（不会空悬）                   |
| [`ranges::sized_range`](https://en.cppreference.com/w/cpp/ranges/sized_range) |                      支持 O(1) `size()`                      |
| [`ranges::view`](https://en.cppreference.com/w/cpp/ranges/view) |                   支持 O(1) `operator=()`                    |
| [`ranges::input_range`](https://en.cppreference.com/w/cpp/ranges/input_range) | 支持 [`input_iterator`](https://en.cppreference.com/w/cpp/iterator/input_iterator) |
| [`ranges::output_range`](https://en.cppreference.com/w/cpp/ranges/output_range) | 支持 [`output_iterator`](https://en.cppreference.com/w/cpp/iterator/output_iterator) |
| [`ranges::forward_range`](https://en.cppreference.com/w/cpp/ranges/forward_range) | 支持 [`forward_iterator`](https://en.cppreference.com/w/cpp/iterator/forward_iterator) |
| [`ranges::bidirectional_range`](https://en.cppreference.com/w/cpp/ranges/bidirectional_range) | 支持 [`bidirectional_iterator`](https://en.cppreference.com/w/cpp/iterator/bidirectional_iterator) |
| [`ranges::random_access_range`](https://en.cppreference.com/w/cpp/ranges/random_access_range) | 支持 [`random_access_iterator`](https://en.cppreference.com/w/cpp/iterator/random_access_iterator) |
| [`ranges::contiguous_range`](https://en.cppreference.com/w/cpp/ranges/contiguous_range) | 支持 [`contiguous_iterator`](https://en.cppreference.com/w/cpp/iterator/contiguous_iterator) |
| [`ranges::common_range`](https://en.cppreference.com/w/cpp/ranges/common_range) |                                                              |
| [`ranges::viewable_range`](https://en.cppreference.com/w/cpp/ranges/viewable_range) | 可以安全地转化为 [`view`](https://en.cppreference.com/w/cpp/ranges/view) |
| [`ranges::constant_range`](https://en.cppreference.com/w/cpp/ranges/constant_range) (C++23) |                           元素只读                           |

标准库在命名空间 `std::ranges` 中还提供了一些对常用算法的封装，使得形如 `std::sort(v.begin(), v.end())` 的调用可简化为 `std::ranges::sort(v)`，从而避免传错迭代器。

## View

[View](https://en.cppreference.com/w/cpp/ranges/view) 是对 range 的轻量化封装（适配器）。

标准库在命名空间 `std::ranges` 中提供了一些常用的 views：

|           `VIEW`            |         `for (auto x : VIEW) { use(x); }` 的传统写法         |
| :-------------------------: | :----------------------------------------------------------: |
|        `all_view{r}`        |                  `for (auto x : r) use(x);`                  |
|     `filter_view{r, p}`     |             `for (auto x : r) if (p(x)) use(x);`             |
|   `transform_view{r, f}`    |                `for (auto x : r) use(f(x));`                 |
|      `take_view{r, n}`      | `int i{0}; for (auto x : r) if (i++ == n) break; else use(x);` |
|      `drop_view{r, n}`      | `int i{0}; for (auto x : r) if (i++ < n) continue; else use(x);` |
|   `take_while_view{r, p}`   |      `for (auto x : r) if (!p(x)) break; else use(x);`       |
|   `drop_while_view{r, p}`   |     `for (auto x : r) if (p(x)) continue; else use(x);`      |
|       `join_view{r}`        |         `for (auto &y : r) for (auto x : y) use(x);`         |
|        `key_view{r}`        |               `for (auto [x, y] : r) use(x);`                |
|       `value_view{r}`       |               `for (auto [y, x] : r) use(x);`                |
|        `ref_view{r}`        |                 `for (auto &x : r) use(x);`                  |
|        以下为生成器         |                                                              |
|       `iota_view{y}`        |           `for (int i = 0: true; ++i) use(y + i);`           |
|      `iota_view{y, z}`      |            `for (auto x = y: x < z; ++x) use(x);`            |
| `istream_view<double>{cin}` |             `double x; while (cin >> x) use(x);`             |

表中 `ranges::X_view{ARGS}` 等价于 `views::X(ARGS)`，即每个 `views::X` 函数生成一个 `ranges::X_view` 对象。

例如按条件过滤：

```cpp
auto filter_odd(ranges::forward_range auto& r) {
  ranges::filter_view v {r, [](int x) { return x % 2; } };  // v 的用户只访问 r 中的奇数
  return v;  // 轻量化封装，直接按值返回
}
int main() {
  auto v = vector<int>{ 3, 5, 1, 2 };
  cout << "odd numbers: ";
  auto fv = filter_odd(v);
  static_assert(ranges::forward_range<decltype(fv)>);  // view 依然是 range
  ranges::for_each(fv, [](int x) { cout << x << ' '; });  // 可以像 range 一样使用 view
  cout << "\n";
}
```

可以创建 view of view，例如：

```cpp
ranges::forward_range/* 类型限制 */ auto
take_2(ranges::view auto/* view 无需传引用 */ fv) {
  ranges::take_view tv {fv, 100};  // 只访问前 2 个奇数
  // 等价于 auto tv = views::take(fv, 100);
  return tv;
}
int main() {
  auto v = vector<int>{ 3, 5, 1, 2 };
  cout << "odd numbers: ";
  auto fv = filter_odd(v);
  auto tv = take_2(fv);  // view of view
  ranges::for_each(tv, [](int x) { cout << x << ' '; });
  cout << "\n";
}
```

## Pipeline
