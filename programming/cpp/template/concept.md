---
title: 概念
---

# 类型限制

## 使用 `concept`

未加类型限制的模板，类型错误到『实例化时 (instantiation time)』才会发现：

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

标准库在 `<concepts>` 中定义了一些常用概念。

### 核心语言概念


- [`same_as<T, U>`](https://en.cppreference.com/w/cpp/concepts/same_as) means `T` is the same as `U`.
- [`derived_from<T, U>`](https://en.cppreference.com/w/cpp/concepts/derived_from) means `T` is derived from `U`.
- [`convertible_to<T, U>`](https://en.cppreference.com/w/cpp/concepts/convertible_to) means `T` is convertible to `U`.
- [`common_reference_with<T, U>`](https://en.cppreference.com/w/cpp/concepts/common_reference_with) means `T` shares a common reference type with `U`.
- [`common_with<T, U>`](https://en.cppreference.com/w/cpp/concepts/common_with) means `T` shares a common type ([`common_type_t<T, U>`](https://en.cppreference.com/w/cpp/types/common_type)) with `U`.
- [`integral<T>`](https://en.cppreference.com/w/cpp/concepts/integral) means `T` is a type of integers.
- [`signed_integral<T>`](https://en.cppreference.com/w/cpp/concepts/signed_integral) means `T` is a type of signed integers.
- [`unsigned_integral<T>`](https://en.cppreference.com/w/cpp/concepts/unsigned_integral) means `T` is a type of unsigned integers.
- [`floating_point<T>`](https://en.cppreference.com/w/cpp/concepts/floating_point) means `T` is a type of floating point numbers.
- [`assignable_from<T, U>`](https://en.cppreference.com/w/cpp/concepts/assignable_from) means `T` is assignable from `U`.
- [`swappable_with<T, U>`](https://en.cppreference.com/w/cpp/concepts/swappable) means `T` is swappable with `U`.

  - [`swappable<T>`](https://en.cppreference.com/w/cpp/concepts/swappable) is short for `swappable_with<T, T>`.


### 比较概念


- *[`boolean-testable`](https://en.cppreference.com/w/cpp/concepts/boolean-testable)* (exposition-only) means `T` can be used in Boolean contexts.
- [`equality_comparable_with<T, U>`](https://en.cppreference.com/w/cpp/concepts/equality_comparable) means `T` is equality comparable with `U`.
  - [`equality_comparable<T>`](https://en.cppreference.com/w/cpp/concepts/equality_comparable) is short for `equality_comparable_with<T, T>`.
- [`three_way_comparable_with<T, U>`](https://en.cppreference.com/w/cpp/utility/compare/three_way_comparable) means
  - [`three_way_comparable<T>`](https://en.cppreference.com/w/cpp/utility/compare/three_way_comparable) is short for `three_way_comparable_with<T, T>`.
- [`totally_ordered_with<T, U>`](https://en.cppreference.com/w/cpp/concepts/totally_ordered) ([`<compare>`](https://en.cppreference.com/w/cpp/header/compare)) means
  - [`totally_ordered<T>`](https://en.cppreference.com/w/cpp/concepts/totally_ordered) is short for `totally_ordered_with<T, T>`.

### 对象概念

- [`destructible<T>`](https://en.cppreference.com/w/cpp/concepts/destructible) means `T` is destructible.
- [`constructible_from<T, Args>`](https://en.cppreference.com/w/cpp/concepts/constructible_from) means `T` can be constructed from an argument list of type `Args`.
- [`default_initializable<T>`](https://en.cppreference.com/w/cpp/concepts/default_initializable) means `T` can be default constructed.
- [`move_constructible<T>`](https://en.cppreference.com/w/cpp/concepts/move_constructible) means `T` can be move constructed.
  - [`movable<T>`](https://en.cppreference.com/w/cpp/concepts/movable) means `move_constructible<T> && assignable_from<T&, T> && swappable<T>`.

- [`copy_constructible<T>`](https://en.cppreference.com/w/cpp/concepts/copy_constructible) means `T` can be copy constructed.
  - [`copyable<T>`](https://en.cppreference.com/w/cpp/concepts/movable) means `copy_constructible<T> && assignable_from<T, const T&> && movable<T>`.

- [`semiregular<T>`](https://en.cppreference.com/w/cpp/concepts/semiregular) means `copyable<T> && default_initializable<T>`.
  - [`regular<T>`](https://en.cppreference.com/w/cpp/concepts/regular) means `semiregular<T> && equality_comparable<T>`.


### 可调用概念

- [`invocable<F, Args>`](https://en.cppreference.com/w/cpp/concepts/invocable) means an `F` can be invoked with an argument list of type `Args`. 
  - [`regular_invocable<F, Args>`](https://en.cppreference.com/w/cpp/concepts/invocable) means an `invocable<F, Args>` that is equality preserving (i.e. $\forall (x = y)\implies f(x)=f(y)$, which (currently) cannot be represented in code).
    - [`predicate<F, Args>`](https://en.cppreference.com/w/cpp/concepts/predicate) means an `regular_invocable<F, Args>` that returns a `bool`.
      - [`relation<F, T, U>`](https://en.cppreference.com/w/cpp/concepts/relation) means `predicate<F, T, U>`.
        - [`equivalence_relation<F, T, U>`](https://en.cppreference.com/w/cpp/concepts/equivalence_relation) means an `relation<F, T, U>` that provides an equivalence relation (, which (currently) cannot be represented in code).
        - [`strict_weak_order<F, T, U>`](https://en.cppreference.com/w/cpp/concepts/strict_weak_order) means an `relation<F, T, U>` that provides strict weak ordering (, which (currently) cannot be represented in code).


## `<iterator>`

### 迭代器概念

- [`indirectly_readable`](https://en.cppreference.com/w/cpp/iterator/indirectly_readable) specifies that a type is indirectly readable by applying `operator *`.
- [`indirectly_writable`](https://en.cppreference.com/w/cpp/iterator/indirectly_writable) specifies that a value can be written to an iterator's referenced object.
- [`weakly_incrementable`](https://en.cppreference.com/w/cpp/iterator/weakly_incrementable) specifies that a [`semiregular`](https://en.cppreference.com/w/cpp/concepts/semiregular) type can be incremented with pre- and post-increment operators.
  - [`incrementable`](https://en.cppreference.com/w/cpp/iterator/incrementable) specifies that the increment operation on a [`weakly_incrementable`](https://en.cppreference.com/w/cpp/iterator/weakly_incrementable) type is [equality-preserving](https://en.cppreference.com/w/cpp/concepts#Equality_preservation) and that the type is [`equality_comparable`](https://en.cppreference.com/w/cpp/concepts/equality_comparable).

- [`input_or_output_iterator`](https://en.cppreference.com/w/cpp/iterator/input_or_output_iterator) specifies that objects of a type can be incremented and dereferenced.
  - [`sentinel_for`](https://en.cppreference.com/w/cpp/iterator/sentinel_for) specifies a type is a sentinel for an [`input_or_output_iterator`](https://en.cppreference.com/w/cpp/iterator/input_or_output_iterator) type.
  - [`sized_sentinel_for`](https://en.cppreference.com/w/cpp/iterator/sized_sentinel_for) specifies that the `-` operator can be applied to an iterator and a sentinel to calculate their difference in constant time.

- [`input_iterator`](https://en.cppreference.com/w/cpp/iterator/input_iterator) specifies that a type is an input iterator, that is, its referenced values can be read and it can be both pre- and  post-incremented.
  - [`forward_iterator`](https://en.cppreference.com/w/cpp/iterator/forward_iterator) specifies that an [`input_iterator`](https://en.cppreference.com/w/cpp/iterator/input_iterator) is a forward iterator, supporting equality comparison and multi-pass.
    - [`bidirectional_iterator`](https://en.cppreference.com/w/cpp/iterator/bidirectional_iterator) specifies that a [`forward_iterator`](https://en.cppreference.com/w/cpp/iterator/forward_iterator) is a bidirectional iterator, supporting movement backwards.
      - [`random_access_iterator`](https://en.cppreference.com/w/cpp/iterator/random_access_iterator) specifies that a [`bidirectional_iterator`](https://en.cppreference.com/w/cpp/iterator/bidirectional_iterator) is a random-access iterator, supporting advancement in constant time and subscripting.
        - [`contiguous_iterator`](https://en.cppreference.com/w/cpp/iterator/contiguous_iterator) specifies that a [`random_access_iterator`](https://en.cppreference.com/w/cpp/iterator/random_access_iterator) is a contiguous iterator, referring to elements that are contiguous in memory.

- [`output_iterator`](https://en.cppreference.com/w/cpp/iterator/output_iterator) specifies that a type is an output iterator for a given value  type, that is, values of that type can be written to it and it can be  both pre- and post-incremented.

### 算法概念

为简化算法对类型的限制，标准库定义了一组概念：

- [`indirectly_movable<In, Out>`](https://en.cppreference.com/w/cpp/iterator/indirectly_movable) specifies that values may be moved from an [`indirectly_readable`](https://en.cppreference.com/w/cpp/iterator/indirectly_readable) type `In` to an [`indirectly_writable`](https://en.cppreference.com/w/cpp/iterator/indirectly_writable) type `Out`.
- [`indirectly_movable_storable<In, Out>`](https://en.cppreference.com/w/cpp/iterator/indirectly_movable_storable) specifies that [`indirectly_movable<In, Out>`](https://en.cppreference.com/w/cpp/iterator/indirectly_movable) and that the move may be performed via an intermediate object.
- [`indirectly_copyable<In, Out>`](https://en.cppreference.com/w/cpp/iterator/indirectly_copyable) specifies that values may be copied from an [`indirectly_readable`](https://en.cppreference.com/w/cpp/iterator/indirectly_readable) type `In` to an [`indirectly_writable`](https://en.cppreference.com/w/cpp/iterator/indirectly_writable) type `Out`.
- [`indirectly_copyable_storable<In, Out>`](https://en.cppreference.com/w/cpp/iterator/indirectly_copyable_storable) specifies that [`indirectly_copyable<In, Out>`](https://en.cppreference.com/w/cpp/iterator/indirectly_copyable) and that the copy may be performed via an intermediate object.
- [`indirectly_swappable<I1, I2=I1>`](https://en.cppreference.com/w/cpp/iterator/indirectly_swappable) specifies that the values referenced by two [`indirectly_readable`](https://en.cppreference.com/w/cpp/iterator/indirectly_readable) types can be swapped.
- [`indirectly_comparable<I1, I2, Comp>`](https://en.cppreference.com/w/cpp/iterator/indirectly_comparable) specifies that the values referenced by two [`indirectly_readable`](https://en.cppreference.com/w/cpp/iterator/indirectly_readable) types can be compared.
- [`permutable<I>`](https://en.cppreference.com/w/cpp/iterator/permutable) specifies the common requirements of algorithms that reorder elements in place, i.e. [`forward_iterator<I>`](https://en.cppreference.com/w/cpp/iterator/forward_iterator) `&&` [`indirectly_movable_storable<I, I>`](https://en.cppreference.com/w/cpp/iterator/indirectly_movable_storable) `&&` [`indirectly_swappable<I, I>`](https://en.cppreference.com/w/cpp/iterator/indirectly_swappable).
  - [`sortable<In, Comp = ranges::less, Proj = std::identity>`](https://en.cppreference.com/w/cpp/iterator/sortable) specifies the common requirements of algorithms that permute sequences into ordered sequences. i.e. [`permutable<I>`](https://en.cppreference.com/w/cpp/iterator/permutable) `&&` [`indirect_strict_weak_order<Comp, std::projected<I, Proj>>`](https://en.cppreference.com/w/cpp/iterator/indirect_strict_weak_order).
- [`mergeable<I1, I2, Out, Comp = ranges::less, Proj1 = std::identity, Proj2 = std::identity>`](https://en.cppreference.com/w/cpp/iterator/mergeable) specifies the requirements of algorithms that merge sorted sequences into an output sequence by copying elements, i.e. [`input_iterator<I1>`](https://en.cppreference.com/w/cpp/iterator/input_iterator) `&&` [`input_iterator<I2>`](https://en.cppreference.com/w/cpp/iterator/input_iterator) `&&` [`weakly_incrementable<Out>`](https://en.cppreference.com/w/cpp/iterator/weakly_incrementable) `&&` [`indirectly_copyable<I1, Out>`](https://en.cppreference.com/w/cpp/iterator/indirectly_copyable) `&&` [`indirectly_copyable<I2, Out>`](https://en.cppreference.com/w/cpp/iterator/indirectly_copyable) `&&` [`indirect_strict_weak_order<Comp, std::projected<I1, Proj1>, std::projected<I2, Proj2>>`](https://en.cppreference.com/w/cpp/iterator/indirect_strict_weak_order).

其中

- `std::projected` (defined in `<iterator>`) 为封装了迭代器 `<In>` 及可调用对象 `Proj` 的类型：
  
  ```cpp
  template< std::indirectly_readable In,
            std::indirectly_regular_unary_invocable<In> Proj >
  struct projected {
    using value_type = std::remove_cvref_t<std::indirect_result_t<Proj&, In>>;
    std::indirect_result_t<Proj&, In> operator*() const; // not defined
  };
  ```
- `std::identity` (defined in `<functional>`) 为函数对象类型，其 `operator()` 原样返回实参。

## `<ranges>`

### Range

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

### View

[View](https://en.cppreference.com/w/cpp/ranges/view) 是对 range 的轻量化封装（适配器）。

标准库在命名空间 `std::ranges` 中提供了一些常用的 views：

|                            `VIEW`                            |         `for (auto x : VIEW) { use(x); }` 的传统写法         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| [`all_view{r}`](https://en.cppreference.com/w/cpp/ranges/all_view) |                  `for (auto x : r) use(x);`                  |
| [`filter_view{r, p}`](https://en.cppreference.com/w/cpp/ranges/filter_view) |             `for (auto x : r) if (p(x)) use(x);`             |
| [`transform_view{r, f}`](https://en.cppreference.com/w/cpp/ranges/transform_view) |                `for (auto x : r) use(f(x));`                 |
| [`take_view{r, n}`](https://en.cppreference.com/w/cpp/ranges/take_view) | `int i{0}; for (auto x : r) if (i++ == n) break; else use(x);` |
| [`drop_view{r, n}`](https://en.cppreference.com/w/cpp/ranges/drop_view) | `int i{0}; for (auto x : r) if (i++ < n) continue; else use(x);` |
| [`take_while_view{r, p}`](https://en.cppreference.com/w/cpp/ranges/take_while_view) |      `for (auto x : r) if (!p(x)) break; else use(x);`       |
| [`drop_while_view{r, p}`](https://en.cppreference.com/w/cpp/ranges/drop_while_view) |     `for (auto x : r) if (p(x)) continue; else use(x);`      |
| [`join_view{r}`](https://en.cppreference.com/w/cpp/ranges/join_view) |         `for (auto &y : r) for (auto x : y) use(x);`         |
| [`keys_view{r}`](https://en.cppreference.com/w/cpp/ranges/keys_view) |               `for (auto [x, y] : r) use(x);`                |
| [`values_view{r}`](https://en.cppreference.com/w/cpp/ranges/values_view) |               `for (auto [y, x] : r) use(x);`                |
| [`ref_view{r}`](https://en.cppreference.com/w/cpp/ranges/ref_view) |                 `for (auto &x : r) use(x);`                  |
|                         以下为生成器                         |                                                              |
| [`iota_view{y}`](https://en.cppreference.com/w/cpp/ranges/iota_view) |           `for (int i = 0: true; ++i) use(y + i);`           |
| [`iota_view{y, z}`](https://en.cppreference.com/w/cpp/ranges/iota_view) |            `for (auto x = y: x < z; ++x) use(x);`            |
| [`istream_view<double>{cin}`](https://en.cppreference.com/w/cpp/ranges/basic_istream_view) |             `double x; while (cin >> x) use(x);`             |

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

### Pipeline

标准库 range 及 view 支持 `|` 运算符，可以像在 Unix shell 中串联多个命令一样，串联多个 filters：

```cpp
void user(ranges::forward_range auto& r) {
  auto odd = [](int x) { return x % 2; };
  for (int x : r | views::filter(odd) | views::take(3)) {
    cout << x << ' ';
  }
}
// 等价于
void user_pre20(ranges::forward_range auto& r) {
  auto odd = [](int x) { return x % 2; };
  int cnt = 0;
  for (int x : r) {
    if (odd(x)) {
    	cout << x << ' ';
      if (++cnt == 3) {
        break;
      }
    }
  }
}
```

