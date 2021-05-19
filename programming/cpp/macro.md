---
title: 预定义宏
---

# 数值界限

```c
#include <cstdint>
PTRDIFF_MIN  // minimum value of object of `std::ptrdiff_t`
PTRDIFF_MAX  // maximum value of object of `std::ptrdiff_t`
SIZE_MAX     // maximum value of object of `std::size_t` 

#include <climits>
CHAR_BIT   // number of bits in a byte
CHAR_MIN   // minimum value of `char`
CHAR_MAX   // maximum value of `char`

SCHAR_MIN  // minimum value of `signed char`
 SHRT_MIN  // minimum value of `short`
  INT_MIN  // minimum value of `int`
 LONG_MIN  // minimum value of `long`
LLONG_MIN  // minimum value of `long long`

SCHAR_MAX  // maximum value of `signed char`
 SHRT_MAX  // maximum value of `short`
  INT_MAX  // maximum value of `int`
 LONG_MAX  // maximum value of `long`
LLONG_MAX  // maximum value of `long long`

 UCHAR_MAX // maximum value of `unsigned char`
 USHRT_MAX // maximum value of `unsigned short`
  UINT_MAX // maximum value of `unsigned int`
 ULONG_MAX // maximum value of `unsigned long`
ULLONG_MAX // maximum value of `unsigned long long`
```

# 调试代码

在 C/C++ 代码中，应大量使用 `assert()` 检查循环不变量、数据结构不变量。
编译时，开启 `-DNDEBUG` 选项，可屏蔽这些代码。

```c
#ifdef NDEBUG
#define assert(condition) ((void)0)
#else
#define assert(condition) /*implementation defined*/
#endif
```

类似地，可定义“调试期打印”函数：

```c
#ifndef NDEBUG
#define debug_printf(...) printf(__VA_ARGS__)
#else
#define debug_printf(...)
#endif
```

多个（预处理期）条件分支：

```c
#ifdef A /* same as `#if defined A` */
    /* code for A */
#elif defined B
    /* code for B */
#elif defined C
    /* code for C */
#endif
```
