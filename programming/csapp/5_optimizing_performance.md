---
title: 优化程序性能
---

# 编译器的优化能力及限制

```c
void twiddle1(long *xp, long *yp) {
  *xp += *yp;
  *xp += *yp;
}
void twiddle2(long *xp, long *yp) {
  *xp += 2* *yp;
}
```

当 `xp == yp` 时，这两个函数并不等价 —— 此时称它们互为『内存别名 (memory alias)』。
编译器无法判断程序意图，故不会将 `twiddle1` 优化为 `twiddle2`。

# 表达程序性能

现代处理器的『主频』通常达到 GHz，其倒数被称为『时钟周期 (clock cycle)』。
后者是度量指令运行时间的基本单位。

『CPE (Cycles Per Element)』比『Cycles Per Iteration』更适合用来度量『循环 (loop)』的性能。
这是因为『循环展开 (loop unrolling)』技术会在一次『迭代 (iteration)』中安排多个计算『单元 (element)』。

# 程序示例

```c
/* Create abstract data type for vector */
typedef double data_t;  /* `data_t` may also be `int`, `long`, `float` */
typedef struct {
  long len;
  data_t *data;
} vec_rec, *vec_ptr;
/* `IDENT` and `OP` may also be `1` and `*` respectively */
#define IDENT 0
#define OP +
```

测量结果表明：`data_t` 取作
- `int` 与 `long` 差别不大。
- `float` 与 `double` 差别不大。

最大程度保留数据抽象的原始版本：

```c
/* Implementation with maximum use of data abstraction */
void combine1(vec_ptr v, data_t *dest) {
  long i;
  *dest = IDENT;
  for (i = 0; i < vec_length(v); i++) {
    data_t val;
    get_vec_element(v, i, &val);
    *dest = *dest OP val;
  }
}
```

# 移出重复操作

尽管 `vec_length(v)` 具有 $O(1)$ 复杂度，在循环条件中反复调用仍是浪费：

```c
/* Move call to vec_length out of loop */
void combine2(vec_ptr v, data_t *dest) {
  long i;
  long n = vec_length(v);  /* 缓存循环不变量 */
  *dest = IDENT;
  for (i = 0; i < n; i++) {
    data_t val;
    get_vec_element(v, i, &val);
    *dest = *dest OP val;
  }
}
```

像 `strlen(s)` 这样具有 $O(N)$ 复杂度的操作，更应缓存其结果于循环体外：

```c
/* Sample implementation of library function strlen */
size_t strlen(const char *s) {
  long length = 0;
  while (*s != ’\0’) {
    s++;
    length++;
  }
  return length;
}

/* Convert string to lowercase: slow */
void lower1(char *s) {
  long i;
  for (i = 0; i < strlen(s); i++)
    if (s[i] >= ’A’ && s[i] <= ’Z’)
      s[i] -= (’A’ - ’a’);
}
```

# 减少函数调用

`combine2()` 的循环体中调用了 `get_vec_element(v, i, &val)`，它会检查 `i` 是否越界。
将数组的首地址『偷出』，可避开不必要的越界检查：

```c
data_t *get_vec_start(vec_ptr v) {
  return v->data;
}
/* Direct access to vector data */
void combine3(vec_ptr v, data_t *dest) {
  long i, n = vec_length(v);
  data_t *data = get_vec_start(v);  /* 偷出数组首地址 */
  *dest = IDENT;
  for (i = 0; i < n; i++) {
    *dest = *dest OP data[i];
  }
}
```

⚠️ 此优化并没有显著提升性能，因为循环体内还有其他低效操作。

# 减少内存访问

将中间结果缓存于寄存器中，而不是反复读写内存，可显著提高性能：

```c
/* Accumulate result in local variable */
void combine4(vec_ptr v, data_t *dest) {
  long i, n = vec_length(v);
  data_t *data = get_vec_start(v);
  data_t result = IDENT;  /* 缓存于寄存器中 */
  for (i = 0; i < n; i++) {
    result = result OP data[i];
  }
  *dest = result;
}
```
