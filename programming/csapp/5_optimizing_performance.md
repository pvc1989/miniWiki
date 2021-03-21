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

# 理解现代处理器

以 Intel 为代表的现代处理器能够同时执行多条指令，并且保证所得结果与顺序执行机器码的结果一致。
这种『乱序 (out-of-order)』执行指令的加速机制被称为『指令级并行 (instruction-level parallelism)』。

## 基本操作

『指令控制单元 (instruction control unit, ICU)』从『指令缓存 (instruction cache)』中读取将要被执行的指令（通常大大超前于当前指令），将其『解码 (decode)』为若干『初等操作 (primitive operation)』并交由『执行单元 (execution unit, EU)』去执行。

遇到[条件跳转](./3_machine_level_programming.md#跳转指令)指令时，ICU 会做出『分支预测 (branch prediction)』，即令 EU 在所预测的分支上提前执行一些指令。
若预测错误，则 EU 会丢弃所得结果，并在正确的分支上重新计算。

EU 由『功能单元 (functional unit, FU)』和『数据缓存 (data cache)』组成。
其中，一个 FU 可兼容多种功能，一种功能也可由多个 FU 完成。

ICU 中的『退休单元 (retirement unit)』用于确保乱序执行指令的结果与顺序执行这些指令的结果一致。
只有当分支预测正确时，才会更新寄存器。

在 EU 中，一条（初等）操作的结果，可以直接转发给后续操作，而不需要读写寄存器。
该机制被称为『寄存器重命名 (register renaming)』。

## 功能单元性能

性能指标：
- 【延迟 (latency)】执行该操作所需的总时间。
- 【发起时间 (issue time)】同种操作之间的最短间隔。
- 【容量 (capacity)】可用于该操作的功能单元数量。
- 【吞吐量 (throughput)】单位时间内可发起的同种操作的数量。

常用算术运算：
- `+` 及 `*` 可被『管道』加速，故『发起时间』只需一个时钟周期。
- `/` 不能被管道加速，故『发起时间』等于『延迟』。

## 处理器操作的抽象模型

```gas
# Inner loop of combine4. data_t = double, OP = *
# acc in %xmm0, data+i in %rdx, data+length in %rax
.L25: loop:
  vmulsd (%rdx), %xmm0, %xmm0  # Multiply acc by data[i]
  addq $8, %rdx                # Increment data+i
  cmpq %rax, %rdx              # Compare to data+length
  jne .L25                     # If !=, goto loop
```

四类寄存器：
- 【只读】`%rax`
- 【只写】
- 【局部】条件码寄存器
- 【循环】`%rdx`、`%xmm0`

指令 `vmulsd` 被解码为 `load` 与 `mul` 两步操作，后者依赖于前者的结果。

在所有操作中，计算浮点数乘积的 `mul` 操作耗时最长，且其他操作（如更新计数器 `%rdx` 的 `add`）可以同步（并行）执行，故由 `mul` 串联所得的『关键路径 (critical path)』决定了该循环耗时的『下界』。

# 下一节

『偷出』

