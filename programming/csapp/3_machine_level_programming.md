# 程序的机器级表示

## 1 历史视野

### Intel

Intel 长期主导 笔记本、台式机、服务器 处理器市场。

里程碑产品：

|    名称    | 时间 | 主频（MHz） |     技术要点      |
| :--------: | :--: | :---------: | :-----------: |
|    8086    | 1978 |    5~10     |     16 位     |
|    i386    | 1985 |    16~33    |     32 位     |
| Pentium 4E | 2004 |  2800~3800  | 64 位、超线程 |
|   Core 2   | 2006 |  1060~3500  |     多核    |
|  Core i7   | 2008 |  1700~3900  |     四核、超线程  |

- i386 引入了 *IA32 (Intel Architecture 32-bit)* 指令集。
- Pentium 4E 采用了 *EM64T* 指令集（基本等价于 *x86-64* 指令集）。
- 使用这些处理器的计算机都属于 *CISC (Complex Instruction Set Computer)*，其性能通常不如 *RISC (Reduced Instruction Set Computer)*，但功能及通用性远胜后者。

### AMD

*AMD (Advanced Micro Devices)* 是 Intel 的主要竞争对手。

- 性能略逊于 Intel，但价格有优势。
- 在 64 位处理器的研发上，采取了渐进式策略，推出了 *x86-64* 指令集。

## 2 程序编码

|               英文名               |  中文名  |           含义           |
| :--------------------------------: | :------: | :----------------------: |
| ISA (Instruction Set Architecture) | 体系结构 |  指令集定义、寄存器命名  |
|         Micro-architecture         |  微架构  | ISA 的物理实现（微电路） |
|            Machine Code            |  机器码  | 机器读取或执行的 bit 流  |
|           Assembly Code            |  汇编码  |     机器码的文字形式     |

汇编码可见的信息：

- 程序计数器 (Program Counter)：下一条指令的地址（在 x86-64 中由 `%rip` 保存）。
- 寄存器 (Register)：位于 CPU 内部的临时数据存储器（顶级缓存）。
- 条件码 (Condition Code)：存储最近一条指令的状态，用于条件分支。
- 内存 (Memory)：可按字节寻址的（抽象）数组，用于存放数据及指令。

### 示例代码

```c
/* hello.c */
#include <stdio.h>
int main() {
  printf("hello\n");
  return 0;
}
```

|           步骤           |           命令            |        输出        |
| :----------------------: | :-----------------------: | :----------------: |
| 构建（编译、汇编、链接） |   `cc -o hello hello.c`   | 可执行文件 `hello` |
|      编译 (Compile)      |      `cc -S hello.c`      | 汇编文件 `hello.s` |
|     汇编 (Assemble)      |  `as -o hello.o hello.s`  | 目标文件 `hello.o` |
|       链接 (Link)        | `ld -o hello hello.o -lc` | 可执行文件 `hello` |
|   反汇编 (Disassemble)   |    `objdump -d hello`     |       汇编码       |

反汇编也可以在 *调试器 (debugger)* 中进行：
```shell
gdb hello  # 进入调试环境，引导符变为 (gdb)
(gdb) disassemble main  # 定位到 main() 函数，输出其汇编码
```
输出结果（可能随系统、编译器而变化）如下：
```assembly
Dump of assembler code for function main:
   0x0000000100000f60 <+0>:     push   %rbp
   0x0000000100000f61 <+1>:     mov    %rsp,%rbp
   0x0000000100000f64 <+4>:     sub    $0x10,%rsp
   0x0000000100000f68 <+8>:     movl   $0x0,-0x4(%rbp)
   0x0000000100000f6f <+15>:    lea    0x34(%rip),%rdi        # 0x100000faa
   0x0000000100000f76 <+22>:    mov    $0x0,%al
   0x0000000100000f78 <+24>:    callq  0x100000f8a
   0x0000000100000f7d <+29>:    xor    %ecx,%ecx
   0x0000000100000f7f <+31>:    mov    %eax,-0x8(%rbp)
   0x0000000100000f82 <+34>:    mov    %ecx,%eax
   0x0000000100000f84 <+36>:    add    $0x10,%rsp
   0x0000000100000f88 <+40>:    pop    %rbp
   0x0000000100000f89 <+41>:    retq   
End of assembler dump.
```
### 汇编代码

- 函数名下方的每一行分别对应一条指令。
  - 形如 `0x0000000100000f60` 的 64 位 16 进制数，表示各条指令的首地址。
  - `<+n>` 表示当前指令相对于函数入口的 *偏移量*（以字节为单位）。相邻两行的偏移量之差 等于 前一行指令的机器码长度。
  - 通常，指令的机器码长度与其使用频率成反比，最长 1 字节，最短 15 字节。
- 以 `%` 起始的 `%rsp`、`%rdi` 等符号表示 *寄存器*，用于存储临时数据：
  - 整数：长度为 1、2、4、8 字节，表示整数或地址。
  - 浮点数：长度为 4、8、10 字节，表示浮点数。
  - 没有聚合类型（数组、结构体）。
- 形如英文单词的 `mov`、`lea` 等符号表示 *指令*，用于
  - 对寄存器、内存中的数据作算术运算。
  - 在寄存器、内存间传递数据（双向读写）。
  - 无条件跳转、条件分支。

## 3 数据格式

### 指令后缀

| 后缀 |          名称           | 长度（bit） |    C 语言类型     |
| :--: | :---------------------: | :---------: | :---------------: |
| `b`  |          Byte           |      8      |      `char`       |
| `w`  |          Word           |     16      |      `short`      |
| `l`  |   (Long) double word    |     32      |       `int`       |
| `q`  |        Quad word        |     64      | `long` 或 `void*` |
| `s`  |    Single precision     |     32      |      `float`      |
| `l`  | (Long) double precision |     64      |     `double`      |

## 4 访问信息

### 整数寄存器

| 全 64 位 | 后 32 位 | 后 16 位 | 后 8 位 |     字面含义      |   实际含义   |      |
| :------: | :------: | :------: | :-----: | :---------------: | :----------: | ---- |
|  `%rax`  |  `%eax`  |  `%ax`   |  `%al`  |    accumulate     |    返回值    |      |
|  `%rbx`  |  `%ebx`  |  `%bx`   |  `%bl`  |       base        | 被调函数保留 |      |
|  `%rcx`  |  `%ecx`  |  `%cx`   |  `%cl`  |      counter      | 第 4 个实参  |      |
|  `%rdx`  |  `%edx`  |  `%dx`   |  `%dl`  |       data        | 第 3 个实参  |      |
|  `%rsi`  |  `%esi`  |  `%si`   |         |   source index    | 第 2 个实参  |      |
|  `%rdi`  |  `%edi`  |  `%di`   |         | destination index | 第 1 个实参  |      |
|  `%rbp`  |  `%ebp`  |  `%bp`   |         |   base pointer    | 被调函数保留 |      |
|  `%rsp`  |  `%esp`  |  `%sp`   |         |   stack pointer   | 函数调用栈尾 |      |
|  `%r8`   |  `%r8d`  |          |         |                   | 第 5 个实参  |      |
|  `%r9`   |  `%r9d`  |          |         |                   | 第 6 个实参  |      |
|  `%r10`  | `%r10d`  |          |         |                   | 主调函数保留 |      |
|  `%r11`  | `%r11d`  |          |         |                   | 主调函数保留 |      |
|  `%r12`  | `%r12d`  |          |         |                   | 被调函数保留 |      |
|  `%r13`  | `%r13d`  |          |         |                   | 被调函数保留 |      |
|  `%r14`  | `%r14d`  |          |         |                   | 被调函数保留 |      |
|  `%r15`  | `%r15d`  |          |         |                   | 被调函数保留 |      |

### 运算对象

| 表达式类型 |           格式           |   含义   |
| :--------: | :----------------------: | :------: |
| Immediate  |    以 `$` 起始的整数     | 整型常量 |
|  Register  |   以 `%` 起始的寄存器    | 局部变量 |
|   Memory   | 形如 `D(B,I,S)` 的表达式 | 内存寻址 |

其中 `D(B,I,S)` 表示取地址 `M[R[B] + S * R[I] + D]` 的值。
- `M` 表示 *Memory*，可以视为是一个以 *64 位整数* 为索引的整型数组。
- `R` 表示 *Register*，可以视为是一个以 *寄存器名称* 为索引的整型数组。
- `D` 表示 *Displacement*，可以是 1、2、4 字节整数，若缺省则视为 `0`。
- `B` 表示 *Base*，可以是 16 个整型寄存器之一，不可缺省。
- `I` 表示 *Index*，可以是 `%rsp` 外的 15 个整型寄存器之一，若缺省则 `Register[I]` 视为 `0`。
- `S` 表示 *Scale*，可以是 `1`、`2`、`4`、`8` 之一，若缺省值则视为 `0`。

### 移动数据

```assembly
movq source, destination
```

其中

- `mov` 后面的 `q` 表示 *quad-word*；有时也会是其他后缀，详见《[指令后缀](#指令后缀)》。
- `source` 及 `destination` 为该指令的 *[运算对象](#运算对象)*，且只能有一个为 Memory。

```c
 void swap(long* px, long* py) {
   long tx = *px;
   long ty = *py;
   *px = ty;
   *py = tx;
 }
```

```assembly
movq    (%rdi), %rax
movq    (%rsi), %rdx
movq    %rdx, (%rdi)
movq    %rax, (%rsi)
ret
```

### 压栈出栈

|   指令    |      含义      |              语义               |
| :-------: | :------------: | :-----------------------------: |
| `pushq s` | PUSH quad word | `R[%rsp] -= 8; M[R[%rsp]] = s;` |
| `popq d`  | POP quad word  | `d = M[R[%rsp]]; R[%rsp] += 8;` |


## 5 算术及逻辑运算

### 取地址运算

```assembly
leaq source, destination
```

- `lea` 由 *Load Effective Address* 的首字母构成，相当于 C 语言的 `p = &x[i]` 语句。
- `source` 只能是 Memory。
- `destination` 只能是 Register，用于存储 `source` 所表示的 *地址值*，但不访问该地址。
- 该指令速度极快，故编译器可能用它来分解算术运算，例如：
  ```c
  long m12(long x) { return x * 12; }
  ```
  通常被编译为
  ```assembly
  leaq    (%rdi,%rdi,2), %rax  # 相当于 t = x + 2 * x
  salq    $2, %rax             # 相当于 t <<= 2
  ret
  ```

### 一元运算

|    指令     |          含义          |   语义    |
| :---------: | :--------------------: | :-------: |
|   `inc d`   |       INCrement        |   `d++`   |
|   `dec d`   |       DECrement        |   `d--`   |
|   `neg d`   |         NEGate         | `d = -d`  |
|   `not d`   |       complement       | `d = ~d`  |

### 二元运算

|    指令     |     含义     |   语义   |
| :---------: | :----------: | :------: |
| `add s, d`  |     ADD      | `d += s` |
| `sub s, d`  |   SUBtract   | `d -= s` |
| `imul s, d` |   MULtiply   | `d *= s` |
| `xor s, d`  | eXclusive OR | `d ^= s` |
|  `or s, d`  |  bitwise OR  | `d |= s` |
| `and s, d`  | bitwise AND  | `d &= s` |

### 移位运算

|    指令    |              含义               |   语义    |    移出的空位    |
| :--------: | :-----------------------------: | :-------: | :--------------: |
| `shl k, d` |   SHift logically to the Left   | `d <<= k` |  在右端，补 `0`  |
| `sal k, d` | Shift Arithmeticly to the Left  | `d <<= k` |  在右端，补 `0`  |
| `shr k, d` |  SHift logically to the Right   | `d >>= k` |  在左端，补 `0`  |
| `sar k, d` | Shift Arithmeticly to the Right | `d >>= k` | 在左端，补符号位 |

## 6 控制

## 7 过程（函数）

## 8 数组分配与访问

## 9 异构数据结构（结构体）

## 10 结合控制与数据

## 11 浮点代码