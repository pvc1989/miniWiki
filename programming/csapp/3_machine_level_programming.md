# 程序的机器级表示

## 1 历史视野

### Intel

Intel 长期主导 笔记本、台式机、服务器 处理器市场。

里程碑产品：

|    名称    | 时间 | 主频（单位：MHz） |     技术要点      |
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

### Moore's Law

> Gordon Moore (1965): the number of transistors per chip would double every year for the next 10 years.

> 实际：处理器上的晶体管数量平均每 18 个月翻一番。

## 2 程序编码

### 编码形式

|               英文名               |  中文名  |           含义            |
| :--------------------------------: | :------: | :-----------------------: |
| ISA (Instruction Set Architecture) | 体系结构 |  机器码格式及行为的定义   |
|         Micro-architecture         |  微架构  | ISA 的物理实现（微电路）  |
|            Machine Code            |  机器码  | 用 0/1 序列表示的指令序列 |
|           Assembly Code            |  汇编码  | 用汇编语言表示的指令序列  |
|            Source Code             |  源代码  | 用高级语言表示的指令序列  |

汇编码可见（源代码不可见）的信息：
- 程序计数器 (Program Counter)：下一条指令的地址（在 x86-64 中由 `%rip` 保存）。
- 寄存器 (Register)：位于 CPU 内部的临时数据存储器（顶级缓存）。
- 条件码 (Condition Code)：存储最近一条指令的状态，用于条件分支。
- 内存 (Memory)：可按字节寻址的（抽象）数组，用于存放数据及指令。

源代码可见（汇编码不可见）的信息：

- 变量名
- 聚合数据结构
- 不同类型的指针
- 指针与整数的区别

汇编码的可读性介于机器码与源代码之间：

- 汇编语言由机器（处理器架构）决定，面向不同机器的汇编程序员需要学习不同版本的汇编语言。《CS:APP3e》只介绍  x86-64 这一目前最主流的版本，并重点关注 GCC 与 Linux 用到的那一小部分。
- 高级语言隔离了这种机器相关性（具有跨平台性），编译器负责将同一份高级语言代码在不同机器上转换为相应的汇编码。

对高级语言程序员而言，学习汇编语言主要是为了 *读* 而不是 *写* 汇编码：
- 理解编译优化，分析代码效率。
- 理解、分析代码的运行期行为（调试）。
- 发现、修复系统程序的安全漏洞。

### 示例代码

```c
/* hello.c */
#include <stdio.h>
int main() {
  printf("hello\n");
  return 0;
}
```

|         步骤         |             命令             |         输出         |
| :------------------: | :--------------------------: | :------------------: |
|    构建（四合一）    |    `cc -o hello hello.c`     |  可执行文件 `hello`  |
| 预处理 (Preprocess)  |  `cc -E hello.c > hello.i`   |   含库函数的源代码   |
|    编译 (Compile)    |       `cc -S hello.i`        |  汇编文件 `hello.s`  |
|   汇编 (Assemble)    |   `as -o hello.o hello.s`    |  目标文件 `hello.o`  |
|     链接 (Link)      |  `ld -o hello hello.o -lc`   |  可执行文件 `hello`  |
| 反汇编 (Disassemble) | `objdump -d hello > hello.d` | 由机器码反推的汇编码 |

⚠️ 如果用 GCC 编译，可加编译选项 `-Og` 以使机器码与源代码具有大致相同的结构。

汇编文件 `hello.s` 的内容大致如下（可能随系统、编译器而变化），其中以 `.` 开头的行是用于引导汇编器、链接器的指令，可忽略：

```assembly
_main:                                  ## @main
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	leaq	L_str(%rip), %rdi
	callq	_puts
	xorl	%eax, %eax
	popq	%rbp
	retq
```

目标文件 `hello.o` 及可执行文件 `hello` 的反汇编结果大致如下（可能随系统、编译器而变化）：
```assembly
# objdump -d hello.o
0000000000000000 _main:
       0: 55                            pushq   %rbp
       1: 48 89 e5                      movq    %rsp, %rbp
       4: 48 8d 3d 09 00 00 00          leaq    9(%rip), %rdi
       b: e8 00 00 00 00                callq   0 <_main+0x10>
      10: 31 c0                         xorl    %eax, %eax
      12: 5d                            popq    %rbp
      13: c3                            retq
# objdump -d hello
0000000100000f70 _main:
100000f70: 55                           pushq   %rbp
100000f71: 48 89 e5                     movq    %rsp, %rbp
100000f74: 48 8d 3d 2b 00 00 00         leaq    43(%rip), %rdi
100000f7b: e8 04 00 00 00               callq   4 <dyld_stub_binder+0x100000f84>
100000f80: 31 c0                        xorl    %eax, %eax
100000f82: 5d                           popq    %rbp
100000f83: c3                           retq
```
反汇编也可以在 *调试器 (debugger)* 中进行：

```shell
gdb hello  # 进入调试环境，引导符变为 (gdb)
(gdb) disassemble main  # 定位到 main() 函数，输出其汇编码
```
输出结果（可能随系统、编译器而变化）如下：
```assembly
Dump of assembler code for function main:
   0x0000000100000f70 <+0>:     push   %rbp
   0x0000000100000f71 <+1>:     mov    %rsp,%rbp
   0x0000000100000f74 <+4>:     lea    0x2b(%rip),%rdi        # 0x100000fa6
   0x0000000100000f7b <+11>:    callq  0x100000f84
   0x0000000100000f80 <+16>:    xor    %eax,%eax
   0x0000000100000f82 <+18>:    pop    %rbp
   0x0000000100000f83 <+19>:    retq   
End of assembler dump.
```
### 汇编要素

- 函数名下方的每一行分别对应一条指令。
  - 形如 `100000f70` 或 `0x0000000100000f60` 的 64 位 16 进制整数，表示各条指令的首地址。由的 `hello.o` 与 `hello` 的反汇编结果可见，一些函数的地址会被 *链接器 (linker)* 修改，详见《[链接](#./linking.md)》。
  - 首地址后面的若干 16 进制整数（每 8 位一组，即每组 1 字节），表示该行指令的 *机器码*，由此可以算出各条指令的长度（字节数）。整个函数的机器码可以在 `gdb` 中连续打印：
    ```shell
    gdb hello
    (gdb) x/20xb main  # 打印始于 main 的 20 个字节
    ```
    得到
    ```assembly
    0x100000f70 <main>:     0x55    0x48    0x89    0xe5    0x48    0x8d    0x3d      0x2b
    0x100000f78 <main+8>:   0x00    0x00    0x00    0xe8    0x04    0x00    0x00      0x00
    0x100000f80 <main+16>:  0x31    0xc0    0x5d    0xc3
    ```
  - `<+n>` 表示当前指令相对于函数入口的 *偏移量*（以字节为单位）。由相邻两行的偏移量之差也可以算出前一行指令的机器码长度。
  - 指令的 *机器码长度* 与其 *使用频率* 及 *[运算对象](#运算对象)个数* 大致成反比，最长 1 字节，最短 15 字节。
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

| 全 64 位 | 后 32 位 | 后 16 位 | 后 8 位 |     字面含义      |   实际含义   |
| :------: | :------: | :------: | :-----: | :---------------: | :----------: |
|  `%rax`  |  `%eax`  |  `%ax`   |  `%al`  |    accumulate     |    返回值    |
|  `%rbx`  |  `%ebx`  |  `%bx`   |  `%bl`  |       base        | 被调函数保留 |
|  `%rcx`  |  `%ecx`  |  `%cx`   |  `%cl`  |      counter      | 第 4 个实参  |
|  `%rdx`  |  `%edx`  |  `%dx`   |  `%dl`  |       data        | 第 3 个实参  |
|  `%rsi`  |  `%esi`  |  `%si`   | `%sil`  |   source index    | 第 2 个实参  |
|  `%rdi`  |  `%edi`  |  `%di`   | `%dil`  | destination index | 第 1 个实参  |
|  `%rbp`  |  `%ebp`  |  `%bp`   | `%bpl`  |   base pointer    | 被调函数保留 |
|  `%rsp`  |  `%esp`  |  `%sp`   | `%spl`  |   stack pointer   | 函数调用栈尾 |
|  `%r8`   |  `%r8d`  |          | `%r8b`  |                   | 第 5 个实参  |
|  `%r9`   |  `%r9d`  |          | `%r9b`  |                   | 第 6 个实参  |
|  `%r10`  | `%r10d`  |          | `%r10b` |                   | 主调函数保留 |
|  `%r11`  | `%r11d`  |          | `%r11b` |                   | 主调函数保留 |
|  `%r12`  | `%r12d`  |          | `%r12b` |                   | 被调函数保留 |
|  `%r13`  | `%r13d`  |          | `%r13b` |                   | 被调函数保留 |
|  `%r14`  | `%r14d`  |          | `%r14b` |                   | 被调函数保留 |
|  `%r15`  | `%r15d`  |          | `%r15b` |                   | 被调函数保留 |

每个 64 位寄存器的后 32、16、8 位，都可以被当做“短”寄存器来访问。约定：

- 生成 1、2 字节结果的指令，不会修改其余字节。
- 生成 4 字节结果的指令，会将前 4 个字节置 `0`。

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
- `B` 表示 *Base*，可以是 16 个整型寄存器之一，若缺省则 `Register[B]` 视为 `0`。
- `I` 表示 *Index*，可以是 `%rsp` 外的 15 个整型寄存器之一，若缺省则 `Register[I]` 视为 `0`。
- `S` 表示 *Scale*，可以是 `1`、`2`、`4`、`8` 之一，若缺省值则视为 `0`。

### 移动数据

```assembly
movq source, destination
movl source, destination
movw source, destination
movb source, destination
```
- `source` 及 `destination` 为该指令的 *[运算对象](#运算对象)*，且只能有一个为 Memory。
- `mov` 后面的 `q` 表示 `destination` 的大小为 *quad-word*；其他后缀的含义见《[指令后缀](#指令后缀)》。
- `mov` 的一个主要变种是 `movabs`：
  
  - 若 `movq` 的 `source` 是 immediate，则只能是 32 位带符号整数，其符号位将被填入 `desination` 的前 32 位。若要阻止填充，则应使用 `movl` 等。
  - 若 `movabsq` 的 `source` 是 immediate，则可以是 64 位整数，此时 `desination` 必须是 register。
- `mov` 还有另外几个变种：
  
  ```assembly
  movz s, d  # d = ZeroExtend(s)
    movzbw s, d
    movzbl s, d
    movzwl s, d
    movzbq s, d  # 通常被 movzbl s, d 代替，理由同 movzlq s, d
    movzwq s, d
    # movzlq s, d 不存在，其语义可通过 movl s, d 实现，这是因为：
    # 生成 4 字节结果的指令，会将前 4 个字节置 `0`。
  movs s, d  # d = SignExtend(s)
    movsbw s, d
    movsbl s, d
    movswl s, d
    movsbq s, d
    movswq s, d
    movslq s, d
  cltq  # 等效于 movslq %eax, %rax 但其机器码更短
  ```
- 以下示例体现了这几个版本的区别：
  ```assembly
  movabsq $0x0011223344556677, %rax  # %rax = 0011223344556677
  movb    $-1,                 %al   # %rax = 00112233445566FF
  movw    $-1,                 %ax   # %rax = 001122334455FFFF
  movl    $-1,                 %eax  # %rax = 00000000FFFFFFFF
  movq    $-1,                 %rax  # %rax = FFFFFFFFFFFFFFFF
  movabsq $0x0011223344556677, %rax  # %rax = 0011223344556677
  movb    $0xAA,               %dl   # %dl  = AA
  movb    %dl,                 %al   # %rax = 00112233445566AA
  movsbq  %dl,                 %rax  # %rax = FFFFFFFFFFFFFFAA
  movzbq  %dl,                 %rax  # %rax = 00000000000000AA
  ```

### 交换数据

虽然不存在直接交换数据的指令，但可以通过一组 `mov` 来实现：

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

x86-64 规定：栈顶元素的地址保存在寄存器 `%rsp` 中，并且小于栈内其他元素的地址。

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

其运算对象既是 source 又是 destination。

|    指令     |          含义          |   语义    |
| :---------: | :--------------------: | :-------: |
|   `inc d`   |       INCrement        |   `d++`   |
|   `dec d`   |       DECrement        |   `d--`   |
|   `neg d`   |         NEGate         | `d = -d`  |
|   `not d`   |       complement       | `d = ~d`  |

### 二元运算

第一个运算对象是 source，第二个运算对象既是 source 又是 destination。

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

其中 `k` 可以是 immediate，也可以是 `%cl` 这个特殊的 register：
- 后缀为 `b` 的版本，取 `%cl` 的后 3 位所表示的值，至多移 7 位。
- 后缀为 `w` 的版本，取 `%cl` 的后 4 位所表示的值，至多移 15 位。
- 后缀为 `l` 的版本，取 `%cl` 的后 5 位所表示的值，至多移 31 位。
- 后缀为 `q` 的版本，取 `%cl` 的后 6 位所表示的值，至多移 63 位。

### 特殊算术运算

x86-64 还提供了一些长达 128 位的整数（Intel 称之为 *oct word*）的算术运算指令：

- 一元乘法：
  - `imulq s` 为带符号乘法，`mulq s` 为无符号乘法
  - 语义为 `R[%rdx]:R[%rax] = s * R[%rax]`
- 一元除法：
  - `idivq s` 为带符号除法，`divq s` 为无符号除法
  - 二者均以 `R[%rdx]:R[%rax]` 为 *被除数 (dividend)*，以 `s` 为 *除数 (divisor)*，所得的 *商 (quotient)* 存入 `%rax`，*余数 (remainder)* 存入 `%rdx`
  - ⚠️ 不存在“二元除法”指令
- `cqto` 用于构造带符号除法的被除数：
  - 字面含义为 Convert Quad-word To Oct-word
  - 语义为 `R[%rdx]:R[%rax] = SignExtend(R[%rax])`

## 6 控制

### 条件码

CPU 用一组名为 *条件码 (condition code)* 的 1-bit 寄存器记录最近一条指令的状态，这些状态被用于控制条件分支。最常用的条件码有如下几个：

| 符号 |     名称      |             含义              |
| :--: | :-----------: | :---------------------------: |
| `CF` |  Carry  Flag  | 最近一条指令触发 *无符号溢出* |
| `ZF` |   Zero Flag   |    最近一条指令获得 *零值*    |
| `SF` |   Sign Flag   |    最近一条指令获得 *负值*    |
| `OF` | Overflow Flag | 最近一条指令触发 *带符号溢出* |

一些（不显然的）约定：
- `leaq` 指令不改变条件码。
- 逻辑运算将 `CF` 及 `OF` 置零。
- 移位运算将 `CF` 设为最后一个被移出的位，而 `OF` 仅在 *移动一位* 时会被修改。
- 自增自减将 `OF` 及 `ZF` 置一，并保持 `CF` 不变。（原因较复杂）

除[算术及逻辑运算](#算术及逻辑运算)指令，以下指令（这里省略[指令后缀](#指令后缀)）也可以修改条件码：

- `cmp a, b` 根据 `sub a, b` 指令的结果设置条件码（但不执行 `sub` 指令）。
- `test a, b` 根据 `and a, b` 指令的结果设置条件码（但不执行 `and` 指令）。

### 读取条件码

高级语言代码经常将逻辑表达式的结果用整数（`0` 或 `1`）表示，这在汇编码中是通过 `set_` 系列指令来实现的：


|     指令     |       后缀含义       |          语义          |
| :----------: | :------------------: | :--------------------: |
| `set[e|z] d` |    Equal \| Zero     |        `d = ZF`        |
|   `sets d`   |  Signed (Negative)   |        `d = SF`        |
|   `setg d`   | Greater (signed `>`) | `d = ~(SF ^ OF) & ~ZF` |
|   `setl d`   |  Less (signed `<`)   |     `d = SF ^ OF`      |
|   `seta d`   | Above (unsigned `>`) |    `d = ~CF & ~ZF`     |
|   `setb d`   | Below (unsigned `<`) |        `d = CF`        |

表中只列出了几个有代表性的指令，遇到其他指令可以根据以下规则猜出其语义：
- 后缀前端的 `n` 表示 *Not*，后端的 `e` 表示 `or Equal`，因此 `setnle` 就表示 *(set when) Not Less or Equal*，类似的指令可按此规则解读。
- 某些指令具有同义词（例如 `setg` 与 `setnle`），它们具有相同的机器码。编译器、反汇编器在生成汇编码时，从同义词中任选其一。

以上指令根据（前一条 `cmp` 或 `test` 指令的）比较结果，将单字节的 `d` 设为 `0` 或 `1`；为了获得 32 或 64 位的 `0` 或 `1`，通常需配合 `movzbq` 或 `xorl` 将更高位清零：

```c
int greater(long x, long y) { return x > y; }
```

```assembly
cmpq    %rsi, %rdi
setg    %al         # 将最后一位设为 0 或 1
movzbl  %al, %eax   # 将前 7 位清零
```

```assembly
xorl    %eax, %eax  # 将全 8 位清零
cmpq    %rsi, %rdi
setg    %al         # 将最后一位设为 0 或 1
```

关于 `setl` 与 `setb` 语义的解释：

- `setl` 用于 *带符号小于*，有两种情形：
  - `a - b` 未溢出（`OF == 0`）且表示一个负数（`SF == 1`）
  - `a - b` 向下溢出（`OF == 1`）且表示一个正数（`SF == 0`）
- `setb` 用于 *无符号小于*，只有一种情形：
  - `a - b` 向下溢出（`CF == 1`）

## 7 过程（函数）

## 8 数组分配与访问

## 9 异构数据结构（结构体）

## 10 结合控制与数据

## 11 浮点代码