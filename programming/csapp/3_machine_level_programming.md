# 程序的机器级表示

## 1 历史视野

### Intel

Intel 长期主导 笔记本、台式机、服务器 处理器市场。

里程碑产品：

|    名称    | 时间 | 主频（MHz） |   技术要点    |
| :--------: | :--: | :---------: | :-----------: |
|    8086    | 1978 |    5~10     |     16 位     |
|    i386    | 1985 |    16~33    |     32 位     |
| Pentium 4E | 2004 |  2800~3800  | 64 位、超线程 |
|   Core 2   | 2006 |  1060~3500  |     多核      |
|  Core i7   | 2008 |  1700~3900  | 四核、超线程  |

- i386 引入了 ***IA32 (Intel Architecture 32-bit)*** 指令集。
- Pentium 4E 采用了 ***EM64T (Extended Memory 64 Technology)*** 指令集（基本等价于 ***x86-64*** 指令集）。
- 使用这些处理器的计算机都属于 ***CISC (Complex Instruction Set Computer)***，其性能通常不如 ***RISC (Reduced Instruction Set Computer)***，但功能及通用性远胜后者。

### AMD

***AMD (Advanced Micro Devices)*** 是 Intel 的主要竞争对手。

- 性能略逊于 Intel，但价格有优势。
- 在 64 位处理器的研发上，采取了渐进式策略，推出了 ***x86-64*** 指令集。

### Moore's Law

|                     Gordon Moore (1965)                      |                 历史统计值                 |
| :----------------------------------------------------------: | :----------------------------------------: |
| The number of transistors per chip would double every year for the next 10 years. | 处理器上的晶体管数量平均每 18 个月翻一番。 |

## 2 程序编码

### 编码形式

|               英文名               |   中文名   |             含义              |
| :--------------------------------: | :--------: | :---------------------------: |
| ISA (Instruction Set Architecture) | 指令集架构 |    机器码格式及行为的定义     |
|         Micro-architecture         |   微架构   |   ISA 的物理实现（微电路）    |
|            Machine Code            |   机器码   | 用 `0`/`1` 序列表示的指令序列 |
|           Assembly Code            |  汇编代码  |   用汇编语言表示的指令序列    |
|            Source Code             |   源代码   |   用高级语言表示的指令序列    |

汇编代码可见（源代码不可见）的信息：

- ***程序计数器 (Program Counter)***：下一条指令的地址（在 x86-64 中由 `%rip` 保存）。
- ***寄存器 (Register)***：位于 CPU 内部的临时数据存储器（顶级缓存）。
- ***条件码 (Condition Code)***：存储最近一条指令的状态，用于条件分支。
- ***内存 (Memory)***：可按字节寻址的（抽象）数组，用于存放数据及指令。

源代码可见（汇编代码不可见）的信息：

- 变量名
- 聚合数据结构
- 不同类型的指针
- 指针与整数的区别

汇编代码的可读性介于机器码与源代码之间：

- 汇编语言是指令集的文字表示，因此首先由机器（处理器架构）决定：
  - 本书只介绍  x86-64 这一目前最主流的版本，并重点关注 GCC 与 Linux 用到的那一小部分。
  - x86-64 指令集的完整定义参见《[Intel® 64 and IA-32 Architectures Software Developer’s Manual, Volume 2: Instruction Set Reference](https://www.intel.com/content/www/us/en/architecture-and-technology/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.html)》。
  
- 即使在同一台机器上，汇编代码也可以有不同格式，见《[汇编代码格式](#汇编代码格式)》。
- 高级语言隔离了这种机器相关性，因而具有跨平台性。编译器负责将同一份高级语言代码在不同机器上转换为相应的汇编代码。

对日常使用高级语言的程序员而言，学习汇编语言主要是为了 *读* 而不是 *写* 汇编代码：

- 理解编译优化，分析代码效率。
- 理解、分析代码的运行期行为。
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

|          工具           |             命令             |          输出          |
| :---------------------: | :--------------------------: | :--------------------: |
|   工具链 (Tool Chain)   |    `cc -o hello hello.c`     |   可执行文件 `hello`   |
| 预处理器 (Preprocessor) |  `cc -E hello.c > hello.i`   |    含库函数的源代码    |
|    编译器 (Compiler)    |       `cc -S hello.i`        |   汇编文件 `hello.s`   |
|   汇编器 (Assembler)    |   `as -o hello.o hello.s`    |   目标文件 `hello.o`   |
|     链接器 (Linker)     |  `ld -o hello hello.o -lc`   |   可执行文件 `hello`   |
| 反汇编器 (Disassembler) | `objdump -d hello > hello.d` | 由机器码反推的汇编代码 |

⚠️ 如果用 `gcc` 编译，可加编译选项 `-Og` 使 *机器码* 与 *源代码* 具有大致相同的结构。

汇编文件 `hello.s` 的内容大致如下（可能因 操作系统、编译器 不同而存在差异），其中以 `.` 开头的行是用于引导 汇编器、链接器 的指令，可忽略：

```gas
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

目标文件 `hello.o` 及可执行文件 `hello` 的反汇编结果大致如下（可能因 操作系统、编译器 不同而存在差异）：

```gas
; objdump -d hello.o
0000000000000000 _main:
       0: 55                            pushq   %rbp
       1: 48 89 e5                      movq    %rsp, %rbp
       4: 48 8d 3d 09 00 00 00          leaq    9(%rip), %rdi
       b: e8 00 00 00 00                callq   0 <_main+0x10>
      10: 31 c0                         xorl    %eax, %eax
      12: 5d                            popq    %rbp
      13: c3                            retq
; objdump -d hello
0000000100000f70 _main:
100000f70: 55                           pushq   %rbp
100000f71: 48 89 e5                     movq    %rsp, %rbp
100000f74: 48 8d 3d 2b 00 00 00         leaq    43(%rip), %rdi
100000f7b: e8 04 00 00 00               callq   4 <dyld_stub_binder+0x100000f84>
100000f80: 31 c0                        xorl    %eax, %eax
100000f82: 5d                           popq    %rbp
100000f83: c3                           retq
```

反汇编也可以在 [***调试器 (debugger)***](../debug/README.md) 中进行：

```shell
gdb hello  # 进入调试环境，引导符变为 (gdb)
(gdb) disassemble main  # 定位到 main() 函数，输出其汇编代码
```

输出结果（可能因 操作系统、编译器、调试器 不同而存在差异）如下：

```gas
Dump of assembler code for function main:
   0x0000000100000f70 <+0>:     push   %rbp
   0x0000000100000f71 <+1>:     mov    %rsp,%rbp
   0x0000000100000f74 <+4>:     lea    0x2b(%rip),%rdi
   0x0000000100000f7b <+11>:    callq  0x100000f84
   0x0000000100000f80 <+16>:    xor    %eax,%eax
   0x0000000100000f82 <+18>:    pop    %rbp
   0x0000000100000f83 <+19>:    retq   
End of assembler dump.
```

### 汇编要素

- 函数名下方的每一行分别对应一条 ***指令 (instruction)***：

  - 形如 `100000f70` 或 `0x0000000100000f60` 的 64 位 16 进制整数，表示各条指令的首地址。由的 `hello.o` 与 `hello` 的反汇编结果可见，一些函数的地址会被 [***链接器 (linker)***](./linking.md) 修改。

  - 首地址后面的若干 16 进制整数（每 8 位一组，即每组 1 字节），表示该行指令的 *机器码*，由此可以算出各条指令的长度（字节数）。整个函数的机器码可在调试环境中打印：

    ```shell
    gdb hello
    (gdb) x/20xb main  # 以 十六进制 打印始于 main 的 20 字节
    0x100000f70 <main>:     0x55    0x48    0x89    0xe5    0x48    0x8d    0x3d      0x2b
    0x100000f78 <main+8>:   0x00    0x00    0x00    0xe8    0x04    0x00    0x00      0x00
    0x100000f80 <main+16>:  0x31    0xc0    0x5d    0xc3
    ```
    
  - `<+n>` 表示当前指令相对于函数入口（以字节为单位）的 ***偏移量 (offset)***。由相邻两行的偏移量之差也可以算出前一行指令的机器码长度。
  
  - 指令的 *机器码长度* 与其 *使用频率* 及 *[运算对象](#运算对象)个数* 大致成反比（类似于 Hoffman 编码），最长 1 字节，最短 15 字节。
  
- 形如英文单词的 `mov` 等符号表示 *指令名*，用于

  - 对寄存器、内存中的数据作算术运算。
  - 在寄存器、内存间传递数据（双向读写）。
  - 无条件跳转、条件分支。

- 以 `%` 起始的 `%rsp` 等符号表示 *寄存器名*，用于存储临时数据：

  - [整数](#整型寄存器)：长度为 1、2、4、8 字节，用于存储整数或地址。
  - 浮点数：长度为 4、8、10 字节，用于存储浮点数。
  - 没有聚合类型（数组、结构体）。

⚠️ *指令* 的完整含义是由 *指令名* 与 *运算对象* 所构成的整体，但有时也将 *指令名* 简称为 *指令*。

## 3 数据格式

### 指令后缀

| 后缀 |               名称                | 长度（bit） |    C 语言类型     |
| :--: | :-------------------------------: | :---------: | :---------------: |
| `b`  |           Byte（字节）            |      8      |      `char`       |
| `w`  |          Word（单倍词）           |     16      |      `short`      |
| `l`  |   (Long) double word（双倍词）    |     32      |       `int`       |
| `q`  |        Quad word（四倍词）        |     64      | `long` 或 `void*` |
| `s`  |    Single precision（单精度）     |     32      |      `float`      |
| `l`  | (Long) double precision（双精度） |     64      |     `double`      |

## 4 访问信息

### 整型寄存器

| 全 64 位 | 后 32 位 | 后 16 位 | 后 8 位 |     字面含义      |   实际含义   |
| :------: | :------: | :------: | :-----: | :---------------: | :----------: |
|  `%rax`  |  `%eax`  |  `%ax`   |  `%al`  |    Accumulate     |    返回值    |
|  `%rbx`  |  `%ebx`  |  `%bx`   |  `%bl`  |       Base        | 被调函数保存 |
|  `%rcx`  |  `%ecx`  |  `%cx`   |  `%cl`  |      Counter      | 第 4 个实参  |
|  `%rdx`  |  `%edx`  |  `%dx`   |  `%dl`  |       Data        | 第 3 个实参  |
|  `%rsi`  |  `%esi`  |  `%si`   | `%sil`  |   Source Index    | 第 2 个实参  |
|  `%rdi`  |  `%edi`  |  `%di`   | `%dil`  | Destination Index | 第 1 个实参  |
|  `%rbp`  |  `%ebp`  |  `%bp`   | `%bpl`  |   Base Pointer    | 被调函数保存 |
|  `%rsp`  |  `%esp`  |  `%sp`   | `%spl`  |   Stack Pointer   | 函数调用栈顶 |
|  `%r8`   |  `%r8d`  |          | `%r8b`  |                   | 第 5 个实参  |
|  `%r9`   |  `%r9d`  |          | `%r9b`  |                   | 第 6 个实参  |
|  `%r10`  | `%r10d`  |          | `%r10b` |                   | 主调函数保存 |
|  `%r11`  | `%r11d`  |          | `%r11b` |                   | 主调函数保存 |
|  `%r12`  | `%r12d`  |          | `%r12b` |                   | 被调函数保存 |
|  `%r13`  | `%r13d`  |          | `%r13b` |                   | 被调函数保存 |
|  `%r14`  | `%r14d`  |          | `%r14b` |                   | 被调函数保存 |
|  `%r15`  | `%r15d`  |          | `%r15b` |                   | 被调函数保存 |

每个 64 位寄存器的 *后 32、16、8 位* 都可以被当作“短”寄存器来访问，并且约定

- 在 *后 2 字节* 上做运算的指令 不会修改 *前 6 字节*。
- 在 *后 4 字节* 上做运算的指令 会将 *前 4 字节* 设为 `0`。

### 运算对象

在汇编代码中，位于 *指令名* 后方、以逗号分隔的表达式称为 ***运算对象 (operand)***。运算对象有三种类型：

|     表达式类型     |        格式         |   含义   |
| :----------------: | :-----------------: | :------: |
| 即时数 (Immediate) | 以 `$` 起始的整数值 | 整型常量 |
| 寄存器 (Register)  | 以 `%` 起始的寄存器 | 局部变量 |
| 内存地址 (Memory)  |   形如 `D(B,I,S)`   | 内存地址 |

其中 `D(B,I,S)` 表示由 `R[B]+S*R[I]+D` 算出的地址值（除 [`lea`](#加载有效地址) 外均读取 `M[R[B]+S*R[I]+D]` 的值），各符号含义如下：

| 符号 |         名称          |                   解释                    |   缺省值   |
| :--: | :-------------------: | :---------------------------------------: | :--------: |
| `M`  |     内存 (Memory)     |      一个以 *64 位整数* 为索引的数组      |            |
| `R`  |   寄存器 (Register)   |      一个以 *寄存器名* 为索引的数组       |            |
| `B`  |     基础值 (Base)     | 可以是 16 个[整型寄存器](#整型寄存器)之一 | `R[] == 0` |
| `I`  |    索引值 (Index)     |  可以是 `%rsp` 外的 15 个整型寄存器之一   | `R[] == 0` |
| `S`  |    比例值 (Scale)     |      可以是 `1`、`2`、`4`、`8` 之一       |    `0`     |
| `D`  | 位移值 (Displacement) |          可以是 1、2、4 字节整数          |    `0`     |

### 移动数据

```gas
movq source, destination
movl source, destination
movw source, destination
movb source, destination
```

- `source` 及 `destination` 为该指令的[运算对象](#运算对象)，且只能有一个为内存地址。

- `mov` 后面的 `q` 表示 `destination` 的大小为一个 *Quad word*；其他后缀的含义见《[指令后缀](#指令后缀)》。

- `mov` 的一个重要变种是 `movabs`，二者的区别在于

  - 若 `movq` 的 `source` 为 *即时数*，则只能是 32 位带符号整数，其符号位将被填入 `desination` 的前 32 位。若要阻止填充，则应使用 `movl` 等。
  - 若 `movabsq` 的 `source` 为 *即时数*，则可以是 64 位整数，此时 `desination` 必须是 *寄存器*。

- `mov` 还有两类用于 *短整数* 向 *长整数* 扩展的变种：

  ```gas
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
  ```
- `cltq` 是 `movslq %eax, %rax` 的简写（机器码更短）。
- 以下示例体现了这几个版本的区别：

  ```gas
  movabsq $0x0011223344556677, %rax  # %rax = 0011223344556677
  movb    $-1,                 %al   # %rax = 00112233445566FF
  movw    $-1,                 %ax   # %rax = 001122334455FFFF
  movl    $-1,                 %eax  # %rax = 00000000FFFFFFFF ⚠️
  movq    $-1,                 %rax  # %rax = FFFFFFFFFFFFFFFF
  movabsq $0x0011223344556677, %rax  # %rax = 0011223344556677
  movb    $0xAA,               %dl   # %dl  = AA
  movb    %dl,                 %al   # %rax = 00112233445566AA
  movsbq  %dl,                 %rax  # %rax = FFFFFFFFFFFFFFAA ⚠️
  movzbq  %dl,                 %rax  # %rax = 00000000000000AA ⚠️
  ```

### 交换数据

交换数据没有专门的指令，但可以通过一组 `mov` 来实现：

```c
 void swap(long* px, long* py) {
   long tx = *px;
   long ty = *py;
   *px = ty;
   *py = tx;
 }
```

```gas
movq    (%rdi), %rax
movq    (%rsi), %rdx
movq    %rdx, (%rdi)
movq    %rax, (%rsi)
ret
```

### 压栈出栈

x86-64 规定：栈顶字节的地址保存在寄存器 `%rsp` 中，并且小于栈内其他字节的地址。

|   指令    |      含义      |              语义               |
| :-------: | :------------: | :-----------------------------: |
| `pushq s` | PUSH Quad word | `R[%rsp] -= 8; M[R[%rsp]] = s;` |
| `popq d`  | POP quad word  | `d = M[R[%rsp]]; R[%rsp] += 8;` |


## 5 算术及逻辑运算

### 加载有效地址

```gas
leaq source, destination
```

- `lea` 由 ***Load Effective Address（加载有效地址）*** 的首字母构成，相当于 C 语言的 *取地址* 运算（例如 `p = &x[i];`）。

- `source` 只能是 *内存地址*。

- `destination` 只能是 *寄存器*，用于存储 `source` 所表示的 *内存地址*，但不访问该地址。

- 该指令速度极快，常被用来分解算术运算，例如：

  ```c
  long m12(long x) { return x * 12; }
  ```

  通常被编译为

  ```gas
  leaq    (%rdi,%rdi,2), %rax  # 相当于 t = x + 2 * x
  salq    $2, %rax             # 相当于 t <<= 2
  ret
  ```

### 一元运算

唯一的运算对象既是 source 又是 destination：

|  指令   |    含义    |   语义   |
| :-----: | :--------: | :------: |
| `inc d` | INCrement  |  `d++`   |
| `dec d` | DECrement  |  `d--`   |
| `neg d` |   NEGate   | `d = -d` |
| `not d` | complement | `d = ~d` |

### 二元运算

第一个运算对象是 source，第二个运算对象既是 source 又是 destination：

|    指令     |      含义       |   语义   |
| :---------: | :-------------: | :------: |
| `add s, d`  |       ADD       | `d += s` |
| `sub s, d`  |    SUBtract     | `d -= s` |
| `imul s, d` | signed MULtiply | `d *= s` |
| `xor s, d`  |  eXclusive OR   | `d ^= s` |
|  `or s, d`  |   bitwise OR    | `d |= s` |
| `and s, d`  |   bitwise AND   | `d &= s` |

### 移位运算

|    指令    |              含义               |   语义    |    移出的空位    |
| :--------: | :-----------------------------: | :-------: | :--------------: |
| `shl k, d` |   SHift logically to the Left   | `d <<= k` |  在右端，补 `0`  |
| `sal k, d` | Shift Arithmeticly to the Left  | `d <<= k` |  在右端，补 `0`  |
| `shr k, d` |  SHift logically to the Right   | `d >>= k` |  在左端，补 `0`  |
| `sar k, d` | Shift Arithmeticly to the Right | `d >>= k` | 在左端，补符号位 |

其中 `k` 为移动的位数，可以是
- *即时数*，且为 `1` 时可省略（即退化为一元运算）。
- *寄存器*，且只能是 `%cl` 这个特殊的寄存器：
  - 后缀为 `b` 的版本，取 `%cl` 的后 3 位所表示的值，至多移动 7 位。
  - 后缀为 `w` 的版本，取 `%cl` 的后 4 位所表示的值，至多移动 15 位。
  - 后缀为 `l` 的版本，取 `%cl` 的后 5 位所表示的值，至多移动 31 位。
  - 后缀为 `q` 的版本，取 `%cl` 的后 6 位所表示的值，至多移动 63 位。

### 特殊算术运算

x86-64 还提供了一些针对（Intel 称之为 ***八倍词 (oct word)*** 的）128 位整数的算术运算指令：

- 128 位整数用 `R[%rdx]:R[%rax]` 表示，冒号前、后的两个 64 位寄存器分别表示其前、后 64 位。
- 一元乘法：
  - `imulq s` 为带符号乘法，`mulq s` 为无符号乘法。
  - 语义为 `R[%rdx]:R[%rax] = s * R[%rax]`
- 一元除法：
  - `idivq s` 为带符号除法，`divq s` 为无符号除法。
  - 二者均以 `R[%rdx]:R[%rax]` 为 ***被除数 (dividend)***，以 `s` 为 ***除数 (divisor)***，所得的 ***商 (quotient)*** 存入 `%rax`，***余数 (remainder)*** 存入 `%rdx`
  - ⚠️ 不存在“二元除法”指令。
- `cqto` 用于构造带符号除法的被除数：
  - 指令名取自 Convert Quad-word To Oct-word 的首字母。
  - 语义为 `R[%rdx]:R[%rax] = SignExtend(R[%rax])`

### 汇编代码格式

|            |        ATT 格式        |        Intel 格式        |
| :--------: | :--------------------: | :----------------------: |
|   使用者   | GDB, OBJDUMP, CS:APP3e |     Intel, Microsoft     |
|   指令名   |         `movq`         |    去掉 `q`，即 `mov`    |
|  操作对象  |      `movq s, d`       |   逆序，即 `mov d, s`    |
|   即时数   |         `$0x0`         |    去掉 `$`，即 `0x0`    |
|   寄存器   |         `%rsp`         |    去掉 `%`，即 `rsp`    |
|   地址值   |      `D(B, I, S)`      |     `[B + I*S + D]`      |
|  数据长度  |   `movb (%rbx), %al`   | `mov al, BYTE PTR [rbx]` |
|   注释符   |          `#`           |           `;`            |
| 代码块标记 |         `gas`          |          `nasm`          |

由此可见：*Intel 格式* 比 *ATT 格式* 更清晰、易读。除此之外，各种代码高亮插件对 Intel 格式（代码块标记为 `nasm`）的支持也普遍做得更好，因此推荐使用这种格式。

## 6 控制

### 条件码

CPU 用一组名为 ***条件码 (condition code)*** 的寄存器记录最近一条指令的状态。基于这些状态，可以实现 [条件跳转](#跳转指令)、[条件分支](#条件分支)、[循环语句](#循环语句) 等语义。最常用的条件码有如下几个：

| 符号 |     名称      |             含义              |
| :--: | :-----------: | :---------------------------: |
| `CF` |  Carry  Flag  | 最近一条指令触发 *无符号溢出* |
| `ZF` |   Zero Flag   |    最近一条指令获得 *零值*    |
| `SF` |   Sign Flag   |    最近一条指令获得 *负值*    |
| `OF` | Overflow Flag | 最近一条指令触发 *带符号溢出* |

修改条件码可以看做相关指令的副作用。除[算术及逻辑运算](#算术及逻辑运算)指令，以下指令（这里省略[指令后缀](#指令后缀)）也会修改条件码：

- `cmp a, b` 根据 `sub a, b` 指令的结果设置条件码（但不执行 `sub` 指令）。
- `test a, b` 根据 `and a, b` 指令的结果设置条件码（但不执行 `and` 指令）。

⚠️ 一些（不显然的）约定：

- `leaq` 指令不改变条件码。
- 逻辑运算将 `CF` 及 `OF` 设为 `0`。
- 移位运算将 `CF` 设为 *最后一个被移出的位*，而 `OF` 仅在 *移动一位* 时会被修改。
- 自增自减将 `OF` 及 `ZF` 设为 `1`，并保持 `CF` 不变。（原因较复杂）

### 读取条件码

高级语言代码经常将逻辑表达式的结果用整数（`0` 或 `1`）表示，这在汇编代码中是通过 `set_` 系列指令来实现的：


|     指令     |       后缀含义       |          语义          |
| :----------: | :------------------: | :--------------------: |
| `set[e|z] d` |    Equal \| Zero     |        `d = ZF`        |
|   `sets d`   |  Signed (Negative)   |        `d = SF`        |
|   `setg d`   | Greater (signed `>`) | `d = ~(SF ^ OF) & ~ZF` |
|   `setl d`   |  Less (signed `<`)   |     `d = SF ^ OF`      |
|   `seta d`   | Above (unsigned `>`) |    `d = ~CF & ~ZF`     |
|   `setb d`   | Below (unsigned `<`) |        `d = CF`        |

表中 *语义* 一列的关键在于 `setl` 与 `setb` 这两行：

- `setl` 用于 *带符号小于*，有两种情形：
  - `a - b` 未溢出（`OF == 0`）且表示一个负数（`SF == 1`）
  - `a - b` 向下溢出（`OF == 1`）且表示一个正数（`SF == 0`）
- `setb` 用于 *无符号小于*，只有一种情形：
  - `a - b` 向下溢出（`CF == 1`）

表中只列出了几个有代表性的指令，遇到其他指令可以根据以下规则猜出其语义：

- 后缀前端的 `n` 表示 *Not*，后端的 `e` 表示 `or Equal`，因此 `setnle` 就表示 *(set when) Not Less or Equal*，类似的指令可按此规则解读。
- 某些指令具有同义词（例如 `setg` 与 `setnle`），它们具有相同的机器码。编译器、反汇编器在生成汇编代码时，从同义词中任选其一。

以上指令根据（前一条 `cmp` 或 `test` 指令的）比较结果，将单字节的 `d` 设为 `0` 或 `1`；为了获得 32 或 64 位的 `0` 或 `1`，通常需配合 [`movzbq`](#移动数据) 或 [`xorl`](#二元运算) 将更高位清零：

```c
int greater(long x, long y) { return x > y; }
```

```gas
cmpq    %rsi, %rdi
setg    %al         # 将最后一位设为 0 或 1
movzbl  %al, %eax   # 将前 7 位清零
```

```gas
xorl    %eax, %eax  # 将全 8 位清零
cmpq    %rsi, %rdi
setg    %al         # 将最后一位设为 0 或 1
```

### 跳转指令

***跳转 (Jump)*** 指令的基本形式为 `j_ dest`，用于跳转到 `dest` 所表示的指令。其中

- `jmp` 为 *无条件跳转*，对应于 C 语言中 `goto` 语句。
- 其他均为 *有条件跳转*，指令后缀的含义参见上一节。

### 条件分支

C 语言中有三种表达 ***条件分支 (conditional branch)*** 的方式：

```c
/* if-else */
long absdiff(long x, long y) {
  long d;
  if (x > y) d = x-y;
  else       d = y-x;
  return d;
}
/* ?: operator */
long absdiff(long x, long y) {
  long d;
  d = x > y ? x-y : y-x;
  return d;
}
/* goto */
long absdiff(long x, long y) {
  long d;
  if (x <= y) goto Else;
  d = x-y;
  goto Done;
Else:
  d = y-x;
Done:
  return d;
}
```

要得到与源代码结构最接近的汇编代码，需降低优化等级：

```gas
# gcc -Og -S -fno-if-conversion
_absdiff:
        movq    %rdi, %rax  # d = x
        cmpq    %rsi, %rdi
        jle     L2          # x <= y
        subq    %rsi, %rax  # d -= y
        ret
L2:
        subq    %rdi, %rsi  # y -= x
        movq    %rsi, %rax  # d = y
        ret
```

提高优化等级，编译器会在 *两个分支都安全* 且 *计算量都很小* 的情况下，先计算两个分支，再作比较，最后利用 ***条件移动 (conditional move)*** 指令 `cmov_` 完成选择：

```gas
# gcc -O1 -S
_absdiff:
        movq    %rdi, %rdx  # c = x
        subq    %rsi, %rdx  # c -= y
        movq    %rsi, %rax  # d = y
        subq    %rdi, %rax  # d -= x
        cmpq    %rsi, %rdi
        cmovg   %rdx, %rax  # x > y ? d = c : d = d;
        ret
```

### 循环语句

C 语言中有三种（不用 `goto`）表达 ***循环 (loop)*** 的方式：

- `do`-`while` 语句：

  ```c
  /* do-while */
  long pcount(unsigned long x) {
    long count = 0;
    do {
      count += x & 0x1;
      x >>= 1;
    } while (x);
    return count;
  }
  ```

  以 `gcc -Og -S` 编译得如下汇编代码：

  ```gas
  pcount:
          movl    $0, %eax    # count = 0
  L2:  # loop:
          movq    %rdi, %rdx  # t = x
          andl    $1, %edx    # t &= 0x1
          addq    %rdx, %rax  # count += t
          shrq    %rdi        # x >>= 1
       # test:
          jne     L2          # while (x)
       # done:
          ret
  ```

- `while` 语句：

  ```c
  long pcount(unsigned long x) {
    long count = 0;
    while (x) {
      count += x & 0x1;
      x >>= 1;
    }
    return count;
  }
  ```

  以 `gcc -Og -S` 编译得如下汇编代码：

  ```gas
  _pcount:
          movl    $0, %eax    # count = 0
  L2:  # test:
          testq   %rdi, %rdi  # while (x)
          je      L4
       # loop:
          movq    %rdi, %rdx  
          andl    $1, %edx
          addq    %rdx, %rax
          shrq    %rdi
          jmp     L2
  L4:  # done:
          ret
  ```

  此版本在进入循环前先做了一次检测，若恰有 `x == 0` 则会带来性能提升。

- `for` 语句：

  ```c
  #define SIZE 8*sizeof(unsigned long)
  long pcount(unsigned long x) {
    long count = 0;
    for (int i = 0; i != SIZE; ++i) {
      count += (x >> i) & 0x1;
    }
    return count;
  }
  ```

  以 `gcc -O2 -S` 编译得如下汇编代码：

  ```gas
  _pcount:
       # init:
          xorl    %ecx, %ecx  # i = 0
          xorl    %r8d, %r8d  # count = 0
  L2:  # loop:
          movq    %rdi, %rax  # t = x
          shrq    %cl, %rax   # t >>= i
          addl    $1, %ecx    # t &= 0x1
          andl    $1, %eax    # ++i
          addq    %rax, %r8   # count += t
       # test:
          cmpl    $64, %ecx   # i != SIZE
          jne     L2
       # done: 
          movq    %r8, %rax
          ret
  ```

  编译器知道计数器 `i` 的初值与终值，故首次检测被跳过。

### 选择语句

```c
long choose(long x, long y, long z) {
  long w = 1; 
  switch(x) {
    case 1:
      w = y * z;
      break;
    case 2:
      w = y / z;
      /* Fall Through */
    case 3:
      w += z;
      break;
    case 5:
    case 6:
      w -= z;
      break;
    default:
      w = 2;
  }
  return w;
}
```

以 `gcc -Og -S` 编译得如下汇编代码：

```gas
_choose:
        movq    %rdx, %rcx  # z_copy = z
        cmpq    $3, %rdi
        je      L8          # x == 3
        jg      L3          # x > 3
     # x < 3
        cmpq    $1, %rdi
        je      L4          # x == 1
        cmpq    $2, %rdi    # x == 2
        jne     L11
     # case 2:
        movq    %rsi, %rax  # y_copy = y
        cqto                # R[%rdx]:R[%rax] = SignExtend(y_copy)
        idivq   %rcx        # R[%rax] = y_copy / z_copy
        jmp     L2          # Fall into case 3
L11: # default:
        movl    $2, %eax    # w = 2
        ret
L3:  # x > 3
        subq    $5, %rdi    # x -= 5
        cmpq    $1, %rdi
        ja      L12         # x > 1 (case 4, 7, 8, ...)
     # case 6:
        movl    $1, %eax    # w = 1
        subq    %rdx, %rax  # w -= z
        ret
L4:  # case 1:
        movq    %rdx, %rax
        imulq   %rsi, %rax
        ret
L8:  # init:
        movl    $1, %eax    # w = 1
L2:  # case 3:
        addq    %rcx, %rax  # w += z_copy
        ret
L12: # default:
        movl    $2, %eax    # w = 2
        ret
```

- 教材中的示例采用了两次跳转：先跳进表中读取标签，再跳到标签所指位置。
- 编译器可能会打乱各种情形的顺序。
- 若分支总数 $N$ 较大，则用 `switch` 可以减少判断次数：
  - 若情形分布较密集，则编译器会为每一种情形各生成一个标签，故 `switch` 语句只需要 $\Theta(1) $ 次判断。
  - 若情形分布较稀疏，则编译器会生成一棵平衡搜索树，故 `switch` 语句至多需要 $\Theta(\log N)$ 次判断。
  - 与之等价的 `if`-`else` 语句可能（例如 `default` 情形）需要 $\Theta(N)$ 次判断。

## 7 函数

***函数 (function)*** 又称 ***过程 (procedure)***、***方法 (method)***、***子例程 (subroutine)***、***句柄 (handler)***，是模块化编程的基础：每个函数都是一个生产或加工数据的功能模块。几乎所有高级语言都提供了这种机制，并且各种语言用于定义函数的语法都大同小异。这是因为它们几乎都采用了同一种 *机器级实现*，后者正是本节的内容。

它是软件功能的最小单位，因此是模块化编程的基础。

### 运行期栈

若函数 `Q` 被函数 `P` 调用，则 `P` 与 `Q` 分别被称为 ***主调者 (caller)*** 与 ***被调者(callee)***。函数调用正是通过 ***控制 (control)*** 及 ***数据 (data)*** 在二者之间相互 ***传递 (pass)*** 来实现的：

- [**传递控制**](#传递控制)：Caller 利用 `call` 指令将控制转移给 Callee；Callee 运行结束后，利用 `ret` 指令将控制交还给 Caller。
- [**传递数据**](#传递数据)：Caller 将第一个整型输入值存入 `%rdi`、将第二个整型输入值存入 `%rsi`、……，供 Callee 读取；Callee 将整型返回值存入 `%rax` 中，供 Caller 读取。
- [**局部存储**](#局部存储)：在 Callee 运行的那段时间，Caller 处于冻结状态，其状态（局部变量的值、下一条指令的地址、……）被保存在寄存器或内存中。

尽管某些局部变量可以只在寄存器中度过其生存期，但与内存相比，寄存器所能容纳的数据量非常有限，因此后者才是更一般的局部存储机制。

每个函数在内存中都拥有一段被称为 ***帧 (frame)*** 的连续地址空间，其分配与释放遵循 ***栈 (stack)*** 的 ***后进先出 (LIFO)*** 规则，因此这种内存管理机制（或这段内存空间）很自然地被称为 ***运行期栈 (run-time stack)***。栈顶地址

- 始于 `0x7FFFFFFFFFFF` 并随着栈内数据的增加而减小，因此这是一个“向下生长”的栈。
- 保存在 `%rsp` 中（注意与 `%rip` 区分，后者为 *即将被执行的那条指令的地址*）。

### 传递控制

假设有如下两个函数：

```c
long mult2(long a, long b) {
  long s = a * b;
  return s;
}
void multstore(long x, long y, long *dest) {
  long t = mult2(x, y);
  *dest = t;
}
```

编译（所得可执行文件的反汇编）结果为

```gas
0000000100000f5c _mult2:
100000f5c: 48 89 f8                     movq    %rdi, %rax
100000f5f: 48 0f af c6                  imulq   %rsi, %rax
100000f63: c3                           retq

0000000100000f64 _multstore:
100000f64: 53                           pushq   %rbx
100000f65: 48 89 d3                     movq    %rdx, %rbx
100000f68: e8 ef ff ff ff               callq   -17 <_mult2>
100000f6d: 48 89 03                     movq    %rax, (%rbx)
100000f70: 5b                           popq    %rbx
100000f71: c3                           retq 
```

其中

- `call` 指令依次完成以下两步：
  - 将下一条指令的地址（此处为 `0x100000f6d`）压入运行期栈。
  - 将 `%rip` 设为被调函数的首地址（此处为 `0x100000f5c`，它与返回地址 `0x100000f6d` 相距 `-17` 即 `-0x11`），即向被调函数移交控制权。
- `ret` 指令依次完成以下两步：
  - 从运行期栈弹出返回地址（此处为 `0x100000f6d`）。
  - 将 `%rip` 设为上述返回地址，即向主调函数交还控制权。

### 传递数据

- 整型（含指针型）数据：
  - 整型返回值通过 `%rax` 传递。
  - 前六个整型实参通过寄存器（直接）传递，对应关系参见《[整型寄存器](#整型寄存器)》。
  - 更多的整型实参通过 Caller 的帧（间接）传递。
- 浮点型数据：
  - 前几个浮点型实参通过浮点型寄存器（直接）传递，原理与整型数据类似。
  - 更多的浮点型实参通过 Caller 的帧（间接）传递。
- [数组](#数组)：
  - 通过传地址实现。

### 局部存储

函数可以在自己的栈内保存以下数据：

- 局部变量：寄存器无法容纳的局部变量，以及[数组](#数组)、[异质数据结构](#异质数据结构)。
- 返回地址：见[传递控制](#传递控制)。
- 寄存器值：
  - 被调函数保存的寄存器：一个函数在使用此类寄存器前，需将它们的值存储到自己的帧内；在移交控制权前，需将这些寄存器恢复为使用前的状态。[整型寄存器](#整型寄存器)中的 `%rbx`、`%rbp`、`%r12`、`%r13`、`%r14`、`%r15` 均属于此类。
  - 主调函数保存的寄存器：一个函数在调用其他函数（含递归调用自身）前，需将（自己用到的）此类寄存器的值存储到自己的帧内。用于[传递数据](#传递数据)的寄存器都属于这一类；完整列表见《[整型寄存器](#整型寄存器)》。

参考以下示例：

```c
long incr(long *p, long val) {
  long x = *p;
  long y = x + val;
  *p = y;
  return x;
}
```

```gas
_incr:
        movq    (%rdi), %rax  # incr 的局部变量在寄存器内度过其生存期：
        addq    %rax, %rsi    # x 位于 %rax 中，y 位于 %rsi 中。
        movq    %rsi, (%rdi)
        ret
```

```c
long call_incr() {
  long v1 = 15213;
  long v2 = incr(&v1, 3000);
  return v1+v2;
}
```

```gas
_call_incr:
        subq    $24, %rsp        # 分配 call_incr 的帧
        movq    $15213, 8(%rsp)  # 将局部变量 v1 存储到帧内
        leaq    8(%rsp), %rdi    # 构造传给 incr 的第一个实参
        movl    $3000, %esi      # 构造传给 incr 的第二个实参
        call    _incr						 # 返回时 %rax 存储了 v2 的值
        addq    8(%rsp), %rax    # v2 += v1
        addq    $24, %rsp        # 释放 call_incr 的帧
        ret
```

```c
long call_incr2(long x) {
  long v1 = 15213;
  long v2 = incr(&v1, 3000);
  return x+v2;
}
int main(int argc, char* argv[]) {
  call_incr2(argc);
  return 0;
}
```

```gas
_call_incr2:
        pushq   %rbx             # call_incr2 是 main 的被调函数
        subq    $16, %rsp        #
        movq    %rdi, %rbx       # 将 call_incr2 的第一个实参存入 %rdx
        movq    $15213, 8(%rsp)  #
        leaq    8(%rsp), %rdi    # 将 incr 的第一个实参存入 %rdi
        movl    $3000, %esi      #
        call    _incr            #
        addq    %rbx, %rax       # v2 += x
        addq    $16, %rsp        #
        popq    %rbx             # 还原 %rbx 的值
        ret
```

### 递归函数

```c
#include <stdlib.h>
#include <stdio.h>
unsigned long factorial(unsigned n) {
  return n <= 1 ? 1 : n * factorial(n-1);
}
int main(int argc, char* argv[]) {
  unsigned int n = atoi(argv[1]);
  printf("%d! == %ld\n", n, factorial(n));
}
```

```gas
_factorial:
        cmpl    $1, %edi
        ja      L8
        movl    $1, %eax
        ret
L8:
        pushq   %rbx
        movl    %edi, %ebx
        subl    $1, %edi    # n - 1
        call    _factorial  # f(n-1)
        imulq   %rbx, %rax  # n * f(n-1)
        popq    %rbx
        ret
```



## 8 数组

### 基本原则

***数组 (array)*** 是最简单的 ***聚合 (aggregate)*** 类型：

- 它是以同种类型的对象为成员的容器，因此是一种 ***均质的 (homogeneous)*** 数据类型。
- 所有成员在（虚拟）内存中连续分布。
- 通过 ***指标 (index)*** 访问每个成员的时间大致相同。

几乎所有高级编程语言都有数组类型，其中以 C 语言的数组语法（如[指针算术](#指针算术)）最能体现数组的机器级表示。在 C 代码中，以  `a`  为变量名、含 `N` 个 `T` 型（可以是复合类型）对象的数组（通常）以如下方式声明：

```c
T a[N]
```

- 该声明既可以单独作为一条语句（以 `;` 结尾），又可以出现在函数形参列表中。
- 数组 `a` 的类型为 `T[N]`，成员个数 `N` 也是数组类型的一部分。
- 每个成员的大小为 `sizeof(T)` 字节，故整个数组的大小为 `N * sizeof(T)` 字节。
- 数组名 `a` 可以当做指向 `a[0]` 的指针使用，这是 `T[N]` 到 `T*` 的隐式类型转换。

### 指针算术

|      声明       |     `An`     | `*An` 或 `An[0]` | `**An` 或 `An[0][0]` |
| :-------------: | :----------: | :--------------: | :------------------: |
|  `char A1[9]`   |  `char[9]`   |      `char`      |        不合法        |
|  `char* A2[9]`  | `(char*)[9]` |     `char*`      |        `char`        |
| `char (*A3)[9]` | `char(*)[9]` |    `char[9]`     |        `char`        |
| `char* (A4[9])` | `(char*)[9]` |     `char*`      |        `char`        |
|   `char** A5`   |   `char**`   |     `char*`      |        `char`        |

### 多维数组

在 C 代码中，含 `R` 行 `C` 列（共 `R * C` 个成员）的二维数组（通常）以如下方式声明：

```c
T a[R][C]
```

- 整个数组的大小为 `R * C * sizeof(T)` 字节。

- 成员排列遵循 ***行优先 (row-major)*** 规则：
  - 每一行的 `C` 个成员连续分布，即 `a[i][j]` 位于 `a[i][j-1]` 与 `a[i][j+1]` 之间。
  - 第 `i` 行的 `C` 个成员位于 `a[i-1][C-1]` 与 `a[i+1][0]` 之间。
  
- 按上述规则，类型为 `T[R][C]` 的二维数组可以被看作以类型为 `T[C]` 的一维数组为成员的一维数组，即
  ```c
  typedef T Row[C];
  Row a[R];  /* 等价于 T a[R][C] */
  ```
  
- 更一般的：类型为 `T[D1][D2]⋯[Dn]` 的 `n` 维数组可以被看作以 `T[D2]⋯[Dn]` 型 `n-1` 维数组为成员的 `1` 维数组。

### 长度固定的数组

在 ISO C99 引入[长度可变的数组](#长度可变的数组)之前，用于声明多维数组 `T[D1][D2]⋯[Dn]` 的整数（除 `D1` 可以省略外）必须在编译期确定。

```c
#define N 4
long get_element(long a[N][N], long i, long j) {
  return a[i][j];
}
```

```gas
_get_element:  # R[%rdi] = a, R[%rsi] = i, R[%rdx] = j
        salq    $5, %rsi             # R[%rsi] = 8 * N * i
        addq    %rsi, %rdi           # R[%rdi] = a + 8*N*i
        movq    (%rdi,%rdx,8), %rax  # M[a + 8*N*i + 8*j]
        ret
```

其中 `8 * N` 的值可在编译期确定为 `32`，因此（即使优化等级很低，也）可以被优化为（比乘法更快的）移位运算。

### 长度可变的数组

ISO C99 引入了长度可变的数组。

```c
long get_element(long n,/* 必须紧跟在 n 之后 */long a[n][n],
                 long i, long j) {
  return a[i][j];
}
```

```gas
_get_element:  # R[%rdi] = n, R[%rsi] = a, R[%rdx] = i, R[%rcx] = j
        imulq   %rdi, %rdx           # R[%rdx] = n * i
        leaq    (%rsi,%rdx,8), %rax  # R[%rax] = a + 8*n*i
        movl    (%rax,%rcx,8), %rax  # R[%rax] = M[a + 8*n*i + 8*j]
        ret
```

其中 `8 * n` 的值无法在编译期确定，故无法像[长度固定的数组](#长度固定的数组)那样用移位运算代替。

## 9 异质数据结构

### `struct`

***结构体 (struct)*** 是一种 ***异质的 (heterogeneous)*** 数据类型：

- 它（通常）是以不同类型的对象为成员的容器。
- 各成员在（虚拟）内存中按声明的顺序分布，但不一定连续分布。
- 通过 *名称 (name)* 访问每个成员的时间大致相同。

在 C 代码中，结构体用 `struct` 关键词来定义：

```c
struct node_t {
  int a[4];
  size_t i;
  struct node_t *next;
};
```

此 `struct` 含三个成员（类型为 `int[4]` 的数组、类型为 `size_t` 的整数、类型为 `struct node_t *` 的指针），各成员在内存中的分布如下：

```
| a[0]  | a[1]  | a[2]  | a[3]  |       i       |     next      |
^       ^       ^       ^       ^               ^               ^
0       4       8       12      16              24              32
```

编译所得的汇编代码中看不到成员名称，访问成员的操作全部被翻译为偏移量：

```c
void set_val(struct node_t *node, int val) {
  while (node) {
    int i = node->i;
    node->a[i] = val;
    node = node->next;
  }
}
```

```gas
_set_val:  # R[%rdi] = node, R[%rsi] = val
L2:  # loop:
        testq   %rdi, %rdi           # node == 0?
        je      L4
        movslq  16(%rdi), %rax       # i = node->i
        movl    %esi, (%rdi,%rax,4)  # node->a[i] = val
        movq    24(%rdi), %rdi       # node = node->next
        jmp     L2
L4:  # node == 0
        ret
```

以上“成员无缝分布”的 `struct` 并不是普遍的，更一般的 `struct` 需按[数据对齐](#数据对齐)规则安排成员。

### `union`

C 语言中的 `union` 与 `struct` 有相同的 ***语法 (syntax)***，但有不同的 ***语义 (semantics)*** 及相应的机器级表示：

- `union` 的所有成员共享同一段内存空间。
- 整个 `union` 的长度不小于最大成员的长度。

⚠️ 该机制弱化了编译器的类型检查功能，很容易导致错误，应尽量少用。

`union` 可用于定义相似的数据类型。若某种二叉树的

- 叶结点（只）含两个 `double` 成员
- 非叶结点（只）含两个指针成员

则结点类型有两种典型的定义方式：

```c
struct node_s {
  struct {
    struct node_s *left;
    struct node_s *right;
  } children;
  double data[2];
};
union node_u {
  struct {
    union node_u *left;
    union node_u *right;
  } children;
  double data[2];
};
```
前者需要 32 字节，后者只需要 16 字节；但后者无法通过成员判断其是否为叶结点（前者可由 *`left` 与 `right` 是否为空* 判断），为此需引入额外的成员：
```c
typedef enum { N_LEAF, N_INTERNAL } nodetype_t;
struct node_t {
  union {
    struct {
      struct node_t *left;
      struct node_t *right;
    } internal;
    double data[2];
  } info;
  nodetype_t type;
};
```
该方案每个结点的大小为 `16 + 4 + 4 == 24` 字节，其中最后 `4` 个字节不存储数据，仅用于[数据对齐](#数据对齐)。

`union` 的另一个用处是获取其他类型的字节表示：

```c
#include <stdlib.h>
#include <stdio.h>
unsigned long double2bits(double d) {
  union {
    double d;
    unsigned long u;
  } temp;
  temp.d = d;
  return temp.u;
}
int main(int argc, char* argv[]) {
  double x = atof(argv[1]);
  printf("%g's byte representation is\n", x);
  unsigned long l = double2bits(x);
  int shift = 64;
  for (int i = 0; i != sizeof(unsigned long); ++i) {
    // for each byte:
    printf(" ");
    for (int j = 0; j != 8; ++j) {
      printf("%ld", l >> (--shift) & 1);
    }
  }
  assert(shift == 0);
  printf("\n");
}
```

这种用法可以被用来检测系统的类型：

- 若 ***最重要的 (most significant)*** 字节的地址最小，则为 ***大端 (big endian)*** 系统。
- 若 ***最次要的 (least significant)*** 字节的地址最小，则为 ***小端 (little endian)*** 系统。

```c
#include <stdio.h>
int main() {
  union {
    unsigned char c[8];
    unsigned short s[4];
    unsigned int i[2];
    unsigned long l[1];
  } x;
  for (int j = 0; j < 8; j++) {
    x.c[j] = 0xf0 + j;
  }
  printf("chars 0-7 == [0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x, 0x%x]\n",
         x.c[0], x.c[1], x.c[2], x.c[3], x.c[4], x.c[5], x.c[6], x.c[7]);
  printf("shorts 0-3 == [0x%x, 0x%x, 0x%x, 0x%x]\n",
         x.s[0], x.s[1], x.s[2], x.s[3]);
  printf("ints 0-1 == [0x%x, 0x%x]\n", x.i[0], x.i[1]);
  printf("long 0 == [0x%lx]\n", x.l[0]);
}
```

```shell
# on big endian system:
 char[0, 8) == [0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7]
short[0, 4) == [0xf0f1, 0xf2f3, 0xf4f5, 0xf6f7]
  int[0, 2) == [0xf0f1f2f3, 0xf4f5f6f7]
 long[0, 1) == [0xf0f1f2f3f4f5f6f7]
# on little endian system:
 char[0, 8) == [0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7]
short[0, 4) == [0xf1f0, 0xf3f2, 0xf5f4, 0xf7f6]
  int[0, 2) == [0xf3f2f1f0, 0xf7f6f5f4]
 long[0, 1) == [0xf7f6f5f4f3f2f1f0]
```

### 数据对齐

一般的 `struct` 按以下规则 ***对齐 (align)*** 成员：

- 各成员按声明的顺序分布。
- 大小为 `K` 字节的初等（非聚合类型）成员，其 *首地址* 必须是 `K` 的整数倍。
- 整个 `struct` 的 *首地址* 及 *长度* 必须是其 *最大初等成员* 大小的整数倍。

因此整个 `struct` 的大小可能大于其全部成员大小之和。例如：

```c
struct X1 {
  short s;
  int i[2];
  double d;
};
```

各成员在内存中的分布为：

```
|s  |###|i[0]   |i[1]   |#######|d              |
 ^   ^   ^       ^       ^       ^               ^
+0  +2  +4      +8      +12     +16             +24
```

其中 `[2,4)` 与 `[12,16)` 这两段内存不存储数据，而只是用来满足 `i[0]` 与 `d` 的对齐要求。重新安排成员顺序：

```c
struct X2 {
  double d;
  int i[2];
  short s;
};
```

则各成员在内存中的分布为：

```
|d              |i[0]   |i[1]   |s  |###########|
 ^               ^       ^       ^   ^           ^
+0              +8      +12     +16 +18         +24
```

其中 `[18,24)` 不存储数据，只是用来满足整个 `struct` 的对齐要求。

为节省存储空间，可采取 *贪心 (greedy)* 策略，即 *越长的成员越靠前声明*。例如：

```c
struct Y1 {
  short s1;
  double d;
  short s2;
};
struct Y2 {
  double d;
  short s1;
  short s2;
};
```

二者的成员在内存中的分布分别为：

```
|s1 |###########|d              |s2 |###########|
 ^   ^           ^               ^   ^           ^
+0  +2          +8              +16 +18         +24

|d              |s1 |s2 |#######|
 ^               ^   ^   ^       ^
+0              +8  +10 +12     +16
```

## 10 结合控制与数据

### 理解指针

C 语言的 ***指针 (pointer)*** 是对（虚拟）内存读写操作的一种抽象。

- 每个指针都有一个与之关联的类型，表示所指对象的类型。
  - 类型为 `T*` 的指针 指向 类型为 `T` 的对象。
  - 类型为 `void*` 的指针可以指向任意类型的对象，使用前必须被显式或隐式地 ***转换 (cast)*** 为具体类型。
- 每个指针都有一个 ***值 (value)***，表示所指对象在（虚拟）内存中的位置（地址）。
  - 未初始化的指针可能有任意值。
  - 值为 `NULL` 或 `0` 的指针不指向任何对象。
- 取地址运算符 `&` 作用在 ***左值表达式 (lvalue expression)*** 上，得到一个地址值。
  - *左值表达式* 是指可以出现在 *赋值运算符* 左侧的表达式。
  - 在相应的汇编代码中，表现为 `lea` 指令。
- 解引用运算符 `*` 作用在指针上，得到所指对象的一个 ***左值引用 (lvalue reference)***。
  - 在相应的汇编代码中，表现为内存型[运算对象](#运算对象)。
- [数组](#数组)与指针关系紧密。
  - 数组名可以被当作指针使用。
  - 数组表达式 `a[i]` 与指针表达式 `*(a+i)` 完全等价。
- 对一个指针进行（显式或隐式）类型转换，只改变它的类型，不改变它的值。
  - 在相应的汇编代码中，没有 *指针类型转换指令*，但可以从指针加减法的放大倍数上看出影响。
- 指针可以指向[函数](#函数)。
  - 函数类型由其形参类型及返回类型决定。
  - 函数名可以用来给这种类型的指针赋值。

### 使用 GDB 调试器

见《[断点调试](../debug/README.md)》。

### 越界访问与缓冲区溢出

C 标准库的[文件读写](https://en.cppreference.com/w/c/io)及[字符串处理](https://en.cppreference.com/w/c/string/byte)模块提供了一些不安全的字符串接口（函数或预定义宏）：

```c
#include <stdio.h>
char* gets(char* dest)
#include <string.h>
char* stpcpy(char* dest, const char* src);
char* strcat(char* s1, const char* s2);
int sprintf(char* dest, const char* format/* 含 %s */, ...);
```

这些接口不检查（`dest` 或 `s1` 所指的）目标空间是否足够大，使用不当会造成缓冲区溢出，曾经是系统安全漏洞的一大来源。

例如以下从 `stdin` 读取任意长度字符串的接口：

```c
char *gets(char *dest) { /* Declared in header <stdio.h> */
  int c;
  char *curr = dest;
  while ((c = getchar()) != '\n' && c != EOF)
    *curr++ = c;
  if (c == EOF && curr == dest) /* No characters read */
    return NULL;
  *curr++ = '\0'; /* Terminate string */
  return s;
}
```

典型使用场景：

```c
void echo() { /* Read input line and write it back */
  char buffer[8]; /* Way too small! */
  gets(buffer);
  puts(buffer);
}
```

编译时加上 `-fno-stack-protector` 选项，得以下汇编码：

```gas
echo:
  subq	$24, %rsp
  movq	%rsp, %rdi
  call	gets
  movq	%rsp, %rdi
  call	puts
  addq	$24, %rsp
  ret
```

- 第一行在[运行期栈](#运行期栈)内为函数 `echo` 分配了长度为 24 字节的帧。执行这条指令后，函数 `echo` 的返回地址位于 `R[rsp]+24` 为首的 8 字节中。
- 第二、三行说明 `buffer` 位于 `R[rsp]` 到 `R[rsp]+7` 这 8 字节中，而 `R[rsp]+8` 到 `R[rsp]+23` 这段空间没有被用到。
- `gets` 无法判断输入是否过长，有可能发生越界：
  - 若有效输入不多于 7 个字符，则一切正常。
  - 若有效输入介于 8 到 23 个字符之间，则 `echo` 帧内的剩余空间会被污染，但 `echo` 的返回地址未被修改，故 `echo` 还能正确返回。
  - 若有效输入不少于 24 个字符，则 `echo` 的返回地址被破坏，故 `echo` 无法正确返回。

👉 使用含长度检查的接口可避免上述缓冲区溢出：
```c
#include <stdio.h>
char* fgets(char* dest, int n, FILE* stream);
#include <string.h>
char* stpncpy(char* dest, const char* src, size_t n);
char* strncat(char* s1, const char* s2, size_t n);
int snprintf(char* dest, size_t n, const char* format/* 含 %s */, ...);
```

### 阻断缓冲区溢出攻击

缓冲区溢出可能被用来实施攻击：

- 以字符串形式输入一段可执行程序（例如启动命令行终端）的编码。
- 利用缓冲区溢出，将原来的返回地址篡改为上述代码段的起始地址。

#### 地址空间布局随机化

攻击者要成功篡改返回地址，需掌握目标系统的运行期栈信息 —— 这在早期并不困难，攻击者只要能获得目标系统的软件版本，就能在本地推算出这一信息。

***栈随机化 (stack randomization)*** 技术（在创建新进程时）令运行期栈的起始位置随机变化，因此可以在一定程度上避免上述漏洞。现代 Linux 默认采用这项技术，在 Linux 系统上运行如下程序即可看出效果：

```c
/* stack_demo.c */
int main() {
  long local;
  printf("local at %p\n", &local);
  return 0;
}
```

```shell
$ ./stack_demo
local at 0x7ffee86e4820
$ ./stack_demo
local at 0x7ffee22bd820
$ ./stack_demo
local at 0x7ffeeda9f820
```

更一般的，有 ***地址空间布局随机化 (address-space layout randomization, ASLR)*** 技术 —— 它为整个地址空间的所有组成片段（程序代码、库代码、运行期栈、全局变量、堆）都引入了随机性，从而进一步增加了攻击者推算出所需地址的难度。

然而，上述技术并不能彻底解决缓冲区溢出攻击，***`nop` 雪橇*** 就是一种常见的破解方案：

- 在攻击代码前插入一段 `nop` 序列，即以 `nop` 指令（意为 No OPeration）的编码（例如 `0x90`）不断重复而构成的序列。
- 只要（被篡改的）返回地址指向这段 `nop` 序列中的任何一个 `nop` 指令，则程序控制权将 ***滑行 (slide)*** 至攻击代码。
- 假设运行期栈的起始地址有 $2^{N}$ 种概率相同的情形，取 `nop` 序列的长度为 $2^{L}$ 字节（单次攻击覆盖 $2^{L}$ 种情形），则每 $2^{N-L}$ 次攻击即可期望命中 $1$ 次。

#### 栈保护器（金丝雀）

该机制得名于旧时煤矿利用 ***金丝雀 (canary)*** 检测巷道内的危险气体是否超标。

仍以函数 `echo()` 为例，编译时不加 `-fno-stack-protector` 选项，得以下汇编码：

```gas
echo:
  subq	$24, %rsp
  movq  %fs:40, %rax  # 从内存中取出 canary
  movq  %rax, 8(%rsp) # 存放在 buffer 后面
  xorl  %eax, %eax
  movq	%rsp, %rdi
  call	gets
  movq	%rsp, %rdi
  call	puts
  movq  8(%rsp), %rax # 将 buffer 后的 canary 取出
  xorq  %fs:40, %rax  # 与内存中的原始值作比较
  je    .L9
  call  __stack_chk_fail
.L9:
  addq	$24, %rsp
  ret
```

其中 `%fs:40` 可以理解为从内存中读取的随机值（很难被攻击者猜到）。该机制只引入了一点点（读取、比较 canary 的）性能开销，便（几乎）能阻断所有缓冲区溢出攻击。

#### 限制可执行代码的地址范围

虚拟内存空间在逻辑上被划分为若干 ***页 (page)***，每页含 2048 或 4096 字节。多数系统支持为各页赋予不同的访问权限：
- ***可读 (Readable)***
- ***可写 (Writable)***
- ***可执行 (eXecutable)***
- 旧时的 x86 架构将 R 与 X 用同一个标志位（类似于[条件码](#条件码)）表示，故无法区分一段字节是可读数据还是可执行代码。
- 现代的 64 处理器均引入了名为 `NX` 的标志位，用于表示当前内存页 ***不可执行 (Not eXecutable, NX)***。只要栈空间所在的页被标记为 *不可执行*，通过缓冲区溢出植入的攻击代码便无法被执行。

### 支持长度可变的帧

使用库函数 `void* alloca(size_t n)` 或 [长度可变的数组](#长度可变的数组) 都有可能使所在帧的长度可变（无法在编译期确定）。

```c
long vframe(long n, long idx, long *q) {
  long i;
  long *p[n];
  p[0] = &i;
  for (i = 1; i < n; i++)
    p[i] = q;
  return *p[idx];
}
```

其中

- 局部变量 `i` 要有地址，故必须存储在内存（栈）中。
- 局部数组 `p` 的长度 `n` 到运行期才能确定，故其所在帧的长度可变。

长度可变的帧依然遵循[运行期栈](#运行期栈)的一般规则：
- 栈顶地址存储在 `%rsp` 中，且随着栈内数据的增加而减小。
- 借助 `sub` 指令，将 `%rsp` 减小为新值，实现帧的分配。
- 借助 `mov` 指令，将 `%rsp` 重置为旧值，实现帧的释放。

故函数 `vframe()` 的入口与出口分别被编译为

```gas
pushq %rbp # 使用 R[rbp] 前需先保存其旧值
movq %rsp, %rbp # 用 R[rbp] 保存 R[rsp] 的旧值
# ...
leave # 恢复 R[rbp] 及 R[rsp]
ret
```

其中 `leave` 是以下两条指令的简写：

```gas
movq %rbp, %rsp # 释放当前帧
popq %rbp # 恢复 R[rbp] 的旧值
```

紧接在入口之后的是分配局部变量 `i` 及数组 `p` 的指令：

```gas
subq $16, %rsp  # 为 i 分配空间，不小于 8 字节
leaq 22(,%rdi,8), %rax # 长度至少为 8 * n 字节
andq $-16, %rax # 用 0xfff0 将 R[rax] 过滤为 16 的整数倍
subq %rax, %rsp # 为 p 分配空间
```

其中 `R[rax] == 8 * n + (n % 2 ? 8 : 16)` 不小于 `8 * n` 且为 `16` 的整数倍。

紧随其后的是循环初始化指令：

```gas
leaq 7(%rsp), %rax   # R[rax] = R[rsp] + 7
shrq $3, %rax        # R[rax] /= 8
leaq 0(,%rax,8), %r8 # R[r8 ] = &p[0]
movq %r8, %rcx       # R[rcx] = &p[0]
```

此时 `R[r8]` 及 `R[rcp]` 均存储了数组 `p` 的首地址，其值不小于 `R[rsp]` 且为 `8` 的整数倍 —— 这体现了（长度为 `8` 字节的）指针型数组成员的[数据对齐](#数据对齐)规则。


## 11 浮点代码
