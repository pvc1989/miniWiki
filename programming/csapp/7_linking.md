---
title: 链接
---

# 1. 编译器驱动器

【编译器驱动器 (compiler driver)】`gcc -v -Og -o prog main.c sum.c`

- 【预处理器 (preprocessor)】`cpp [other args] main.c /tmp/main.i`
- 【编译器 (compiler)】`cc1 /tmp/main.i -Og [other args] -o /tmp/main.s`
- 【汇编器 (assembler)】`as [other args] -o /tmp/main.o /tmp/main.s`
- 【链接器 (linker)】`ld -o prog [system object files and args] /tmp/main.o /tmp/sum.o`

【加载器 (loader)】`./prog`

- 先将可执行文件中的代码及数据复制到内存中。
- 再将控制权转移到该程序 `main()` 头部。

![](https://csapp.cs.cmu.edu/3e/ics3/link/linker.pdf)

# 2. 静态链接

【静态链接器 (static linker)】

- 【符号解析 (symbol resolution)】
  - 【符号】函数、全局变量、静态变量
  - 【解析】为每个符号“引用 (symbol reference)”关联唯一的符号“定义 (definition)”
- 【[搬迁 (relocation)](#relocate)】为每个符号定义关联唯一的内存地址，将所有对该符号的引用修改为此地址。

![](https://csapp.cs.cmu.edu/3e/ics3/link/staticlibs.pdf)

# 3. 目标文件

【目标文件 (object file)】存储在硬盘上的文件，内部含若干连续字节区块。

- 【可搬迁的 (relocatable)】
- 【可执行的 (executable)】
- 【共享的 (shared)】

【目标文件格式】

- 【Unix】`a.out` format
- 【Windows】Portable Executable (PE) format
- 【Mac OS X】Mach-O format
- 【现代 x86-64 Linux/Unix】Executable and Linkable Format (ELF)

# 4. 可搬迁目标文件

![](https://csapp.cs.cmu.edu/3e/ics3/link/elfrelo.pdf)

- 【ELF header】词长 (word size)、字节顺序 (byte ordering)、ELF header 长度、目标文件类型、机器类型、section header table 的位置及所含 sections 的数量。
- 【`.text`】可执行程序的机器码
- 【`.rodata`】只读数据（字符串常量、`switch` 跳转表）
- 【`.data`】初值非零的全局变量
- 【`.bss`】未初始化的静态变量或初值为零的全局或静态变量（不占实际空间）
  - Block Started by Symbol
  - Better Save Space
- 【`.symtab`】函数、全局变量的符号列表，不含局部变量
- 【`.rel.text`】`.text` 中有待链接器修改的位置（外部函数、全局变量）的列表
- 【`.rel.data`】初值（如：外部函数或全局变量的地址）待定的全局变量
- 【`.debug`】调试信息（编译时开启 `-g` 才有）
- 【`.line`】源代码位置与指令位置的映射（编译时开启 `-g` 才有）
- 【`.strtab`】用于 `.symtab` 及 `.debug` 中符号表的字符串列表
- 【section header table】除以上 sections 外，（可搬迁的目标文件）还有三个预留 sections：
  - 【`ABS`】本地全局符号
  - 【`UNDEF`】外部全局符号
  - 【`COMMON`】未初始化的全局变量（区别于 `.bss`），即[弱符号](#weak-symbol)

# 5. 符号、符号列表

对于链接器，有三种符号：

- 当前模块定义的全局符号（non`static` 函数、全局变量）
- 其他模块定义的全局符号（non`static` 函数、全局变量）
- 只在当前模块内出现的局部符号（`static` 函数、`static` 变量）

```c
// 不同函数中的局部 `static` 变量对应不同链接符号
int f() { static int x = 0; return x; }
int g() { static int x = 1; return x; }
```

符号列表由汇编器创建，各条目格式如下，可以用 GNU `readref` 程序查看：

```c
typedef struct {
  int name;      /* String table offset */
  char type:4,   /* Function or data (4 bits) */
    binding:4;   /* Local or global (4 bits) */
  char reserved; /* Unused */
  short section; /* Section header index */
  long value;    /* Section offset or absolute address */
  long size;     /* Object size in bytes */
} Elf64_Symbol;
```

# 6. 符号解析

全局符号解析的一般规则：

- 若当前模块内没有定义，则假设其定义在其他模块中。
- 若多个模块有重复定义，则可能报错或从中选择一个。

## 6.1. 同名符号的解析

- 【强符号 (strong symbol)】函数、有初值的全局变量
- 【弱符号 (weak symbol)】无初值的全局变量<a href id="weak-symbol"></a>

⚠️ C++ 中类的重载方法有不同的符号名，例如 `Foo::bar(int, long)` 的符号名为 `bar__3Fooil`。

链接规则：

1. 强符号不能同名。
2. 一强多弱，则取一强。
3. 零强多弱，任取一若。

## 6.2. 静态库的链接

【静态库 (static library)】打包一组目标文件所得的输出文件。

- 【Linux】文件名后缀为 `.a`，意为 **a**rchive

![](https://csapp.cs.cmu.edu/3e/ics3/link/staticlibs.pdf)

链接规则：创建可执行文件时，从静态库中复制被引用的目标代码区块。

```shell
gcc -c addvec.c multvec.c main2.c
ar rcs libvector.a addvec.o multvec.o
gcc -static -o prog2c main2.o ./libvector.a
# 或等价的
gcc -static -o prog2c main2.o -L. -lvector
```

其中 `-static` 表示生成“完全链接的 (fully linked)”可执行文件。

## 6.3. 用静态库解析引用

自左向右扫描命令行参数列表中的可搬迁目标文件或静态库：

- 若遇到可[搬迁](#relocate)目标文件，则将其放入待[搬迁](#relocate)文件列表 $E$，并更新未定义符号列表 $U$ 及已定义符号列表 $D$。
- 若遇到静态库，则遍历其中的成员（可[搬迁](#relocate)目标文件）。若某成员可以解析 $U$ 中的成员（未定义符号），则将其放入 $E$，并更新 $U$ 及 $D$。
- 最后，若 $U$ 为空，则合并及[搬迁](#relocate) $E$ 中的文件以生成可执行文件；否则，链接器报错并结束运行。

⚠️ 链接是否成功及输出结果，依赖于各文件在命令行参数列表中的顺序。

```shell
# foo.c 调用 libx.a 中的函数 f，f 调用 liby.a 中的 g，g 调用 libx.a 中的函数 h
gcc foo.c libx.a liby.a libx.a # OK
gcc foo.c libx.a liby.a        # Error: h undefined
gcc foo.c        liby.a libx.a # Error: g undefined
```

# 7. 搬迁<a href id="relocate"></a>



# 8. 可执行目标文件

![](https://csapp.cs.cmu.edu/3e/ics3/link/elfexec.pdf)

# 9. 加载可执行目标文件

# 10. 动态链接共享库

![](https://csapp.cs.cmu.edu/3e/ics3/link/sharedlibs.pdf)

# 11. 加载及链接共享库

# 12. 位置无关代码

|                      首次调用                       |                      后续调用                       |
| :-------------------------------------------------: | :-------------------------------------------------: |
| ![](https://csapp.cs.cmu.edu/3e/ics3/link/plt1.pdf) | ![](https://csapp.cs.cmu.edu/3e/ics3/link/plt2.pdf) |

# 13. 库打桩

# 14. 操作目标文件的工具

## `ar`

创建静态库，插入、删除、列出、提取成员

## `strings`

列出所有可打印字符串

## `strip`

删除符号列表信息

## `nm`

列出所有符号

## `size`

列出目所有字段的名称和大小

## `readelf`

显示目标文件的完整结构

## `objdump`

显示所有信息，常用于显示 `.text` 中二进制指令的反汇编代码

## `ldd`

列出某个可执行文件运行时所需的共享库

