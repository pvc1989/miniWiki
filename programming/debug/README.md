# 断点调试

## 编译选项

```shell
$ cc -g hello.c -o hello
$ c++ -g hello.cpp -o hello
```

## GDB

### 参考文档

- [Homepage](https://www.gnu.org/software/gdb/)
- [Debugging with GDB](https://sourceware.org/gdb/current/onlinedocs/gdb/)

- [Beej's Quick Guide to GDB](http://beej.us/guide/bggdb/)
- [Debugging Under Unix: `gdb` Tutorial](https://www.cs.cmu.edu/~gilpin/tutorial/)

### 图形界面
- GDB 自带的 GUI 模式：以 `gdb -tui` 启动。
- [DDD：Data Display Debugger](http://www.gnu.org/software/ddd/)

### 在 macOS 上使用 GDB

1. 安装或更新：

   ```shell
   brew install gdb  # 首次安装
   brew upgrade gdb  # 更新版本
   ```

1. 创建并信任证书：
   参考 https://sourceware.org/gdb/wiki/PermissionsDarwin

1. 在 GDB 环境用如下命令（可写入 `~/.gdbinit` 中）关闭从终端启动被调试程序：

   ```
   set startup-with-shell off
   ```

## LLDB

### 参考文档

- [Homepage](#http://lldb.llvm.org/)
- [Tutorial](https://lldb.llvm.org/use/tutorial.html)
- [GDB to LLDB command map](https://lldb.llvm.org/use/map.html)

### 命令结构
LLDB 命令具有以下结构：
```
<noun> <verb> [-options [option-value]] [argument [argument...]]
```
可见：LLDB 命令通常比与之等价的 GDB 命令更长，但前者含义更加明确。
在命令缩写（别名）、自动补全的帮助下，使用 LLDB 命令并不需要输入很多字符。

## 命令速查

### 进入、退出调试器环境

*调试器环境* 是以 `(gdb)` 或 `(lldb)` 为行首提示符的命令行环境。

- 进入调试器环境：在命令行终端中输入 `gdb` 或 `lldb` 命令（通常附上 *被调试程序 (debugee)* 的文件名）。
- 退出调试器环境：在 `(gdb/lldb)` 后输入 `quit` 或 `q` 或按 `Ctrl + D` 组合键。

本节剩余部分所述 *命令* 均为调试器环境中的命令。
在没有歧义的情况下，这些命令（及选项）通常都支持（首字母、首二字母）简写：
```shell
(gdb)  break                 test.c   :    12
(gdb)  b                     test.c   :    12
(lldb) breakpoint set --file test.c --line 12
(lldb) br         s    -f    test.c  -l    12
```

在调试器环境中，输入
- `help <command>` 获取帮助信息。
- 回车键 重复上一条命令。

### 启动、退出被调试程序

```shell
# 加载被调试程序：
(gdb/lldb) file <debugee>
# 运行被调试的程序：
(gdb/lldb) run [argv]
(gdb/lldb) r   [argv]
# 为下次 `run` 设置输入参数
(gdb)           set            args [argv]
(lldb) settings set target.run-args [argv]
# 查看下次 `run` 的输入参数
(gdb)           show            args
(lldb) settings show target.run-args
# 退出被调试的程序
(gdb/lldb) kill
```

### 断点

```shell
# 设断点于函数 `main` 的入口处
(gdb)  break                 main
(lldb) breakpoint set --name main
# 设断点于所有名称中含有 `push` 的（C++）类方法的入口处
(gdb)  break                   push
(lldb) breakpoint set --method push
# 设断点于 C++ 类方法特定版本的入口处
(gdb)  break        Class::Method(int, int)
(lldb) br s --name 'Class::Method(int, int)'  # 含空格，需置于 ' ' 中
# 设断点于文件 `test.c` 的第 `12` 行
(gdb)  break                 test.c   :    12
(lldb) breakpoint set --file test.c --line 12
# 设断点于 `0x400544` 处
(gdb)  break                    *0x400544
(lldb) breakpoint set --address  0x400544
# 列出所有断点
(gdb)  info breakpoints
(lldb) breakpoint list
# 停用 `1` 号断点
(gdb)             disable 1
(lldb) breakpoint disable 1
# 启用 `1` 号断点
(gdb)             enable 1
(lldb) breakpoint enable 1
# 删除 `1` 号断点
(gdb)             delete 1
(lldb) breakpoint delete 1
# 删除 `main` 入口处的断点
(gdb)  clear main
# 删除所有断点
(gdb)             delete
(lldb) breakpoint delete
# 创建条件断点
(gdb)  break                 foo   if         count == 37
(lldb) breakpoint set --name foo --condition 'count == 37'
# 为现有断点设置条件（置于 ' ' 内）
(gdb)  condition N             CONDITION
(lldb) br modify N --condition CONDITION
```

### 执行

```shell
# 跳转至指定位置
(gdb/lldb) jump *0x1007145
# 源码级步进（遇到函数则进入其内部）：
(lldb) thread step-in
(gdb/lldb)    step
# 源码级跨越（遇到函数则视其为一行）：
(lldb) thread step-over
(gdb/lldb) next
# 指令级步进（遇到函数则进入其内部）：
(lldb) thread step-inst
(gdb)         stepi
(gdb/lldb)    si
# 指令级跨越（遇到函数则视其为一行）：
(lldb) thread step-inst-over
(gdb)         nexti
(gdb/lldb)    ni
# 恢复执行：
(gdb/lldb) continue
# 执行至当前函数返回：
(lldb) thread step-out
(gdb/lldb) finish
```

### 查看代码

```shell
# 反汇编 当前函数
(gdb)  disassemble
(lldb) disassemble --frame
# 反汇编 名为 `main` 的函数
(gdb)  disassemble        main
(lldb) disassemble --name main
# 反汇编 含有 `0x400544` 的函数
(gdb)  disassemble           0x400544
(lldb) disassemble --address 0x400544
# 反汇编 始于 `0x1eb8` 终于 `0x1ec3` 的指令
(gdb)  disassemble                 0x1eb8               0x1ec3
(lldb) disassemble --start-address 0x1eb8 --end-address 0x1ec3
# 反汇编 始于 `0x1eb8` 的 `20` 条指令
(gdb/lldb)                   x/20i 0x1eb8
(lldb) disassemble --start-address 0x1eb8 --count 20
```

### 查看数据

```shell
# 查看 `foo` 的值，其中
  # `foo` 可以是 变量、寄存器（表达式）、整数（表达式）
  # `/` 或 `--format` 后面的 `x`、`d`、`o`、`t`
  # 分别表示 十六进制、十进制（默认）、八进制、二进制
(gdb) print /x foo
(gdb/lldb) p/x foo
# 查看 给定地址处的给定类型的值（相当于 C 中的强制类型转换）
(gdb/lldb) print *(int *) 0x100d33  # 查看 `0x100d33`  处的整数值
(gdb/lldb) print *(int *) ($rsp+8)  # 查看 `R[%rsp]+8` 处的整数值
(gdb/lldb) print (char *) 0x100d33  # 查看 `0x100d33`  处的字符串
# x/[N][S][F] address
  # x = eXamine
  # N = Number of objects to display
  # S = Size of each object (b=byte, h=half-word,
  #                          w=word, g=giant (quad-word))
  # F = which Format (d=decimal, x=hex, o=octal, etc.)
  # 若 S 或 F 未给定，则取前一次的值。
(gdb/lldb) x/w   0x100d33  # 查看 始于 `0x100d33` 的 4-byte word
(gdb/lldb) x/wd  $rsp      # 查看 始于 `R[%rsp]`  的 4-byte word，以十进制显示
(gdb/lldb) x/2g  $rsp      # 查看 始于 `R[%rsp]`  的 `2` 个 quad-word
(gdb/lldb) x/a   $rsp      # 查看 `R[%rsp]` 中的地址
(gdb/lldb) x/s   0x100d33  # 查看 始于 `0x100d33` 的字符串
(gdb/lldb) x/20b main      # 查看 始于 `main` 的前 `20` 个字节
(gdb/lldb) x/10i main      # 查看 始于 `main` 的前 `10` 条指令
# 查看 局部变量的值
(gdb)  info args
(gdb)  info locals
(lldb) frame variable  # 当前帧内所有局部变量的值
(lldb) frame variable --format x foo  # 按格式 `x` 显示 `foo` 的值
# 查看 全局变量的值
(lldb) target variable foo
```

### 查看寄存器

```shell
# 查看 所有整型寄存器的值
(gdb)  info registers
(lldb) register read
# 查看 所有寄存器的值
(gdb)  info all-registers
(lldb) register read --all
# 查看 指定寄存器的值
(gdb)  info all-registers rdi rax
(lldb) register read      rdi rax
```


# 创建、修改变量
```shell
# 为帧内现有的变量赋值
(gdb)  set variable i = 40
(gdb)  set         (i = 40)
(lldb) expression   i = 40    # 除 expression 外，类似 C 语句
# 创建临时变量
(gdb)  set variable  $i = 40
(gdb)  set          ($i = 40)
(lldb) expression int i = 40  # 除 expression 外，类似 C 语句
```

### 查看其它信息

```shell
# 查看 当前调用栈信息
(gdb)         backtrace
(lldb) thread backtrace
(gdb/lldb)    bt
(gdb)  where
(gdb)  info stack  # stack 不能用首字母简写，故很少使用
# 查看 当前程序状态
(gdb)  info program
# 查看 当前程序中的函数名
(gdb)  info functions
# 查看 当前帧的信息
(gdb)  info frame
```

## 调试 CMake 项目

### [CMake Tools](../make/README.md#CMake-Tools)

