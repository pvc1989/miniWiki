# 断点调试

## GDB

### 参考文档

- [Homepage](#https://www.gnu.org/software/gdb/)
- [Debugging with GDB](#https://sourceware.org/gdb/current/onlinedocs/gdb/)

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
- [Tutorial](#https://lldb.llvm.org/use/tutorial.html)
- [GDB to LLDB command map](#https://lldb.llvm.org/use/map.html)

## 命令速查

### 进入、退出调试器环境

*调试器环境* 是以 `(gdb)` 或 `(lldb)` 为行首提示符的命令行环境。

- 进入调试器环境：在命令行终端中输入 `gdb` 或 `lldb` 命令（通常附上 *被调试程序 (debugee)* 的文件名）。
- 退出调试器环境：在 `(gdb)` 后输入 `quit` 或 `q` 或按 `Ctrl + D` 组合键。

本节剩余部分所述 *命令* 均为调试器环境中的命令。

### 启动、退出被调试程序

```shell
# 运行被调试的程序：
(gdb/lldb) run <args>
(gdb/lldb) r   <args>
# 为下次 `run` 设置输入参数
(gdb)           set            args 1
(lldb) settings set target.run-args 1
# 打印下次 `run` 的输入参数
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
(lldb) br         s    -n    main
# 设断点于文件 `test.c` 的第 `12` 行
(gdb)  break                 test.c:12
(lldb) b                     test.c:12
(lldb) breakpoint set --file test.c --line 12
(lldb) br         s    -f    test.c  -l    12
# 设断点于地址 `0x400540` 处
(gdb) break *0x400540
(lldb)
# 列出所有断点
(gdb) info break
(lldb) breakpoint list
(lldb) br         l
# 停用 `1` 号断点
(gdb)             disable 1
(lldb) breakpoint disable 1
(lldb) br         dis     1
# 启用 `1` 号断点
(gdb)             enable 1
(lldb) breakpoint enable 1
(lldb) br         en     1
# 删除 `1` 号断点
(gdb)             delete 1
(lldb) breakpoint delete 1
(lldb) br         del    1
# 删除所有断点
(gdb)             delete
(lldb) breakpoint delete
(lldb) br         del
```

### 执行

```shell
# 源码级步进（遇到函数则进入其内部）：
(lldb) thread step-in
(gdb/lldb)    step
(gdb/lldb)    s
# 源码级跨越（遇到函数则视其为一行）：
(lldb) thread step-over
(gdb/lldb) next
(gdb/lldb) n
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

### 查看反汇编

```shell
# 反汇编 当前函数
(gdb)  disassemble
(lldb) disassemble --frame
(lldb) di           -f
# 反汇编 `main`
(gdb)  disassemble        main
(lldb) disassemble --name main
(lldb) di           -n    main
# 反汇编 含有 `0x400544` 的函数
(gdb)  disassemble    0x400544
(lldb) disassemble -a 0x400544
(lldb) di          -a 0x400544
# 反汇编 始于 `0x1eb8` 终于 `0x1ec3` 的指令
(gdb)  disassemble                 0x1eb8               0x1ec3
(lldb) disassemble --start-address 0x1eb8 --end-address 0x1ec3
(lldb) di           -s             0x1eb8  -e           0x1ec3
# 反汇编 始于 `0x1eb8` 的 `20` 条指令
(gdb)                        x/20i 0x1eb8
(lldb) disassemble --start-address 0x1eb8 --count 20
(lldb) di           -s             0x1eb8  -c     20
# 打印 始于 `0x1eb8` 的 `20` 字节
(gdb/lldb) x/20b 0x1eb8
# 打印 下一条指令的地址
(gdb)  print /x $rip
(gdb/lldb)  p/x $rip
```

### 查看变量

```shell
# 打印 foo 的值，其中
  # foo 可以是 变量、寄存器（表达式）、整数（表达式）
  # x、d、o、t 分别表示 十六进制、十进制（默认）、八进制、二进制
(gdb) print /x foo
(gdb/lldb) p/x foo
# 打印 局部变量的值
(lldb) frame variable --format x foo
(lldb) fr    v         -f      x foo
# 打印 全局变量的值
(lldb) target variable foo
(lldb) ta     v        foo
# 打印
```

### 查看寄存器

```shell
# 打印 当前帧的信息
(gdb) info frame
# 打印 所有寄存器的值
(gdb) info registers
(lldb) register read
(gdb) info all-registers
(lldb) register read --all
# 打印 指定寄存器的值
(gdb) info all-registers rdi rax
(lldb) register read     rdi rax
```

## 调试 CMake 项目
