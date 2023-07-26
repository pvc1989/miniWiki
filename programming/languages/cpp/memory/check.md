---
title: 内存检查
---

# Valgrind

## 参考

- [Home Page](https://www.valgrind.org)
- [Quick Start Guide](https://www.valgrind.org/docs/manual/quick-start.html)
- [User Manual](https://www.valgrind.org/docs/manual/manual.html)

## 示例

### 泄漏检测

```c
/* myprog.c */
#include <stdlib.h>

void f(void) {
    int* x = malloc(10 * sizeof(int));
    x[10] = 0;       // problem 1: heap block overrun
}                    // problem 2: memory leak -- x not freed

int main(void) {
    f();
    return 0;
}
```

编译、运行：

```shell
cc -g -o myprog myprog.c
valgrind --leak-check=yes ./myprog                                                   
```

输出以下检测报告：

```
==252730== Memcheck, a memory error detector
==252730== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==252730== Using Valgrind-3.16.1 and LibVEX; rerun with -h for copyright info
==252730== Command: ./myprog
==252730== 
==252730== Invalid write of size 4
==252730==    at 0x10916B: f (myprog.c:5)
==252730==    by 0x109180: main (myprog.c:9)
==252730==  Address 0x4a4a068 is 0 bytes after a block of size 40 alloc'd
==252730==    at 0x483C7F3: malloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==252730==    by 0x10915E: f (myprog.c:4)
==252730==    by 0x109180: main (myprog.c:9)
==252730== 
==252730== 
==252730== HEAP SUMMARY:
==252730==     in use at exit: 40 bytes in 1 blocks
==252730==   total heap usage: 1 allocs, 0 frees, 40 bytes allocated
==252730== 
==252730== 40 bytes in 1 blocks are definitely lost in loss record 1 of 1
==252730==    at 0x483C7F3: malloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==252730==    by 0x10915E: f (myprog.c:4)
==252730==    by 0x109180: main (myprog.c:9)
==252730== 
==252730== LEAK SUMMARY:
==252730==    definitely lost: 40 bytes in 1 blocks
==252730==    indirectly lost: 0 bytes in 0 blocks
==252730==      possibly lost: 0 bytes in 0 blocks
==252730==    still reachable: 0 bytes in 0 blocks
==252730==         suppressed: 0 bytes in 0 blocks
==252730== 
==252730== For lists of detected and suppressed errors, rerun with: -s
==252730== ERROR SUMMARY: 2 errors from 2 contexts (suppressed: 0 from 0)
```

### 访存记录

```
valgrind --log-fd=1 --tool=lackey -v --trace-mem=yes ls -l
```

运行 `ls -l`，捕获其内存访问，并打印到 `stdout`，输出格式为 `operation address,size`，如

```
I  100094f5b,3
 L 104925bc0,8
 M 104925bc8,4
 S 104925ba8,8
```

其中
- `I` 表示 Instruction load
- `L` 表示 data Load
- `M` 表示 data Modify
- `S` 表示 data Store
