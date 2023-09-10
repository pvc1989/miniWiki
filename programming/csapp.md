---
title: CSAPP
---

# 学习资源

## 教材

[Computer Systems: A Programmer's Perspective, 3/E (CS:APP3e)](https://csapp.cs.cmu.edu/3e/home.html)

- [勘误表](https://csapp.cs.cmu.edu/3e/errata.html)
- [配图](https://csapp.cs.cmu.edu/3e/figures.html)
- [代码](http://csapp.cs.cmu.edu/3e/code.tar)

## 课程

CMU [15-213/18-213: Introduction to Computer Systems (ICS)](https://www.cs.cmu.edu/~213/)

- [Fall 2015](https://www.cs.cmu.edu/afs/cs/academic/class/15213-f15/www/) 由二位作者按照教材第三版授课
  - [视频](https://scs.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx#folderID=%22b96d90ae-9871-4fae-91e2-b1627b43e25e%22)
  - [代码](http://www.cs.cmu.edu/afs/cs/academic/class/15213-f15/www/code.tar)

## 参考

- [The Linux Programming Interface](https://man7.org/tlpi/index.html)
- [Linux man pages online](https://man7.org/linux/man-pages/dir_all_alphabetic.html)

# 全书目录

## 1. 计算机系统速览

## 第一部分：程序的结构与执行（计组）

### [2. 信息的表示及运算](./csapp/2_bits_bytes_ints_floats.md)（计组）

- 二进制、十六进制
- 位、字节、地址、字节顺序
- 整数、位运算、逻辑运算、四则运算
- 浮点数

### [3. 程序的机器级表示](./csapp/3_machine_level_programming.md)（计组）

- 汇编码、指令、寄存器
- 有效地址、读写内存
- 位运算、逻辑运算、四则运算
- 条件码、条件跳转、无条件跳转
- 运行期栈、函数调用、递归函数
- 数组、结构体、数据对齐
- 越界攻击、GOP 攻击

### 4. 处理器架构（计组）

### [5. 优化程序性能](./csapp/5_optimizing_performance.md)（计组）

- 缓存常量、缓存中间值
- 指令级并行、循环展开、重新结合
- 内存性能
- 性能检测工具

### [6. 存储器等级体系](./csapp/6_memory_hierarchy.md)（计组）

- SRAM、DRAM、硬盘、固态硬盘
- 数据局部性、指令局部性
- 缓存器、直接映射、集合关联、完全关联

## 第二部分：在系统上运行程序（OS）

### [7. 链接](./csapp/7_linking.md)（OS）

- 目标文件、静态链接
- 符号、符号列表、符号解析
- 可执行文件、加载可执行文件
- 共享库、动态链接、位置无关代码
- 库打桩

### [8. 异常控制流](./csapp/8_exceptional_control_flow.md)（OS）

- 异常、中断、系统调用、故障、终止
- 逻辑控制流、并发流、进程、上下文切换
- 创建进程、结束进程、收割子进程、加载程序
- 信号、信号处置器、屏蔽信号

### [9. 虚拟内存](./csapp/9_virtual_memory.md)（OS）

- 物理地址、虚拟地址、地址翻译
- 页面列表、页面命中、页面故障
- 内存映射、共享资源、写时复制
- 动态内存、隐式链表、显式链表、分离链表、垃圾回收

## 第三部分：程序间互动及通信（OS + 网络）

### [10. 系统级读写](./csapp/10_system_level_io.md)（OS）

- Unix I/O、开关文件、读写文件
- Robust I/O
- 文件元数据、目录内容
- 共享文件、读写重定向

### [11. 网络编程](./csapp/11_network_programming.md)（网络）

- [计算机网络层次结构](./csapp/network_hierarchy.md)
- 客户端、服务端
- 网络、局域网、互联网、因特网
- IP 地址、端口号、套接字
- 网页服务、HTTP、动态内容

### [12. 并发编程](./csapp/12_concurrent_programming.md)（OS + 网络）

- 多进程并发、读写多路复用并发、多线程并发
- 共享变量、信号量、同步、互斥
- 多线程并行、强扩展性、弱扩展性
- 线程安全、再入函数、竞争、死锁

### 附录：异常处置
[`csapp.h`](http://csapp.cs.cmu.edu/3e/ics3/code/include/csapp.h), [`csapp.c`](http://csapp.cs.cmu.edu/3e/ics3/code/src/csapp.c)

# [Lab Assignments](https://csapp.cs.cmu.edu/3e/labs.html)

## 1. [Data Lab](./csapp/labs/data/README.md)
## 2. [Bomb Lab](./csapp/labs/bomb/README.md)
## 3. [Attack Lab](./csapp/labs/attack/README.md)
## 4. Architecture Lab
## 5. [Cache Lab](./csapp/labs/cache/README.md)
## 6. [Shell Lab](./csapp/labs/shell/README.md)
## 7. [Malloc Lab](./csapp/labs/malloc/README.md)
## 8. [Proxy Lab](./csapp/labs/proxy/README.md)

