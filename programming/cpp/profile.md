---
title: 性能检测
---

# GNU `gprof`

⚠️ 仅在 Linux 系统中可用。

## 编译

编译、链接时，开启 `-pg` 选项：

```shell
cc -c myprog.c -g -pg
cc -o myprog myprog.o -pg
# 或
cc -o myprog myprog.c -g -pg
```

## 运行

运行程序，生成 `gmon.out` 二进制文件：

```shell
./myprog
```

## 报告

```shell
gprof options [executable-file [profile-data-files...]] [> outfile]
# 纯文字报告：-p 或 --flat-profile
gprof -p myprog gmon.out > flat-report
# 图形化报告：-q 或 --graph
gprof -q myprog gmon.out > graph-report
```
