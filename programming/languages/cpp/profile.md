---
title: 性能检测
---

# Google `gperftools`<a href name="gperftools"></a>

## 安装

```shell
git clone https://github.com/gperftools/gperftools.git
cd gperftools
./autogen.sh
./configure
make
sudo make install
```

## 链接

```shell
c++ -o <executable-file> <source-files> -Wl,--no-as-needed,-lprofiler,--as-needed
```

## 运行

```shell
CPUPROFILE=<path-to-profile-file> <path-to-executable-file>
```

## 报告

```shell
pprof [options] <path-to-executable-file> <path-to-profile-file(s)>
# options:
#    --text              Generate text report
#    --gv                Generate Postscript and display
#    --evince            Generate PDF and display
#    --web               Generate SVG and display
```

# GNU `gprof`<a href name="gprof"></a>

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

运行程序，生成二进制文件 `gmon.out`：

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
