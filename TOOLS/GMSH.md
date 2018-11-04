# Gmsh 简介

[Gmsh](http://gmsh.info/) 是一款带有简单 CAD 和后处理功能的三维有限元网格生成软件.
在遵循 [GPL](http://gmsh.info/#Licensing) 条款的前提下, 用户可以修改或重新发布其[源代码](https://gitlab.onelab.info/gmsh/gmsh).
初学者可以直接下载各种操作系统下的[预编译版](http://gmsh.info/bin/).

## 使用方式
Gmsh 可以按三种方式来使用: GUI 互动程序, 脚本驱动程序, C++ 程序库.
其中, 脚本驱动模式是学习价值最高的一种:
- 绝大多数 GUI 功能都有对应的脚本命令, 每一条 GUI 操作都会被记录在脚本文件中;
- 在 GUI 中可以很容易地打开或加载脚本文件:
```
Modules
    Geometry
        Reload script
        Edit script
```
> 建议: 在脚本文件中定义简单几何实体, 在 GUI 中执行选择或变换等更加复杂的操作.

在终端中, 可以直接令 Gmsh 完成网格生成和输出操作:
```shell
gmsh t1.geo -2
```
常用命令行参数:
| 参数 | 功能 |
| ---- | ---- |
| `-1`/`-2`/`-3` | 生成一/二/三维网格 |
| `-o filename` | 将网格输出到指定文件 |
| `-format string` | 选择网格文件格式, 例如 `msh4`, `msh2`, `vtk` |
| `-bin` | 以二进制模式输出 |
| `-part n` | 将网格分割为 `n` 块 (用于并行计算) |
完整列表参见 [Gmsh Reference Manual](http://gmsh.info/doc/texinfo/gmsh.html) 的 [3.3 Command-line options](http://gmsh.info/doc/texinfo/gmsh.html#Command_002dline-options).


## Running

## General Tools

## Geometry Module

## Mesh Module

## Post-processing Module

## File Formats

