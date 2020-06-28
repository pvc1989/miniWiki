# 学习笔记

## 目录

### 数学

|                       卷                        |  章  |  节  |  进度   |
| :---------------------------------------------: | :--: | :--: | :-----: |
|  [线性代数](./mathematics/algebra/README.lyx)   |      |      | `00001` |
|     [实分析](./mathematics/real/README.lyx)     |      |      | `00001` |
|   [复分析](./mathematics/complex/README.lyx)    |      |      | `11111` |
| [泛函分析](./mathematics/functional/README.lyx) |      |      | `00011` |
|   [常微分方程](./mathematics/ode/README.lyx)    |      |      | `00111` |
|   [偏微分方程](./mathematics/pde/README.lyx)    |      |      | `00111` |
|  [微分几何](./mathematics/geometry/README.lyx)  |      |      | `00001` |

### 物理

|                    卷                    |  章  |  节  |  进度   |
| :--------------------------------------: | :--: | :--: | :-----: |
|   [热力学](./physics/heat/README.lyx)    |      |      | `00001` |
|  [流体力学](./physics/fluid/README.lyx)  |      |      | `00111` |
| [量子力学](./physics/quantum/README.lyx) |      |      | `00011` |


### 算法

|                           卷                           |  章  |  节  |  进度   |
| :----------------------------------------------------: | :--: | :--: | :-----: |
| [数值分析](./algorithms/numerical_analysis/README.lyx) |      |      | `00001` |
|      [优化](./algorithms/optimization/README.lyx)      |      |      | `00001` |
|    [有限元](./algorithms/finite_element/README.lyx)    |      |      | `00011` |

### 编程
- 计算机系统
  - [CSAPP (Computer Systems: A Programmer's Perspective)](./programming/csapp/README.md)
- 设计思想
  - [设计原则](./programming/principles/README.md)
  - [设计模式](./programming/patterns/README.md)
- 编程语言
  - [C++](./programming/cpp/README.md)
  - [Octave/MATLAB](./programming/octave.md)
  - [Python](./programming/python.md)
  - [UML](./programming/uml/README.md)
    - [PlantUML](./programming/uml/README.md#PlantUML)
- 并行计算
  - [MPI](./programming/mpi/README.md) (**M**essage **P**assing **I**nterface)
- 网格离散
  - [CGNS](./programming/cgns/README.md) (**C**FD **G**eneral **N**otation **S**ystem)
  - [Gmsh](./programming/gmsh/README.md) (a 3D finite element mesh generator with built-in CAD engine and post-processor)
  - [VTK](./programming/vtk/README.md) (**V**isualization **T**ool**K**it)
    - [ParaView](./programming/vtk/README.md#ParaView) (an open-source, multi-platform data analysis and visualization application)
- 构建工具
  - [版本控制](./programming/git.md)
    - [Git](./programming/git.md#Git)
    - [GitHub](./programming/git.md#GitHub)
  - [批量构建](./programming/make/README.md)
    - [GNU Make](./programming/make/README.md#GNU-Make)
    - [CMake](./programming/make/README.md#CMake)
  - [断点调试](./programming/debug/README.md)
    - [GDB](./programming/debug/README.md#GDB)
    - [LLDB](./programming/debug/README.md#LLDB)
- 开发环境
  - [Linux](./programming/linux/README.md) (a family of open source Unix-like operating systems)
    - [安装与配置](./programming/linux/install/README.md)
    - [SSH](./programming/linux/ssh.md) (**S**ecure **SH**ell)
    - [Vim](./programming/linux/vim.md) (**V**i **IM**proved, a programmer's text editor)
  - [Docker](./programming/docker/README.md) (a platform for developers and sysadmins to build, run, and share applications with containers)

### 文档
- [LaTeX](./documenting/latex/README.md)
  -  [LyX](./documenting/latex/README.md#LyX)
- [Markdown](./documenting/markdown.md)
  -  [Typora](./documenting/markdown.md#Typora)
- [书签](./programming/bookmark)
  - [PDFBookmarker](./programming/bookmark.md#PDFBookmarker) (add bookmarks into PDF using PyPDF2)
  - [DJVUSED](./programming/bookmark.md#DJVUSED) (a multi-purpose DjVu document editor)

## 说明

### 标记语言

《[数学](#数学)》《[物理](#物理)》《[算法](#算法)》含有大量数学公式，因此整理为 [LyX](./documenting/latex/README.md#LyX) 文档。
本页内的链接指向可单独编译的分卷，不同分卷可能含有重复的章节。
根目录下的《[合订本](./README.lyx)》大致按逻辑顺序重新编排了章节（重复的只保留一份）。

《[编程](#编程)》《[文档](#文档)》含有大量代码，因此整理为 [Markdown](./programming/markdown.md) 文档。

### 编译 LyX 文档

编译 LyX 文档（生成 PDF 或 HTML 文件）需要：

0. 安装 [TeX Live](./documenting/latex/README.md#TeX-Live) 和 [LyX](./documenting/latex/README.md#LyX)，并开启 [中文支持](./documenting/latex/README.md#中文支持) 及 [代码高亮](./documenting/latex/README.md#代码高亮)。
1. 安装 [STIX](https://github.com/stipub/stixfonts) 字体。
2. 在任意路径下创建本[仓库](./programming/git.md)的副本。
3. 在本地 `$TEXMFHOME/tex/latex` 下创建一个指向 [`miniWiki/documenting/latex/pvcstyle.sty`](./documenting/latex/pvcstyle.sty) 的 ***符号链接 (symbolic link)***：

   |  操作系统  | `TEXMFHOME` 的默认值 |  创建符号链接的命令  |
   | :--------: | :------------------: | :------------------: |
   |   macOS    |  `~/Library/texmf`   | `ln -s TARGET LINK`  |
   | Windows 10 |      `~/texmf`       | `mklink LINK TARGET` |

