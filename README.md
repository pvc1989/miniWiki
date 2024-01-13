---
layout: page
title: 葵花宝典
---

# 内容目录

## 数学

- 高等代数
- 微积分、数学分析
- 实变函数论、实分析
- [复变函数论、复分析](./mathematics/complex_analysis.md)
- 泛函分析
- 常微分方程
- 偏微分方程
- 微分几何

## 物理

- [理论力学、分析力学](./physics/theoretical_mechanics.md)
- [电磁学、电动力学](./physics/electromagnetism.md)
- [量子力学](./physics/quantum_mechanics.md)
- 热力学与统计物理
- [连续介质力学](./physics/continuum/README.md)

## 算法

- [数据结构](./algorithms/data_structures/README.md)
- 数值计算
- 优化方法

## 编程

### 计算机基础

#### 组成原理

- [CSAPP](./programming/csapp.md)

#### 操作系统

- [链接](./programming/csapp/7_linking.md)
- [异常控制流](./programming/csapp/8_exceptional_control_flow.md)
- [虚拟存储](./programming/9_virtual_memory.md)
- [系统级读写](./programming/10_system_level_io.md)
- [并发编程](./programming/12_concurrent_programming.md)
- [Linux](./programming/linux.md)
- [Docker](./programming/docker.md)

#### 计算机网络

- [网络编程](./programming/csapp/11_network_programming.md)
- [SSH](./programming/linux/ssh.md)
- [网页](./documenting/web/README.md)

#### 数据库系统

- [数据模型](./programming/database/data_models.md)
- [SQL](./programming/database/sql.md)

### 软件开发

#### 高级语言

- [C/C++](./programming/languages/cpp.md)
- [MATLAB/Octave](./programming/languages/octave.md)
- [Python](./programming/languages/python.md)
- 编译原理

#### 架构设计

- [UML](./programming/design/uml/README.md) --- The Unified Modeling Language is a general-purpose, developmental, modeling language in the field of software engineering that is intended to provide a standard way to visualize the design of a system.
- [设计原则](./programming/design/principles/README.md)
- [设计模式](./programming/design/patterns/README.md)

#### 版本控制

- [Git](./programming/git.md) --- Git is a free and open source distributed version control system designed to handle everything from small to very large projects with speed and efficiency.

### 第三方库

#### 并行计算

- [Pthread](./programming/csapp/12_concurrent_programming.md#parallel) --- POSIX threads

- [MPI](./programming/mpi.md) --- Message Passing Interface (MPI) is a standardized and portable message-passing standard designed to function on parallel computing architectures.

- [CUDA](./programming/cuda.md)

#### CAD 建模

- [SALOME](www.salome-platform.org)
- [Open CASCADE](https://www.opencascade.com)

#### 网格

- [Gmsh](./programming/mesh/gmsh.md) --- Gmsh is an open source 3D finite element mesh generator with a built-in CAD engine and post-processor.
- [CGNS](./programming/mesh/cgns.md) --- The CFD General Notation System (CGNS) provides a general, portable, and extensible standard for the storage and retrieval of computational fluid dynamics (CFD) analysis data.
- [VTK](./programming/mesh/vtk.md) --- The Visualization Toolkit (VTK) is open source software for manipulating and displaying scientific data.
- [Plot3D](./programming/mesh/plot3d.md)

#### PDE 求解

- [OpenFOAM](./programming/openfoam.md)

## 文档
- [LaTeX](./documenting/latex/README.md)
  - [LyX](./documenting/latex/README.md#LyX)
  - [MathJax](./documenting/latex/README.md#MathJax)
- [Markdown](./documenting/markdown.md)
  - [Typora](./documenting/markdown.md#Typora)
- [网页](./documenting/web/README.md)
  - [HTML](./documenting/web/html.md)
  - [CSS](./documenting/web/css.md)
  - [JavaScript](./documenting/web/javascript.md)
- [Doxygen](./documenting/doxygen.md)
- [书签编辑器](./documenting/bookmark)
  - [PDFBookmarker](./documenting/bookmark.md#PDFBookmarker)
  - [DJVUSED](./documenting/bookmark.md#DJVUSED)

# 编写说明

## 语言
- [LyX](./documenting/latex/README.md#LyX)
  - 《[数学](#数学)》《[物理](#物理)》《[算法](#算法)》含有大量数学公式，因此整理为 LyX 文档。
  - 本页内的链接指向可单独编译的分卷，不同分卷可能含有重复的章节。
  - 『顶层 `README.lyx`』大致按逻辑顺序重新编排了章节（重复的只保留一份）。
- [Markdown](./documenting/markdown.md)
  - 《[Programming](#Programming)》《[Documenting](#Documenting)》含有大量代码，因此整理为 Markdown 文档。
  - 本仓库已启用 [GitHub Pages](https://docs.github.com/en/github/working-with-github-pages)（[首页](https://pvcstillingradschool.github.io/miniWiki/)），目前仅支持 Markdown 文档。

## 编译

- 在任意路径下创建本仓库的副本。
- 编译 LyX 文档（生成 PDF 文件）：
  1. 安装最新版的 [TeX Live](./documenting/latex/README.md#TeX-Live) 和 [LyX](./documenting/latex/README.md#LyX)，并开启[中文支持](./documenting/latex/README.md#LyX-中文支持)及[代码高亮](./documenting/latex/README.md#LyX-代码高亮)。
  1. 安装 [STIX](https://github.com/stipub/stixfonts) 及 [NewCM](https://ctan.org/tex-archive/fonts/newcomputermodern) 字体。
  1. 如果要编译除『顶层 `README.lyx`』以外的其他 LyX 或 TeX 文档，则需在本地 `$TEXMFHOME/tex/latex` 下创建一个指向 [`miniWiki/documenting/latex/pvcstyle.sty`](./documenting/latex/pvcstyle.sty) 的『符号链接 (symbolic link)』：
    
     |  操作系统  | `TEXMFHOME` 的默认值 |  创建符号链接的命令  |
     | :--------: | :------------------: | :------------------: |
     |   macOS    |  `~/Library/texmf`   | `ln -s TARGET LINK`  |
     | Windows 10 |      `~/texmf`       | `mklink LINK TARGET` |

- 编译 Markdown 文档（生成 HTML 文件）：
  1. 打开命令行终端，安装 [Ruby](https://www.ruby-lang.org/en/documentation/installation) 及 [Bundler](https://bundler.io/guides/bundler_2_upgrade.html#installing-bundler-2)。
     ```shell
     $ brew install ruby@2.7
     $ ruby --version
     ```
  1. 进入本仓库根目录，在命令行中输入
     ```shell
     $ bundle install
     $ bundle exec jekyll serve
     ```
  1. 生成的文件位于 `./_site/` 中。
  1. 在浏览器中访问 [`http://127.0.0.1:4000/`](http://127.0.0.1:4000/) 或 [`http://localhost:4000/`](http://localhost:4000/)。

## 排版

- 【空格】以下情形手动空一格：
  - 英文单词、行内公式、行内代码 两侧（左侧为行首、右侧为标点者除外）
  - 可能因断词产生歧义的汉字之间（例如「以 光武帝之德」与「以光 武帝之德」）
- 【标点】
  - 用中文标点修饰中文内容，例如 “……”『……』（……）〔……〕
  - 用英文标点修饰英文内容，e.g. "..." '...' (...) [...]
  - 数学环境内部使用英文标点，数学环境外部服从所属语言环境。
