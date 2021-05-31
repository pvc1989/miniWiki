---
layout: page
title: 首页
---

# 编写说明

## 语言
- [LyX](./documenting/latex/README.md#LyX)
  - 《[数学](#数学)》《[物理](#物理)》《[算法](#算法)》含有大量数学公式，因此整理为 LyX 文档。
  - 本页内的链接指向可单独编译的分卷，不同分卷可能含有重复的章节。
  - 『[顶层 `README.lyx`](./README.lyx)』大致按逻辑顺序重新编排了章节（重复的只保留一份）。
- [Markdown](./programming/markdown.md)
  - 《[编程](#编程)》《[文档](#文档)》含有大量代码，因此整理为 Markdown 文档。
  - 本仓库已启用 [GitHub Pages](https://docs.github.com/en/github/working-with-github-pages)（[首页](https://pvcstillingradschool.github.io/miniWiki/)），目前仅支持 Markdown 文档。

## 编译

- 在任意路径下创建本仓库的副本。
- 编译 LyX 文档（生成 PDF 文件）：
  1. 安装最新版的 [TeX Live](./documenting/latex/README.md#TeX-Live) 和 [LyX](./documenting/latex/README.md#LyX)，并开启 [中文支持](./documenting/latex/README.md#LyX-中文支持) 及 [代码高亮](./documenting/latex/README.md#LyX-代码高亮)。
  1. 安装 [STIX](https://github.com/stipub/stixfonts) 字体。
  1. 如果要编译除『[顶层 `README.lyx`](./README.lyx)』以外的其他 LyX 或 TeX 文档，则需在本地 `$TEXMFHOME/tex/latex` 下创建一个指向 [`miniWiki/documenting/latex/pvcstyle.sty`](./documenting/latex/pvcstyle.sty) 的『符号链接 (symbolic link)』：
    
     |  操作系统  | `TEXMFHOME` 的默认值 |  创建符号链接的命令  |
     | :--------: | :------------------: | :------------------: |
     |   macOS    |  `~/Library/texmf`   | `ln -s TARGET LINK`  |
     | Windows 10 |      `~/texmf`       | `mklink LINK TARGET` |

- 编译 Markdown 文档（生成 HTML 文件）：
  1. 打开命令行终端，安装 [prerequisites](https://help.github.com/en/github/working-with-github-pages/testing-your-github-pages-site-locally-with-jekyll#prerequisites)。
  1. 进入本仓库根目录，在命令行中输入
     ```shell
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
- 【待补充】

## 纠错
移步 [Issues](https://github.com/pvcStillInGradSchool/miniWiki/issues)，尽量做到一个 issue 对应一个问题。

# 内容目录

## [数学](./mathematics/README.md)

- [线性代数](./mathematics/algebra/README.lyx)
- [实分析](./mathematics/real_analysis/README.lyx)
- [复分析](./mathematics/complex/README.lyx)
- [泛函分析](./mathematics/functional/README.lyx)
- [常微分方程](./mathematics/ode/README.lyx)
- [偏微分方程](./mathematics/pde/README.lyx)
- [微分几何](./mathematics/geometry/README.lyx)

## [物理](./physics/README.md)

- [理论力学](./physics/mechanics/README.md)
- [电磁学与电动力学](./physics/electromagnetism/README.md)
- [量子力学](./physics/quantum/README.md)
- [热力学与统计物理](./physics/heat/README.lyx)
- [流体力学（理论与计算）](./physics/fluid/README.md)

## [算法](./algorithms/README.md)

- [数据结构](./algorithms/data_structures/README.md)
- [数值分析](./algorithms/numerical_analysis/README.lyx)
- [优化方法](./algorithms/optimization/README.lyx)
- [有限单元](./algorithms/finite_element/README.lyx)

## [编程](./programming/README.md)
- 计算机系统
  - [CSAPP](./programming/csapp/README.md)
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
  - [MPI](./programming/mpi/README.md)
- 网格离散
  - [CGNS](./programming/cgns/README.md)
  - [Gmsh](./programming/gmsh/README.md)
  - [数据可视化](./programming/vtk/README.md)
    - [VTK](./programming/vtk/README.md#VTK)
    - [ParaView](./programming/vtk/README.md#ParaView)
- 构建工具
  - [版本控制](./programming/git.md)
    - [Git](./programming/git.md#Git)
    - [GitHub](./programming/git.md#GitHub)
  - [批量构建](./programming/cpp/make/README.md)
    - [GNU Make](./programming/cpp/make/README.md#GNU-Make)
    - [CMake](./programming/cpp/make/README.md#CMake)
    - [Ninja](./programming/cpp/make/README.md#Ninja)
  - [断点调试](./programming/debug/README.md)
    - [GDB](./programming/debug/README.md#GDB)
    - [LLDB](./programming/debug/README.md#LLDB)
- 开发环境
  - [Linux](./programming/linux/README.md)
    - [安装与配置](./programming/linux/install/README.md)
    - [SSH](./programming/linux/ssh.md)
    - [Vim](./programming/linux/vim.md)
  - [Docker](./programming/docker/README.md)

## [文档](./documenting/README.md)
- [LaTeX](./documenting/latex/README.md)
  - [LyX](./documenting/latex/README.md#LyX)
  - [MathJax](./documenting/latex/README.md#MathJax)
- [Markdown](./documenting/markdown.md)
  - [Typora](./documenting/markdown.md#Typora)
- [网页](./documenting/web/README.md)
  - [HTML](./documenting/web/html.md)
  - [CSS](./documenting/web/css.md)
  - [JavaScript](./documenting/web/javascript.md)
- [书签编辑器](./documenting/bookmark)
  - [PDFBookmarker](./documenting/bookmark.md#PDFBookmarker)
  - [DJVUSED](./documenting/bookmark.md#DJVUSED)
