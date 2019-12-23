# 学习笔记

## 说明

《[数学](#数学)》《[物理](#物理)》《[算法](#算法)》部分的内容含有大量数学公式，因此整理为 [LyX](./programming/latex/README.md#LyX) 文档。
本页内的链接指向可单独编译的分卷，不同分卷可能含有重复章节；根目录下的[合订本](#README.lyx)大致按逻辑顺序重新编排了章节，删去了重复的内容。

《[编程](#编程)》《[工具](#工具)》部分的内容含有大量演示代码，因此整理为 [Markdown](./tools/markdown.md) 文档。

### 编译 LyX 文档

编译 LyX 文档（生成 PDF 或网页）需要：

0. 安装 [TeX Live](./programming/latex/README.md#TeX-Live) 和 [LyX](./programming/latex/README.md#LyX)，并完成相应配置，以[支持中文](./programming/latex/README.md#中文支持)及[代码高亮](./programming/latex/README.md#代码高亮)。
1. 安装 [STIX](https://github.com/stipub/stixfonts) 字体。如果想启用 `euler` 选项，则还需安装 [Neo Euler](https://github.com/khaledhosny/euler-otf) 字体。
2. 在任意路径下创建本[仓库](./programming/git.md)的副本。
3. 在本地 `$TEXMFHOME/tex/latex` 下创建一个指向 [`./programming/latex/pvcstyle.sty`](./programming/latex/pvcstyle.sty) 的「符号链接 (symbolic link)」：

|  操作系统  | `TEXMFHOME` 的默认值 |  创建符号链接的命令  |
| :--------: | :------------------: | :------------------: |
|   macOS    |  `~/Library/texmf`   | `ln -s TARGET LINK`  |
| Windows 10 |      `~/texmf`       | `mklink LINK TARGET` |

## 目录

### 数学

- [实分析 (Real Analysis)](./mathematics/analysis/README.lyx)
- [复分析 (Complex Analysis)](./mathematics/complex/README.lyx)
- [泛函分析 (Functional Analysis)](./mathematics/functional/README.lyx)
- [常微分方程 (Ordinary Differential Equations, ODE)](./mathematics/ode/README.lyx)
- [偏微分方程 (Partial Differential Equations, PDE)](./mathematics/pde/README.lyx)
- [微分几何 (Differential Geometry)](./mathematics/geometry/README.lyx)

### 物理
- [热力学 (Thermodynamics)](./physics/heat/README.lyx)
- [流体力学 (Fluid Mechanics)](./physics/fluid/README.lyx)

### 算法
- [数值分析 (Numerical Analysis)](./algorithms/numerical_analysis/README.lyx)
- [优化方法 (Optimization)](./algorithms/optimization/README.lyx)
- [有限单元法 (Finite Element Methods)](./algorithms/finite_element/README.lyx)
- [计算流体动力学 (Computational Fluid Dynamics)](./algorithms/cfd/README.lyx)

### 编程
- 设计思想
  - [设计原则 (Design Principles)](./programming/principles/README.md)
  - [设计模式 (Design Patterns)](./programming/patterns/README.md)
- 编程语言
  - [C++](./programming/cpp/README.md)
  - [Octave/MATLAB](./programming/octave.md)
  - [Python](./programming/python.md)
  - [UML](./programming/uml/README.md) + [PlantUML](./programming/uml/README.md#PlantUML)
- 构建工具
  - [Git](./programming/git.md#Git) + [GitHub](./programming/git.md#GitHub)
  - [GNU Make](./programming/make/README.md#GNU-Make) + [CMake](./programming/make/README.md#CMake)
- 排版工具
  - [Markdown](./programming/markdown.md) + [Typora](./programming/markdown.md#Typora)
  - [LaTeX](./programming/latex/README.md) + [LyX](./programming/latex/README.md#LyX)

### 工具
- [Gmsh](./tools/gmsh/README.md)
- [Linux](./tools/linux/README.md)
- [SSH](./tools/ssh.md)
- [Vim](./tools/vim.md)
- [VTK](./tools/vtk/README.md) + [ParaView](./tools/vtk/README.md#ParaView)
- [CGNS](./tools/cgns/README.md)
- [书签编辑工具](./tools/bookmark.md)
