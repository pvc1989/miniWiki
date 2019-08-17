# VTK: Visualization Toolkit

在不同的语境下，[VTK](https://www.vtk.org) 可能表示：
- 一款跨平台的「数据显示 (Data Visualization)」开源程序库。
- VTK 程序所使用的默认[文件格式](#文件格式)，包括[传统 VTK 格式](./legacy_vtk_format.md#传统-VTK-格式)和[现代 XML 格式](#现代-XML-格式)。[*VTK User's Guide*](https://www.kitware.com/products/books/VTKUsersGuide.pdf) 的 *19.3 VTK File Formats* 一节详细定义了这两种格式。
- [传统 VTK 格式](./legacy_vtk_format.md#传统-VTK-格式)文件的默认扩展名。

# [传统 VTK 格式](./legacy_vtk_format.md#传统-VTK-格式)

# 现代 XML 格式
这是一种支持「随机访问 (random access)」和「并行读写 (parallel IO)」的文件格式，以 `.vt?` 为默认扩展名，其中 `?` 可以是 `[irsup]` 之一：

| 扩展名 | 数据集类型              |
| ----- | --------------------- |
| `vti` | `vtkImageData`        |
| `vtr` | `vtkRectlinearGrid`   |
| `vts` | `vtkStructuredGrid`   |
| `vtu` | `vtkUnstructuredGrid` |
| `vtp` | `vtkPolyData`         |

## API 示例
[`ugrid_demo.py`](./ugrid_demo.py) 演示了如何用 `vtk` 模块提供的 API 输出非结构网格及数据。
⚠️ 要运行该示例，必须先在本地[构建 VTK 程序库及 Python 模块](https://vtk.org/Wiki/VTK/Configure_and_Build)。
如果构建成功，则运行以下命令
```shell
python3 ugrid_demo.py 
```
会在当前目录下生成四个文件：
```shell
ugrid_demo_ascii.vtk
ugrid_demo_ascii.vtu
ugrid_demo_binary.vtk
ugrid_demo_binary.vtu
```

## PyEVTK
[PyEVTK](https://bitbucket.org/pauloh/pyevtk) 可用于将 Python 程序中的数据输出为 [XML 格式的文件](#现代-XML-格式)。
完全用 Python & Cython 实现，编译、安装不依赖于 VTK。
安装后即可在本地 Python 程序中 `import` 该模块，具体用法可以参照 `src/examples` 目录下的示例。

# ParaView
[ParaView](https://www.paraview.org) 是基于 VTK 的 GUI 前端。

启动后，在 `Help` 列表中有各种本地或在线文档的链接，其中 *Getting Started* 可用于快速入门。

使用 ParaView 主要分三个基本步骤：
1. 从数据文件中「读取 (read)」原始数据。
2. 对原始数据进行「过滤 (filter)」，提取出感兴趣的信息。
3. 将提取得到的信息在图形界面上进行「渲染 (render)」。

对于同一个数据文件，第 1 步只需执行 1 次，而第 2、3 步可以执行多次。
