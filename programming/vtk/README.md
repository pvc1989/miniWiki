---
title: 基于 VTK 的数据显示工具
---

# VTK

在不同的语境下，[VTK (Visualization ToolKit)](https://www.vtk.org) 可能表示：
- 用于**数据显示 (Data Visualization)** 的 C++ 程序库（或 Python wrapper）。
- 用于记录数据的[文件格式](#文件格式)，包括[传统 VTK 格式](./legacy_vtk_format.md)和[现代 XML 格式](#XML)。
- [传统 VTK 格式](./legacy_vtk_format.md)文件的默认扩展名。

## 面向对象设计

### VTK 类名

VTK 中的所有类名都是以 `vtk` 起始的。为节省空间以突出继承关系，下图中省去了该字段（即 `DataSet` 应补全为 `vtkDataSet`）。

[![](./classes.svg)](./classes.txt)

### 对象模型

在 VTK 中，几乎所有的类都是 `vtkObject` 的派生类。

所有**对象 (object)** 都是动态的，在 C++ 中，它们都必须由所属类的 `New()` 方法创建，并由所属类的 ` Delete()` 方法销毁：

```cpp
vtkObjectBase* obj = vtkExampleClass::New();  // 创建
otherObject->SetExample(obj);                 // 使用
obj->Delete();                                // 销毁
```

`New()` 方法返回指向动态数据的**原始指针 (raw pointer)**。
如果（在离开创建它的作用域前）忘记调用 `Delete()`，则会造成**内存泄漏 (memory leak)**。
VTK 提供了一种基于**引用计数 (reference count)** 的**智能指针 (smart pointer)** 来管理动态对象：

```cpp
auto obj = vtkSmartPointer<vtkExampleClass>::New();  // 创建
otherObject->SetExample(obj);                        // 使用
// obj->Delete();                                    // 销毁（自动完成）
```

### API

[在线文档](https://vtk.org/doc/nightly/html)只给出了 C++ 版的 API。
**解释型 (interpreted)** 语言（如 Python）的语法机制没有 C++ *丰 (fan) 富 (suo)*，*获 (cai) 得 (ce)* 相应的 API 需要对 C++ 版作适当的*压缩*。

## 计算机图形学基础

计算机图形学借用了一些电影行业的术语。**场景 (scene)** 由以下要素构成：

- 表示被显示的数据或对象的**演员 (actor)**。
- 发出光线照亮演员的**光源 (light source)**。
- 摄取演员在底片上投影的**镜头 (camera)**。

在 VTK 中，上述概念由以下类实现：

- `vtkRenderWindow` 对象负责管理**render window (渲染窗口)** 及**渲染器 (renderer)**。
  - 不同操作系统有不同的视窗实现方式。
  - VTK 隐藏了这些与具体系统相关的细节。
- `vtkRender` 对象负责协调演员、光源、镜头以生成图像。
  - 每个 `vtkRender` 对象都必须关联到一个 `vtkRenderWindow` 对象，且对应于其中的一个由*归一化坐标*定义的长方形**视窗 (viewport)**。
  - 若某个 `vtkRenderWindow` 对象只关联了一个 `vtkRender` 对象，则相应视窗的归一化坐标为 `(0,0,1,1)`，即占据整个渲染窗口。
- `vtkLight` 对象用于照亮其所处的场景。
  - 可调参数包括位置、方向、颜色、亮度、开关状态。
- `vtkCamera` 对象用于摄取场景。
  - 可调参数包括位置、方向、焦点、前（后）剪切平面、视场。
- `vtkActor` 对象表示场景中的演员，即被显示的数据或对象。
- `vtkMapper` 对象是数据的内部表示与显示模块之间的接口。

示例：

- [`${VTK_EXAMPLES}/src/GeometricObjects/Cone.py`](https://gitlab.kitware.com/vtk/vtk-examples/-/tree/master/src/Python/GeometricObjects/Cone.py)：几何源、渲染器、互动器。
- [`${VTK_EXAMPLES}/src/Interaction/CallBack.py`](https://gitlab.kitware.com/vtk/vtk-examples/-/tree/master/src/Python/Interaction/CallBack.py)：坐标轴、事件、观察者、回调函数。
- [`${VTK_EXAMPLES}/src/Renderings/Cone3.py`](https://gitlab.kitware.com/vtk/vtk-examples/-/tree/master/src/Python/Rendering/Cone3.py)：在一个 `vtkRenderWindow` 对象中并排显式两个 `vtkRender` 对象。
- [`${VTK_EXAMPLES}/src/Renderings/Cone4.py`](https://gitlab.kitware.com/vtk/vtk-examples/-/tree/master/src/Python/Rendering/Cone4.py)：属性对象、变换。

## 可视化管道

**数据可视化 (data visualization)** 包括以下两部分：

- **数据变换 (data transformation)** 是指由**原始数据 (original data)** 获得**图形基元 (graphics primitives)** 乃至**计算机图像 (computer images)** 的过程。描述数据变换过程的模型称为**功能模型 (functional models)**。
- **数据表示 (data representation)** 包括用于存储数据的**数据结构 (data structures)** 及用于显示数据的**图形基元 (graphics primitives)**。描述数据表示的模型称为**对象模型 (object models)**。

**可视化管道 (visualization pipeline)** 或**可视化网络 (visualization network)** 由以下对象构成：

- **数据对象 (data objects)** 用于表示信息。
- **操作对象 (operation objects)** 或**处理对象 (process objects)** 用于操作数据，可分为
  - **源 (source)** ：无输入、有输出。
  - **汇 (sink)** 或**映射器 (mapper)** ：有输入、无输出。
  - **滤镜 (filter)** ：有输入、有输出，以**端口 (port)** 与其他操作对象交互。

## 数据表示（文件格式）

详见《[VTK User's Guide](https://www.kitware.com/products/books/VTKUsersGuide.pdf)》的《VTK File Formats》一节，及《[The Visualization Toolkit](https://gitlab.kitware.com/vtk/textbook)》的《Basic Data Representation》一章。

### 传统 VTK 格式

这种格式的[定义](./legacy_vtk_format.md)较为简单，对于简单的应用，可以独立于 VTK 程序库实现一套 IO 模块。

### 现代 XML 格式<a name="XML"></a>
这是一种支持**随机访问 (random access)** 和**并行读写 (parallel IO)** 的文件格式，以 `.vt[irsupm]` 为扩展名：

| 扩展名 | 数据集类型              |
| ----- | --------------------- |
| `vti` | `vtkImageData`        |
| `vtr` | `vtkRectlinearGrid`   |
| `vts` | `vtkStructuredGrid`   |
| `vtu` | `vtkUnstructuredGrid` |
| `vtp` | `vtkPolyData`         |
| `vtm` | `vtkMultiBlockDataSet` |

这种格式的定义比[传统 VTK 格式](./legacy_vtk_format.md)复杂，建议直接调用 VTK 程序库提供的 [API](#API)。

如果在本地部署 VTK 程序库有困难（无网络、无权限），可以考虑使用 [PyEVTK](https://bitbucket.org/pauloh/pyevtk)。
它完全用 Python & Cython 实现，因此不依赖于 VTK 程序库。
安装后即可在本地 Python 程序中 `import` 该模块，具体用法可以参照 `src/examples` 目录下的示例。

### Python 示例
[`read.py`](./read.py) 和 [`write.py`](./write.py) 演示了如何用 `vtk` 模块提供的 API 读写 `vtkUnstructuredGrid`。
⚠️ 要运行该示例，必须先在本地[构建 VTK 程序库及 Python 模块](https://vtk.org/Wiki/VTK/Configure_and_Build)。
如果构建成功，则运行以下命令

```shell
mkdir build  # 在 write.py 所属目录下，创建 build 目录
cd build
python3 ../write.py 
```
会在 `build` 目录下生成四个文件：
```shell
ugrid_demo_ascii.vtk
ugrid_demo_ascii.vtu
ugrid_demo_binary.vtk
ugrid_demo_binary.vtu
```
读取其中的两个文件：
```
python3 ../read.py 
```

### C++ 示例
源代码仓库的 `Examples` 目录下有各个 VTK 模块的示例。

这里将 `Examples/IO/Cxx⁩/DumpXMLFile.cxx` 简化为 [`read.cpp`](./read.cpp)。
- 手动构建较为繁琐，且依赖于本地 VTK 的安装位置（假设：头文件位于 `/usr/local/include/vtk-8.2`，库文件位于 `/usr/local/lib`）及版本号：
```shell
mkdir build  # 在 read.cpp 所属目录下，创建 build 目录
cd build
c++ -c ../read.cpp -I/usr/local/include/vtk-8.2 -std=c++17
c++ -o read read.o -L/usr/local/lib -lvtkCommonDataModel-8.2 \
  -lvtkCommonCore-8.2 \
  -lvtkIOLegacy-8.2 \
  -lvtkIOXML-8.2 \
  -lvtkIOGeometry-8.2 \
  -lvtkIOImport-8.2 \
  -lvtksys-8.2
```
- 推荐使用 CMake 构建，构建选项以跨平台的方式写在 [`CMakeLists.txt`](./CMakeLists.txt) 中，CMake 会自动查找头文件及库文件位置：
```shell
mkdir build  # 在 read.cpp 所属目录下，创建 build 目录
cd build
cmake -S .. -B .
cmake --build .
./read *.vtk *.vtu  # 读取在《Python 示例》中生成的文件
```

## 可视化算法

VTK 中的**算法 (algorithm)** 是指对数据作变换（以改变数据的表示形式或生成其他数据）的对象。

算法可以按被处理的数据类型来分类：

- 标量算法，如：用等值线图显式标量场。
- 向量算法，如：用有向箭头显式向量场。
- 张量算法，如：提取二阶张量的主分量。
- 建模算法，其他无法归入以上各类的算法。

### 标量算法

**颜色映射 (color mapping)** 是一种常用的标量算法。它将标量值（如：温度值、压力值）映射为颜色值（如：RGB 数组）。该映射通常由以下方式实现：

- **查询表 (lookup table)**：选定被显示标量值的有效区间 `[s_min, s_max]` 及 RGB 数组的长度 `n`，则标量值 `s` 映射到 `(r[i], g[i], b[i])`，其中 `i` 由以下*线性分布*关系确定
  ```python
  i = 0
  if s_min < s:
      if s_max < s:
          i = n - 1
      else:
          i = n * (s - s_min) // (s - s_max)
  ```

- **变换函数 (transfer function)**：只需保证任意 `s` 都可以被唯一地映射到某个 `i`，可视为查询表的推广。

```python
from vtk import *
# 创建默认查询表（由红到蓝）
lookup_table = vtkLookupTable()
lookup_table.SetHueRange(0.6667, 0.0)
lookup_table.Build()
```



## 图像处理

# ParaView

[ParaView](https://www.paraview.org) 是基于 VTK 的 GUI 前端。

启动后，在 `Help` 列表中有各种本地或在线文档的链接，其中《Getting Started》可用于快速入门，《[ParaView Guide](https://docs.paraview.org/en/latest/)》用于系统学习。

## [可执行程序](https://docs.paraview.org/en/latest/UsersGuide/introduction.html#paraview-executables)

### `paraview`

`paraview` 是基于 Qt 的 GUI 程序。

### `pvpython`

`pvpython` 是封装了 ParaView 程序库的 Python shell —— 在其中可以使用 Python 自带的功能，也可以像调用其他 Python 模块（包）一样加载 ParaView 模块：

```python
from paraview.simple import *
```

在 `paraview` (GUI) 中的所有操作，几乎都可以在 `pvpython` (CLI) 中以 Python 指令的形式来完成。这些指令可以被同步记录到 `.py` 文件中，只需在 `paraview` (GUI) 中以 Tools → Start Trace 开启记录、以 Tools → Stop Trace 停止记录。

完整 API 列表参见《[ParaView's Python documentation](https://kitware.github.io/paraview-docs/latest/python/)》。

### `pvbatch`

`pvbatch` 是由 `.py` 文件驱动的 CLI 程序。

### `pvserver`

`pvserver` 是运行在远程主机上的 CLI 程序。

## 简单数据显示

使用 ParaView 主要分三个基本步骤：

1. 从数据文件中**读取 (read)** 原始数据。
2. 对原始数据进行**过滤 (filter)**，提取出感兴趣的信息。
3. 将提取得到的信息在图形界面上进行**渲染 (render)**。

对于同一个数据文件，第 1 步只需执行 1 次，而第 2、3 步可以执行多次。

## 动态数据显示

最简单（最通用）的动态数据显示方式，是将每一时间步的数据写入一个对应于该时刻的文件，这样的一个文件对应于动画中的一帧。ParaView 在加载按以下任意一种风格命名的一**组 (group)** 文件时，会自动将它们识别为一个**文件序列 (file series)**：

```txt
fooN.vtk
Nfoo.vtk
foo.vtk.N
foo_N.vtk
foo.N.vtk
N.foo.vtk
foo.vtksN
```
其中 `foo` 可以是任意非空字符串，`N` 为整数编号，扩展名 `.vtk` 可以替换为任意[ VTK 文件格式](#文件格式)所对应的扩展名。

## [远程数据显式](https://docs.paraview.org/en/latest/ReferenceManual/parallelDataVisualization.html#remote-visualization-in-paraview)
- 在远程及本地主机上：
  - 安装（版本号接近的）ParaView 软件。
  - 不建议用*包管理工具*安装，建议[到官网下载预编译版本](https://www.paraview.org/download/)。
- 在远程主机上：
  - 生成 VTK 数据文件。
  - 启动 `pvserver`。
- 在本地主机上：
  - 启动 `paraview`。
  - 连接到远程主机（首次连接需要 `Add Server`），如果 `Pipeline Browser` 中的 `builtin` 被替换为远程主机的标识符，则表明连接成功。
  - 像操作本地数据一样，加载并显示远程数据。

## [Filters](https://docs.paraview.org/en/latest/UsersGuide/filteringData.html)

### [Calculator](https://docs.paraview.org/en/latest/UsersGuide/filteringData.html#calculator)

```python
# 构造矢量
Velocity = (MomentumX*iHat + MomentumY*jHat + MomentumZ*kHat) / Density
# 构造标量
Pressure = EnergyStagnationDensity - dot(Velocity, Velocity) * Density / 2
```