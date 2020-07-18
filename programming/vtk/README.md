# VTK

在不同的语境下，[***VTK (Visualization ToolKit)***](https://www.vtk.org) 可能表示：
- 用于 ***数据显示 (Data Visualization)*** 的 C++ 程序库。
- 用于记录数据的[文件格式](#文件格式)，包括[传统 VTK 格式](./legacy_vtk_format.md#传统-VTK-格式)和[现代 XML 格式](#现代-XML-格式)。
- [传统 VTK 格式](./legacy_vtk_format.md#传统-VTK-格式)文件的默认扩展名。

## 文件格式

详细定义见《[VTK User's Guide](https://www.kitware.com/products/books/VTKUsersGuide.pdf)》的《VTK File Formats》一节。

### 数据类型

VTK 中的所有类名都是以 `vtk` 起始的，为突出继承关系，图中省去了该字段（即 `DataSet` 应理解为 `vtkDataSet`）。

[![](./classes.svg)](./classes.txt)

### 传统 VTK 格式

这种格式的[定义](./legacy_vtk_format.md#传统-VTK-格式)较为简单，对于简单的应用，可以独立于 VTK 程序库实现一套 IO 模块。

### 现代 XML 格式
这是一种支持 ***随机访问 (random access)*** 和 ***并行读写 (parallel IO)*** 的文件格式，以 `.vt[irsupm]` 为扩展名：

| 扩展名 | 数据集类型              |
| ----- | --------------------- |
| `vti` | `vtkImageData`        |
| `vtr` | `vtkRectlinearGrid`   |
| `vts` | `vtkStructuredGrid`   |
| `vtu` | `vtkUnstructuredGrid` |
| `vtp` | `vtkPolyData`         |
| `vtm` | `vtkMultiBlockDataSet` |

这种格式的定义比[传统 VTK 格式](./legacy_vtk_format.md#传统-VTK-格式)复杂，建议直接调用 VTK 程序库提供的 [API](#API)。

如果在本地部署 VTK 程序库有困难（无网络、无权限），可以考虑使用 [PyEVTK](https://bitbucket.org/pauloh/pyevtk)。
它完全用 Python & Cython 实现，因此不依赖于 VTK 程序库。
安装后即可在本地 Python 程序中 `import` 该模块，具体用法可以参照 `src/examples` 目录下的示例。

## API
[在线文档](https://vtk.org/doc/nightly/html)只给出了 C++ 版的 API。
其他 ***解释型 (interpreted)*** 语言的语法机制没有 C++ 丰 (fan) 富 (suo)，获 (cai) 得 (ce) 相应的 API 需要对 C++ 版作适当的 *压缩*。

### 对象模型
在 VTK 中，几乎所有的类都是 `vtkObject` 的派生类。

所有 ***对象 (object)*** 都是动态的，在 C++ 中，它们都必须由所属类的 `New()` 方法创建，并由所属类的 ` Delete()` 方法销毁：
```cpp
vtkObjectBase* obj = vtkExampleClass::New();  // 创建
otherObject->SetExample(obj);                 // 使用
obj->Delete();                                // 销毁
```

`New()` 方法返回指向动态数据的 ***原始指针 (raw pointer)***。
如果（在离开创建它的作用域前）忘记调用 `Delete()`，则会造成 ***内存泄漏 (memory leak)***。
VTK 提供了一种基于 ***引用计数 (reference count)*** 的 ***智能指针 (smart pointer)*** 来管理动态对象：

```cpp
auto obj = vtkSmartPointer<vtkExampleClass>::New();  // 创建
otherObject->SetExample(obj);                        // 使用
// obj->Delete();                                    // 销毁（自动完成）
```

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

## ParaView

[ParaView](https://www.paraview.org) 是基于 VTK 的 GUI 前端。

启动后，在 `Help` 列表中有各种本地或在线文档的链接，其中《Getting Started》可用于快速入门，《[ParaView Guide](https://www.paraview.org/paraview-guide/)》适合于系统学习。

### 简单数据显示

使用 ParaView 主要分三个基本步骤：

1. 从数据文件中 ***读取 (read)*** 原始数据。
2. 对原始数据进行 ***过滤 (filter)***，提取出感兴趣的信息。
3. 将提取得到的信息在图形界面上进行 ***渲染 (render)***。

对于同一个数据文件，第 1 步只需执行 1 次，而第 2、3 步可以执行多次。

### 动态数据显示

最简单（最通用）的动态数据显示方式，是将每一时间步的数据写入一个对应于该时刻的文件，这样的一个文件对应于动画中的一帧。ParaView 在加载按以下任意一种风格命名的一 ***组 (group)*** 文件时，会自动将它们识别为一个 ***文件序列 (file series)***：

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

### 远程数据显式
- 在远程及本地主机上：
  - 安装（版本号接近的）ParaView 软件。
  - 不建议用 *包管理工具* 安装，建议[到官网下载预编译版本](https://www.paraview.org/download/)。
- 在远程主机上：
  - 生成 VTK 数据文件。
  - 启动 `pvserver`。
- 在本地主机上：
  - 启动 `paraview`。
  - 连接到远程主机（首次连接需要 `Add Server`），如果 `Pipeline Browser` 中的 `builtin` 被替换为远程主机的标识符，则表明连接成功。
  - 像操作本地数据一样，加载并显示远程数据。
