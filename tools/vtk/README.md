# VTK: Visualization Toolkit

在不同的语境下，[VTK](https://www.vtk.org) 可能表示：
- 一款跨平台的「数据显示 (Data Visualization)」开源程序库。
- VTK 程序所使用的默认[文件格式](#文件格式)，包括[传统 VTK 格式](./legacy_vtk_format.md#传统-VTK-格式)和[现代 XML 格式](#现代-XML-格式)。《[VTK User's Guide](https://www.kitware.com/products/books/VTKUsersGuide.pdf)》的《19.3 VTK File Formats》一节详细定义了这两种格式。
- [传统 VTK 格式](./legacy_vtk_format.md#传统-VTK-格式)文件的默认扩展名。

## 文件格式
### 传统 VTK 格式
这种格式的[定义](./legacy_vtk_format.md#传统-VTK-格式)较为简单，对于简单的应用，实现一套 IO 接口（本地不需要部署 VTK 程序库）并不复杂。

### 现代 XML 格式
这是一种支持「随机访问 (random access)」和「并行读写 (parallel IO)」的文件格式，以 `.vt?` 为默认扩展名，其中 `?` 可以是 `[irsup]` 之一：

| 扩展名 | 数据集类型              |
| ----- | --------------------- |
| `vti` | `vtkImageData`        |
| `vtr` | `vtkRectlinearGrid`   |
| `vts` | `vtkStructuredGrid`   |
| `vtu` | `vtkUnstructuredGrid` |
| `vtp` | `vtkPolyData`         |

这种格式的定义比[传统 VTK 格式](./legacy_vtk_format.md#传统-VTK-格式)复杂，一般不建议自己手写 IO 接口，而是应该调用 VTK 程序库提供的 [API](#API)。

如果在本地部署 VTK 程序库有困难（无网络、无权限），可以考虑使用 [PyEVTK](https://bitbucket.org/pauloh/pyevtk)。
它完全用 Python & Cython 实现，因此不依赖于 VTK 程序库。
安装后即可在本地 Python 程序中 `import` 该模块，具体用法可以参照 `src/examples` 目录下的示例。

## ParaView
[ParaView](https://www.paraview.org) 是基于 VTK 的 GUI 前端。

启动后，在 `Help` 列表中有各种本地或在线文档的链接，其中《Getting Started》可用于快速入门。

使用 ParaView 主要分三个基本步骤：
1. 从数据文件中「读取 (read)」原始数据。
2. 对原始数据进行「过滤 (filter)」，提取出感兴趣的信息。
3. 将提取得到的信息在图形界面上进行「渲染 (render)」。

对于同一个数据文件，第 1 步只需执行 1 次，而第 2、3 步可以执行多次。

## API
[在线文档](https://vtk.org/doc/nightly/html)只给出了 C++ 版的 API。
其他「解释型 (interpreted)」语言的语法机制没有 C++ 丰 (fan) 富 (suo)，获 (cai) 得 (ce) 相应的 API 需要对 C++ 版作适当的 *压缩*。

### 对象模型
在 VTK 中，几乎所有的类都是 `vtkObject` 的派生类。

所有「对象 (object)」都是动态的，在 C++ 中，它们都必须由所属类的 `New()` 方法创建，并由所属类的 ` Delete()` 方法销毁：
```cpp
vtkObjectBase* obj = vtkExampleClass::New();  // 创建
otherObject->SetExample(obj);                 // 使用
obj->Delete();                                // 销毁
```

`New()` 方法返回指向动态数据的「原始指针 (raw pointer)」。
如果（在离开创建它的作用域前）忘记调用 `Delete()`，则会造成「内存泄漏 (memory leak)」。
VTK 提供了一种基于「引用计数 (reference count)」的「智能指针 (smart pointer)」来管理动态对象：
```cpp
auto obj = vtkSmartPointer<vtkExampleClass>::New();  // 创建
otherObject->SetExample(obj);                        // 使用
// obj->Delete();                                    // 销毁（自动完成）
```

### Python 示例
[`ugrid_reader_demo.py`](./ugrid_reader_demo.py) 和 [`ugrid_writer_demo.py`](./ugrid_writer_demo.py) 演示了如何用 `vtk` 模块提供的 API 读写 `vtkUnstructuredGrid`。
⚠️ 要运行该示例，必须先在本地[构建 VTK 程序库及 Python 模块](https://vtk.org/Wiki/VTK/Configure_and_Build)。
如果构建成功，则运行以下命令
```shell
python3 ugrid_writer_demo.py 
```
会在当前目录下生成四个文件：
```shell
ugrid_demo_ascii.vtk
ugrid_demo_ascii.vtu
ugrid_demo_binary.vtk
ugrid_demo_binary.vtu
```
