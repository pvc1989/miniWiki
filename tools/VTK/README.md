# VTK --- Visualization Toolkit

在不同的语境下, [VTK](https://www.vtk.org) 可能表示:
- 一款跨平台的 **数据显示 (Data Visualization)** 开源程序库.
- VTK 程序所使用的默认[文件格式](#文件格式), 包括[传统 VTK 格式](#传统-VTK-格式)和[现代 XML 格式](#现代-XML-格式)两种.
- [传统 VTK 格式](#传统-VTK-格式)文件的默认扩展名.

## 文件格式
[VTK User's Guide](https://www.kitware.com/products/books/VTKUsersGuide.pdf) 中的 *19.3 VTK File Formats* 一节详细定义了[传统 VTK 格式](#传统-VTK-格式)和[现代 XML 格式](#现代-XML-格式), 这里小结其要点.

### 传统 VTK 格式

这是一种只支持 **串行读写 (Serial IO)** 的简单格式, 以 `.vtk` 为默认扩展名.
这种文件包含五个基本部分, 其中前三项为必需, 后两种可选:
- [Version and Identifier](#Version-and-Identifier)
- [Header](#Header)
- [File Format](#File-Format)
- [DataSet Structure](#DataSet-Structure)
- [DataSet Attributes](#DataSet-Attributes)

一些约定:
- `dataType` 表示数据类型, 常用的有 `int`, `long`, `float`, `double`.
- 关键词不区分大小写, 习惯上全部采用大写形式.
- 数组指标从 `0` 开始.
- [DataSet Structure](#DataSet-Structure) 必须出现在 [DataSet Attributes](#DataSet-Attributes) 之前.
- [DataSet Attributes](#DataSet-Attributes) 中的结点或单元数据个数必须与 [DataSet Structure](#DataSet-Structure) 中的一致.

#### Version and Identifier
只占一行, 其中 `x.x` 为 **文件格式** 的版本号 (不是 VTK 程序库的版本号):  
```vtk
## vtk DataFile Version x.x
```

#### Header
概述文件内容的字符串 (给人看的辅助信息, 不影响数据读写), 以 `\n` 结尾, 最长含 `256` 个字符.

#### File Format
表示 **数据** 的存储方式, 可以是 `ASCII` 或 `BINARY` 之一.
无论 **数据** 以文本还是二进制形式存储, **关键词** 总是以文本形式出现的.

#### DataSet Structure
描述结点的 **坐标** 及 **拓扑关系**.
以 `DATASET type` 为第一行, 后接具体数据, 其中 `type` 可以是以下几种之一:

| `type`              | `vtkDataSet` 的派生类  | 含义       |
| ------------------- | --------------------- | --------- |
| `STRUCTURED_POINTS` | `vtkImageData`        | 等距点阵   |
| `RECTILINEAR_GRID`  | `vtkRectlinearGrid`   | 非等距点阵 |
| `STRUCTURED_GRID`   | `vtkStructuredGrid`   | 结构网格   |
| `UNSTRUCTURED_GRID` | `vtkUnstructuredGrid` | 非结构网格 |
| `POLYDATA`          | `vtkPolyData`         | 多边形数据 |
| `FIELD`             | `vtkField`            | 场        |

对于隐含拓扑结构的类型 (例如 `STRUCTURED_POINTS`, `RECTILINEAR_GRID`), 约定 `X` 方向的 **结点编号** 增长得最快, `Y` 方向次之, `Z` 方向最慢.

##### `STRUCTURED_POINTS`
该类型对数据 **规律性 (Regularity)** 的要求最高, 定义一个对象所需的信息量最少, 只有 **原点坐标** 和每个方向的 **结点总数** 与 **结点间距** 需要显式给出:
```vtk
DATASET STRUCTURED_POINTS
DIMENSIONS n_x n_y n_z
ORIGIN x y z
SPACING s_x s_y s_z
```

##### `RECTILINEAR_GRID`
该类型放松了 `STRUCTURED_POINTS` 中对 **结点等距** 的要求, 允许每个方向的结点间距 **独立变化**, 因此需要显式给出三个独立递增的 **坐标序列**:
```vtk
DATASET RECTILINEAR_GRID
DIMENSIONS n_x n_y n_z
X_COORDINATES n_x dataType
x[0] x[1] ... x[n_x-1]
Y_COORDINATES n_y dataType
y[0] y[1] ... y[n_y-1]
Z_COORDINATES n_z dataType
z[0] z[1] ... z[n_z-1]
```

##### `STRUCTURED_GRID`
该类型放松了 `RECTILINEAR_GRID` 中对 **结点沿直线分布** 的要求, 但仍需保持 **结构化的 (Structured)** 拓扑, 因此需要显式给出所有 **结点坐标**:
```vtk
DATASET STRUCTURED_GRID
DIMENSIONS n_x n_y n_z
POINTS nPoints dataType
x[0] y[0] z[0]
x[1] y[1] z[1]
...
x[nPoints-1] y[nPoints-1] z[nPoints-1]
```
其中, 等式 `nPoints == n_x * n_y * n_z` 必须成立.
⚠️ VTK 中的一个 **结构网格** 相当于 ICEM 所生成的结构网格中的一个 **区块 (Block)**.

##### `UNSTRUCTURED_GRID`
该类型放松了 `STRUCTURED_GRID` 中对拓扑结构的要求, 允许结点以 **非结构化的 (Unstructured)** 方式连接, 因此除了需要显式给出 **结点位置**:
```vtk
DATASET UNSTRUCTURED_GRID
POINTS nPoints dataType
x[0] y[0] z[0]
x[1] y[1] z[1]
...
x[nPoints-1] y[nPoints-1] z[nPoints-1]
```
还需要显式给出 **各单元结点列表** (即 **结点拓扑关系**) 及 **各单元类型**:
```vtk
CELLS nCells nIntegers
nPoints[0], i, j, k, ... 
nPoints[1], i, j, k, ... 
...
nPoints[nCells-1], i, j, k, ...
CELL_TYPES nCells
type[0]
type[1]
...
type[nCells-1]
```
其中
- `CELLS` 后的参数:
  - `nCells` 表示单元总数
  - `nIntegers` 表示存储结点拓扑关系所需的整数个数
  - `nPoints[i]` 表示第 `i` 个单元的结点个数
- `CELL_TYPES` 后的参数:
  - `nCells` 表示单元总数, 必须与 `CELLS` 后的一致
  - `type[i]` 表示第 `i` 个单元的类型, 详见 [`vtkCellType.h`](https://vtk.org/doc/nightly/html/vtkCellType_8h.html)

##### `POLYDATA`
该类型表示最一般的数据集, 由初等几何对象 (`POINTS`, `VERTICES`, `LINES`, `POLYGONS`, `TRIANGLE_STRIPS`) 拼凑而成.
**结点位置** 由 `POINTS` 字段给出 (与 `STRUCTURED_GRID` 中的 `POINTS` 字段一致):
```vtk
DATASET POLYDATA
POINTS nPoints dataType
x[0] y[0] z[0]
x[1] y[1] z[1]
...
x[nPoints-1] y[nPoints-1] z[nPoints-1]
```
**拓扑关系** 由 `VERTICES`, `LINES`, `POLYGONS`, `TRIANGLE_STRIPS` 等字段给出 (与 `UNSTRUCTURED_GRID` 中的 `CELLS` 字段类似):
```vtk
LINES nLines size
nPoints[0], i[0], j[0], k[0], ...
nPoints[1], i[1], j[1], k[1], ...
...
nPoints[nLines-1], i[nLines-1], j[nLines-1], k[nLines-1], ...
```

##### `FIELD`

#### DataSet Attributes
描述结点或单元上携带的物理信息, 以 `POINT_DATA` 或 `CELL_DATA` 开始, 后接一个表示 `POINT` 或 `CELL` 数量的整数 (以下记为 `n`), 之后各行给出具体信息.

VTK 支持以下几种 Attribute:

| Attribute Name        | 中译名 |
| --------------------- | ----- |
| `SCALARS`             | 标量 |
| `COLOR_SCALARS`       | 颜色标量 |
| `LOOKUP_TABLE`        | 查询表 |
| `VECTORS`             | 向量 |
| `NORMALS`             | 法向量 |
| `TEXTURE_COORDINATES` | 纹理坐标 |
| `TENSORS`             | 张量 |
| `FIELD`               | 场 |

每种 Attribute 都有一个字符串 (以下用 `dataName` 表示) 与之关联, 用以区分各个物理量.
如果格式要求里没有给出 `dataType`, 那么数据类型取决于文件类型:

- `ASCII`: 在 `[0.0, 1.0]` 之间取值的 `float`
- `BINARY`: 在 `[0, 255]` 之间取值的 `unsigned char`

##### `SCALARS` 
`SCALARS` 字段包含一个可选的 `LOOKUP_TABLE` 字段:

```vtk
SCALARS dataName dataType nComponents
LOOKUP_TABLE tableName
s[0]
s[1]
...
s[n-1]
```
其中 `nComponents` 是可选参数, 可以是 `{1,2,3,4}` 中的任意一个值, 默认值为 `1`.

##### `COLOR_SCALARS`
```vtk
COLOR_SCALARS dataName m
c[0][0] c[0][1] ... c[0][m-1]
...
c[n-1][0] c[n-1][1] ... c[n-1][m-1]
```
其中 `m` 表示分量个数, 因此 `COLOR_SCALARS` 可以看作是 `VECTORS` 的推广 (`m=3`).

##### `LOOKUP_TABLE`
每一行的四个数分别代表红, 绿, 蓝, 透明度 (`0` 为透明):
```
LOOKUP_TABLE tableName n
r[0] g[0] b[0] a[0]
r[1] g[1] b[1] a[1]
...
r[n] g[n] b[n] a[n]
```

#####  `VECTORS`
```vtk
VECTORS dataName dataType
u[0] v[0] w[0]
u[1] v[1] w[1]
...
u[n-1] v[n-1] w[n-1]
```

##### `NORMALS`
通常用来表示面单元的法线方向, 格式与 `VECTORS` 类似, 但要求各向量都是 **单位向量**.

##### `TEXTURE_COORDINATES`

##### `TENSORS`
目前只支持 `3*3` 实对称张量:
```vtk
TENSORS dataName dataType
t00[0] t01[0] t02[0]
t10[0] t11[0] t12[0]
t20[0] t21[0] t22[0]
...
t00[n-1] t01[n-1] t02[n-1]
t10[n-1] t11[n-1] t12[n-1]
t20[n-1] t21[n-1] t22[n-1]
```

##### `FIELD`

### 现代 XML 格式
这是一种支持 **随机访问 (Random Access)** 和 **并行读写 (Parallel IO)** 的文件格式, 以 `.vt?` 为默认扩展名, 其中 `?` 可以是 `[irsup]` 之一:

| 扩展名 | 数据集类型              |
| ----- | --------------------- |
| `vti` | `vtkImageData`        |
| `vtr` | `vtkRectlinearGrid`   |
| `vts` | `vtkStructuredGrid`   |
| `vtu` | `vtkUnstructuredGrid` |
| `vtp` | `vtkPolyData`         |

#### API 示例
- 输出非结构网格: [Python](./ugrid_demo.py)

⚠️ 要运行以上示例, 必须先在本地[构建 VTK 程序库及 Python 模块](https://vtk.org/Wiki/VTK/Configure_and_Build).

如果构建成功, 应当可以运行以下命令
```shell
python3 ugrid_demo.py 
```
在当前目录下将会生成四个文件:
```shell
.
├── ugrid_demo.py
├── ugrid_demo_ascii.vtk
├── ugrid_demo_ascii.vtu
├── ugrid_demo_binary.vtk
└── ugrid_demo_binary.vtu
```

#### PyEVTK --- 按 XML 格式输出数据
[PyEVTK](https://bitbucket.org/pauloh/pyevtk) 用于将 Python 程序中的数据输出为 XML 格式的文件.
完全用 Python & Cython 实现, 可以独立于 VTK 程序库进行编译和安装.
安装后即可在本地 Python 程序中 `import` 该模块, 具体用法可以参照 `src/examples` 目录下的示例.

## ParaView --- GUI 前端
[ParaView](https://www.paraview.org) 是基于 VTK 实现的跨平台 GUI 前端.

启动后, 在 `Help` 列表中有各种本地或在线文档的链接, 其中 *Getting Started* 可用于快速入门.

使用 ParaView 主要分三个基本步骤:
1. 从数据文件中 **读取 (Read)** 原始数据.
2. 对原始数据进行 **过滤 (Filter)**, 提取出感兴趣的信息.
3. 将提取得到的信息在图形界面上进行 **渲染 (Render)**.

对于同一个数据文件, 第 1 步只需执行 1 次, 而第 2, 3 步可以执行多次.
