# VTK --- Visualization Toolkit

[VTK](https://www.vtk.org) 是一款跨平台的 **数据显示 (Data Visualization)** 开源程序库.

## 文件格式

VTK User's Guide ([PDF](https://www.kitware.com/products/books/VTKUsersGuide.pdf)) 中的 `19.3 VTK File Formats` 详细介绍了两种文件格式: [VTK](#传统-VTK-格式) 与 [XML](#现代-XML-格式), 这里小结其要点.

### 传统 VTK 格式

这是一种只支持 **串行读写 (Sserial IO)** 的简单文本文件格式.
这种文件包含五个基本部分 (分节介绍), 其中前三项为必需, 后两种可选.

一些约定:
- `dataType` 表示数据类型, 常用的有 `int`, `long`, `float`, `double`.
- 关键词不区分大小写, 习惯上全部采用大写形式.
- 数组指标从 `0` 开始.
- Geometry 部分必须出现在 Attributes 之前.
- Attributes 中的结点或单元数据个数必须与 Geometry 中的结点或单元个数一致.

#### Version and Identifier
只占一行, 内容为:  
```vtk
## vtk DataFile Version x.x
```
其中 `x.x` 为文件格式的版本号, 当前为 `3.0`.

#### Header
描述文件基本信息的字符串, 以 `\n` 结尾, 最长含 `256` 个字符.

#### File Format
只占一行, 内容为 `ASCII` 或 `BINARY`, 表示 **数据** 的存储方式.
无论 **数据** 以文本还是二进制形式存储, **关键词** 总是以文本形式出现的.

#### Dataset Structure

描述结点及单元的 **几何** 位置与 **拓扑** 连接关系, 以 `DATASET type` 为第一行, 后接具体数据, 其中 `type` 可以是以下几种之一:

| `type`              | 中译名 |
| ------------------- | ---- |
| `STRUCTURED_POINTS` | 等距点阵 |
| `STRUCTURED_GRID`   | 结构网格 |
| `RECTILINEAR_GRID`  | 直线网格 |
| `UNSTRUCTURED_GRID` | 非结构网格 |
| `POLYDATA`          | 多边形数据 |
| `FIELD`             | 场 |

对于隐含拓扑的数据结构, 例如 `vtkImageData` 或 `vtkStructuredGrid`,  约定 `X` 方向增长得最快, `Y` 方向次之, `Z` 方向最慢.

##### `STRUCTURED_POINTS`
```vtk
DATASET STRUCTURED_POINTS
DIMENSIONS n_x n_y n_z
ORIGIN x y z
SPACING s_x s_y s_z
```

##### `STRUCTURED_GRID`
结构网格只给出结点位置信息, 节点编号及网格拓扑隐含在其中.
```vtk
DATASET STRUCTURED_GRID
DIMENSIONS n_x n_y n_z
POINTS n dataType
x[0] y[0] z[0]
x[1] y[1] z[1]
...
x[n-1] y[n-1] z[n-1]
```

##### `RECTILINEAR_GRID`
直线网格只给出三个递增的坐标序列, 结点位置与编号及网格拓扑隐含在其中.
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

##### `UNSTRUCTURED_GRID`
非结构网格的 **几何** 信息由 **结点位置** 给出 (与结构网格类似):
```vtk
DATASET UNSTRUCTURED_GRID
POINTS n dataType
x[0] y[0] z[0]
x[1] y[1] z[1]
...
x[n-1] y[n-1] z[n-1]
```
而 **拓扑** 信息则紧随其后, 由 **各单元结点列表**:
```vtk
CELLS n size
nPoints[0], i, j, k, l, ... 
nPoints[1], i, j, k, l, ... 
...
nPoints[n-1], i, j, k, l, ...
```
及 **单元类型**:
```vtk
CELL_TYPES n
type[0]
type[1]
...
type[n-1]
```
给出, 其中
- `CELLS` 后的参数:
  - `n`: 单元总数
  - `size`: 各行所需整数个数之和
- `CELL_TYPES`后的参数:
  - `n`: 单元总数, 必须与 `CELLS` 后的 `n` 一致
  - `type`: 由整数表示的单元类型, 详见 [`vtkCell.h`]()

##### `POLYDATA`
`POLYDATA` 由初等几何对象 (`POINTS`, `VERTICES`, `LINES`, `POLYGONS`, `TRIANGLE_STRIPS`) 拼接而成.
**几何** 信息由 `POINTS` 给出, 格式与结构网格中的 `POINTS` 字段一致:

```vtk
DATASET POLYDATA
POINTS n dataType
x[0] y[0] z[0]
x[1] y[1] z[1]
...
x[n-1] y[n-1] z[n-1]
```
**拓扑** 信息由 `VERTICES`, `LINES`, `POLYGONS`, `TRIANGLE_STRIPS` 字段给出, 格式与非结构网格中 `CELLS` 字段类似:

```vtk
VERTICES n size
nPoints[0], i[0], j[0], k[0], ...
nPoints[1], i[1], j[1], k[1], ...
...
nPoints[n-1], i[n-1], j[n-1], k[n-1], ...
```

##### `FIELD`

#### Dataset Attributes
描述结点或单元上携带的物理属性, 以 `POINT_DATA` 或 `CELL_DATA` 开始, 后接一个表示 `POINT` 或 `CELL` 数量的整数 (以下记为 `n`), 之后各行给出具体信息.

VTK 支持以下几种属性:

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

每种属性都有一个字符串 (以下用 `dataName` 表示) 与之关联, 用以区分各个物理量.
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
这是一种支持 **随机访问 (Random Access)** 和 **并行读写 (parallel IO)** 的文件格式.

#### PyEVTK --- 按 XML 格式输出数据
[PyEVTK](https://bitbucket.org/pauloh/pyevtk) 用于将 Python 程序中的数据输出为 XML 格式的文件.
安装后即可在本地 Python 程序中 `import` 该模块, 具体用法可以参照 `src/examples` 目录下的示例.

## ParaView --- GUI 前端
[ParaView](https://www.paraview.org) 是基于 VTK 实现的跨平台 GUI 前端.

启动后, 在 `Help` 列表中有各种本地或在线文档的链接.

使用 ParaView 主要分三个基本步骤:
1. 从数据文件中 **读取 (Read)** 原始数据.
2. 对原始数据进行 **过滤 (Filter)**, 提取出感兴趣的信息.
3. 将提取得到的信息在图形界面上进行 **渲染 (Render)**.

对于同一个数据文件, 第 1 步只需执行 1 次, 而第 2, 3 步可以执行多次.
