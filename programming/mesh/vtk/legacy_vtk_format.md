---
title: 传统 VTK 格式
---

这是一种只支持**串行读写 (serial IO)** 的简单格式。
相应的文件以 `.vtk` 为默认扩展名，分为五个基本段落，其中后两段可选：

- [Version and Identifier](#Version)
- [Header](#Header)
- [File Format](#FileFormat)
- [DataSet Structure](#Structure)
- [DataSet Attributes](#Attributes)

一些约定：
- `data_type` 表示数据类型，常用的有 `int`, `long`, `float`, `double` 等。
- *关键词*不区分大小写，习惯上全部大写。
- 数组下标从 `0` 开始。
- [DataSet Structure](#Structure) 必须出现在 [DataSet Attributes](#Attributes) 之前，并且二者拥有相同的结点个数及单元个数。

# Version and Identifier<a name="Version"></a>
本段只占一行，其中 `2.0` 为 VTK *文件格式*的版本号，而不是 VTK *程序库*的版本号： 
```vtk
# vtk DataFile Version 2.0
```

# Header
本段为概述文件内容的字符串（给人看的辅助信息，不影响数据读写），以 `\n` 结尾，最多可含 `256` 个字符。

# File Format<a name="FileFormat"></a>
本段表示*数据*的存储方式，可以是 `ASCII` 或 `BINARY` 之一。
无论*数据*以 `ASCII` 还是 `BINARY` 形式存储，*关键词*总是以文本形式存储。

# DataSet Structure<a name="Structure"></a>
描述结点的**坐标 (coordinates)** 及**连接关系 (connection)** 。
以 `DATASET type` 为第一行，后接具体数据，其中 `type` 可以是以下几种之一：

| `type`              | `vtkDataSet` 的派生类  | 含义       |
| ------------------- | --------------------- | --------- |
| `STRUCTURED_POINTS` | `vtkImageData`        | 等距点阵   |
| `RECTILINEAR_GRID`  | `vtkRectlinearGrid`   | 非等距点阵 |
| `STRUCTURED_GRID`   | `vtkStructuredGrid`   | 结构网格   |
| `UNSTRUCTURED_GRID` | `vtkUnstructuredGrid` | 非结构网格 |
| `POLYDATA`          | `vtkPolyData`         | 多边形数据 |
| `FIELD`             | `vtkField`            | 场        |

对于隐含结点连接关系的类型（如 `STRUCTURED_POINTS`, `RECTILINEAR_GRID` 等），约定 `X` 方向的*结点编号*增长得最快、`Y` 方向次之、`Z` 方向最慢。

## `STRUCTURED_POINTS`
该类型对数据**规整度 (regularity)** 的要求最高，定义一个对象所需的信息最少，只有*原点坐标*和每个方向的*结点总数*与*结点间距*需要显式给出：
```vtk
DATASET STRUCTURED_POINTS
DIMENSIONS n_x n_y n_z
ORIGIN x_0 y_0 z_0
SPACING s_x s_y s_z
```

## `RECTILINEAR_GRID`
该类型放松了 `STRUCTURED_POINTS` 对*结点等距*的要求，允许每个方向的结点间距*独立变化*，因此需要显式给出三个独立递增的*坐标序列*：
```vtk
DATASET RECTILINEAR_GRID
DIMENSIONS n_x n_y n_z
X_COORDINATES n_x data_type
x[0] x[1] ... x[n_x - 1]
Y_COORDINATES n_y data_type
y[0] y[1] ... y[n_y - 1]
Z_COORDINATES n_z data_type
z[0] z[1] ... z[n_z - 1]
```

## `STRUCTURED_GRID`<a name="STRUCTURED_GRID"></a>
该类型放松了 `RECTILINEAR_GRID` 对*结点沿直线分布*的要求，但仍需保持**结构化的 (structured)** 连接关系，因此需要显式给出所有*结点坐标*：
```vtk
DATASET STRUCTURED_GRID
DIMENSIONS n_x n_y n_z
POINTS n_points data_type
x[0] y[0] z[0]
x[1] y[1] z[1]
...
x[n_points - 1] y[n_points - 1] z[n_points - 1]
```
其中 `n_points` 必须满足 `n_points == n_x * n_y * n_z`。
⚠️ VTK 中的一个*结构网格*相当于 ICEM 所生成的结构网格中的一个**区块 (block)** 。

## `UNSTRUCTURED_GRID`<a name="UNSTRUCTURED_GRID"></a>
该类型放松了 `STRUCTURED_GRID` 对*结点连接关系*的要求，允许结点以**非结构化的 (unstructured)** 方式连接，因此除了需要显式给出*结点坐标*：
```vtk
DATASET UNSTRUCTURED_GRID
POINTS n_points data_type
x[0] y[0] z[0]
x[1] y[1] z[1]
...
x[n_points - 1] y[n_points - 1] z[n_points - 1]
```
还需要显式给出各单元的*结点列表*及*单元类型*：
```vtk
CELLS n_cells n_ints
n_points[0], i, j, k, ... 
n_points[1], i, j, k, ... 
...
n_points[n_cells - 1], i, j, k, ...
CELL_TYPES n_cells
type[0]
type[1]
...
type[n_cells - 1]
```
其中
- `CELLS` 后的参数依次为
  - `n_cells` 表示单元总数。
  - `n_ints` 表示存储结点连接关系所需的整数个数。
  - `n_points[i]` 表示第 `i` 个单元的结点个数。
  - `i, j, k, ...` 表示结点编号。
- `CELL_TYPES` 后的参数依次为
  - `n_cells` 表示单元总数，必须与 `CELLS` 的 `n_cells` 一致。
  - `type[i]` 表示第 `i` 个单元的类型，详见 [`vtkCellType.h`](https://vtk.org/doc/nightly/html/vtkCellType_8h.html) 中的定义。

### 线性单元
![](https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/VTKBook/Figures/Figure5-2.png)

### 常用高阶单元
![](https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/VTKBook/Figures/Figure5-4.png)

### 任意高阶单元
![](https://gitlab.kitware.com/vtk/vtk/uploads/d18be24480da192e4b70568f050d114f/VtkLagrangeNodeNumbering.pdf)

## `POLYDATA`
该类型表示最一般的数据集，由初等几何对象 `POINTS`, `VERTICES`, `LINES`,  `POLYGONS`, `TRIANGLE_STRIPS` 拼凑而成。
- *结点坐标*由 `POINTS` 字段给出，格式与 [`STRUCTURED_GRID`](#STRUCTURED_GRID) 的 `POINTS` 字段一致：
```vtk
DATASET POLYDATA
POINTS n_points data_type
x[0] y[0] z[0]
x[1] y[1] z[1]
...
x[n_points - 1] y[n_points - 1] z[n_points - 1]
```
- *连接关系*由 `VERTICES`, `LINES`, `POLYGONS`, `TRIANGLE_STRIPS` 等字段给出，格式与 [`UNSTRUCTURED_GRID`](#UNSTRUCTURED_GRID) 的 `CELLS` 字段类似：
```vtk
LINES n_lines size
n_points[0] i[0] j[0] k[0], ...
n_points[1] i[1] j[1] k[1], ...
...
n_points[n_lines - 1] i[n_lines - 1] j[n_lines - 1] k[n_lines - 1] ...
```

## `FIELD`

# DataSet Attributes<a name="Attributes"></a>
描述结点或单元上的物理量，以 `POINT_DATA` 或 `CELL_DATA` 开始，后接一个表示 `POINT` 或 `CELL` 数量的整数（以下用 `n` 表示），之后各行给出具体信息。

VTK 支持以下几种 Attribute：

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

每种 Attribute 都有一个字符串（以下用 `data_name` 表示）与之关联，用以区分各个物理量。
如果格式要求里没有给出 `data_type`，那么数据类型取决于[文件类型](#FileFormat)：

- 如果文件类型为 `ASCII`，则数据为 `[0.0, 1.0]` 内的 `float` 型。
- 如果文件类型为 `BINARY`，则数据为 `[0, 255]` 内的 `unsigned char` 型。

## `SCALARS` 
`SCALARS` 字段包含一个可选的 `LOOKUP_TABLE` 字段：

```vtk
SCALARS data_name data_type n_components
LOOKUP_TABLE table_name
s[0]
s[1]
...
s[n - 1]
```
其中 `n_components` 是可选参数，可以是 `{1, 2, 3, 4}` 中的任何一个，默认为 `1`。

## `COLOR_SCALARS`
```vtk
COLOR_SCALARS data_name m
c[0][0] c[0][1] ... c[0][m - 1]
...
c[n - 1][0] c[n - 1][1] ... c[n - 1][m - 1]
```
其中 `m` 表示分量个数，因此 `COLOR_SCALARS` 可以看作是 `VECTORS` 即 `m == 3` 的推广。

## `LOOKUP_TABLE`
每一行的四个数分别代表 Red, Green, Blue, Alpha (opacity):
```
LOOKUP_TABLE table_name n
r[0] g[0] b[0] a[0]
r[1] g[1] b[1] a[1]
...
r[n] g[n] b[n] a[n]
```

##  `VECTORS`
```vtk
VECTORS data_name data_type
u[0] v[0] w[0]
u[1] v[1] w[1]
...
u[n - 1] v[n - 1] w[n - 1]
```

## `NORMALS`
通常用来表示面单元的*法线方向*，格式与 `VECTORS` 类似，但要求各向量都是*单位向量*。

## `TEXTURE_COORDINATES`

## `TENSORS`
目前只支持 `3 * 3` 实对称张量：
```vtk
TENSORS data_name data_type
t00[0] t01[0] t02[0]
t10[0] t11[0] t12[0]
t20[0] t21[0] t22[0]
...
t00[n - 1] t01[n - 1] t02[n - 1]
t10[n - 1] t11[n - 1] t12[n - 1]
t20[n - 1] t21[n - 1] t22[n - 1]
```

## `FIELD`
