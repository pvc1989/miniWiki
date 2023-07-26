---
title: CGNS
---

CGNS 是一种通用（跨平台、易扩展、受众广）的 CFD 文件（数据库）系统。

- [SIDS](#SIDS) 给出了这种文件格式的抽象定义。
- [MLL](#MLL) 是对 SIDS 的一种具体实现，它提供了读写（以 ADF 或 HDF5 作为底层数据格式的）CGNS 文件的 Fortran 及 C 语言 API。
- 文档中常用缩写含义如下：

| 术语 |                             含义                             |
| :--: | :----------------------------------------------------------: |
| ADF  | [*A*dvanced *D*ata *F*ormat](https://cgns.github.io/CGNS_docs_current/adf) |
| API  |           *A*pplication *P*rogramming *I*nterface            |
| CFD  |              *C*omputational *F*luid *D*ynamics              |
| CGNS | [*C*FD *G*eneral *N*otation *S*ystem](https://cgns.github.io/) |
| CPEX |              *C*GNS *P*roposal for *EX*tension               |
| FMM  |                  *F*ile *M*apping *M*anual                   |
| HDF  | [*H*ierarchical *D*ata *F*ormat](https://cgns.github.io/CGNS_docs_current/hdf5) |
| MLL  | [*M*id-*L*evel *L*ibrary](https://cgns.github.io/CGNS_docs_current/midlevel/) |
| SIDS | [*S*tandard *I*nterface *D*ata *S*tructures](https://cgns.github.io/CGNS_docs_current/sids/) |

# SIDS

## 官方文档

- 入门指南：《[Overview of the SIDS](https://cgns.github.io/CGNS_docs_current/user/sids.html)》
- 完整定义：《[CGNS Standard Interface Data Structures](https://cgns.github.io/CGNS_docs_current/sids/index.html)》

## 文件读写

符合 SIDS 规范的 CGNS 文件（数据库）是按 ADF 或 HDF5 编码的，因此无法用普通的文本编辑器读写，但可以用 [CGNSview](https://cgns.github.io/CGNS_docs_current/cgnstools/cgnsview/)、[HDFView](https://www.hdfgroup.org/downloads/hdfview/) 等工具安全地读写，其操作类似于在操作系统中访问*文件树*。

如果要构建上述工具，则需安装以下依赖库，并在 CMake 中将 `CGNS_BUILD_CGNSTOOLS` 设为 `ON`：

```shell
sudo apt install libx11-dev libxmu-dev
sudo apt install libopengl-dev libglu1-mesa-dev libgl1-mesa-dev 
sudo apt install libtcl8.6 tk8.6-dev
```

⚠️ [Gmsh](../gmsh/README.md)、[VTK](../vtk/README.md) 只能打开比它们所依赖的 *CGNS 库*更*旧*的 *CGNS 文件*。

## 示例文件

- [Example CGNS Files](http://cgns.github.io/CGNSFiles.html)
- [AIAA High Lift Workshop (HiLiftPW)](https://hiliftpw.larc.nasa.gov)
  - [NASA High Lift Common Research Model (HL-CRM) Grids](https://hiliftpw-ftp.larc.nasa.gov/HiLiftPW3/HL-CRM_Grids/)
  - [JAXA Standard Model (JSM) Grids](https://hiliftpw-ftp.larc.nasa.gov/HiLiftPW3/JSM_Grids/)
- [AIAA Drag Prediction Workshop (DPW)](https://aiaa-dpw.larc.nasa.gov) 
  - [Download Grids](https://dpw.larc.nasa.gov/DPW6/)

## 树状结构

每个 CGNS 文件（数据库）在逻辑上是一棵由若干**结点 (node)** 相互链接而成的**树 (tree)**。每个结点都含有以下数据：

- `Label` 表示其类型，通常是以 `_t` 为后缀的预定义类型。
- `Name` 表示其身份，通常是由用户自定义的字符串，但有时需符合命名规范。
- `Data` 是实际数据，可以为空（用 `MT` 表示）。
- 指向其**亲结点 (parent node)** 或**子结点 (child node)** 的链接。

⚠️ 为避免混淆，本文称*树上的结点*为**对象 (object)**，而将网格中的点称为**顶点 (vertex)** 或**网格点 (mesh/grid point)**。

# MLL

## 官方文档

- 入门指南：《[A User's Guide to CGNS](https://cgns.github.io/CGNS_docs_current/user/)》
- 完整定义：《[Mid-Level Library](https://cgns.github.io/CGNS_docs_current/midlevel/)》
  - 并行版本：《[Parallel CGNS Routines](https://cgns.github.io/CGNS_docs_current/pcgns/)》

## 并行版本

并行版的 CGNS/MLL 依赖于并行版的 HDF5，后者依赖于 MPI。安装这两个依赖项最简单的方式为：

```shell
apt  install hdf5-mpi # Ubuntu
brew install hdf5-mpi # macOS
```

或者先[手动构建 MPI](../mpi/README.md#编译运行)，再手动构建 HDF5。
以 Linux 下构建 HDF5-1.14.0 为例：

1. 下载 [HDF5® Source Code](https://www.hdfgroup.org/downloads/hdf5/source-code)，并解压到自选位置。设置环境变量 `HDF5_SOURCE_DIR` 使其指向源文件根目录。
1. 在自选位置创建 `build` 目录，设置环境变量 `HDF5_BUILD_DIR` 使其指向该目录。
1. 运行 `cmake -S ${HDF5_SOURCE_DIR} -B ${HDF5_BUILD_DIR} -D CMAKE_BUILD_TYPE=Release -D HDF5_ENABLE_PARALLEL=ON` 命令，开始配置。配置过程中，CMake 会自动查找已安装的 MPI；若自动查找失败，则需手动设置相关路径。
1. 配置完成后，进入 `${HDF5_BUILD_DIR}` 目录，运行 `cmake --build .` 命令，开始构建。
1. 构建完成后，在自选位置创建 `install` 目录，设置环境变量 `HDF5_INSTALL_DIR` 使其指向该目录。
1. 进入 `${HDF5_BUILD_DIR}` 目录，运行 `cmake --install . --prefix ${HDF5_INSTALL_DIR}` 命令，开始安装。

MPI 及 HDF5 安装完成后，即可在构建 CGNS/MLL 时，开启并行相关选项：

```shell
cmake -D CGNS_ENABLE_PARALLEL=ON \
      -D CGNS_ENABLE_HDF5=ON \
      -D HDF5_NEED_MPI=ON \
      -D HDF5_DIR:PATH=${HDF5_INSTALL_DIR}/cmake \
      -B ${CGNS_BUILD_DIR} -S ${CGNS_SOURCE_DIR}
```

## 数组索引

入门指南主要介绍 Fortran-API，这里（平行地）介绍 C-API ，以便 C/C++ 用户参考。

- C 的多维数组*按行 (row major)* 存储，Fortran 的多维数组*按列 (column major)* 存储，因此 *row in C* 对应于 *column in Fortran*。
- Fortran 的数组索引从 `1` 开始（与 SIDS 一致），而 C 则从 `0` 开始（故调用 C-API 时可能需要转换）。

## 演示代码

下载或克隆 [CGNS 代码库](https://github.com/CGNS/CGNS)后，可在 `${CGNS_SOURCE_DIR}/src/Test_UserGuideCode/` 中找到所有示例的源文件。源文件头部的注释给出了独立构建各示例的方法；若要批量构建所有示例，可在 CMake 中开启 `CGNS_ENABLE_TESTS` 等选项，这样生成的可执行文件位于 `${CGNS_BUILD_DIR}/src/Test_UserGuideCode/` 中。

# 基本模块

## `root`

```c++
root
├── CGNSLibraryVersion_t
│   ├── Name: CGNSLibraryVersion
│   └── Data: 版本号
└── CGNSBase_t  // 可有多个，各表示一个算例
```

```c++
/* open/close a CGNS file */
ier = cg_open(char *file_name,
    int mode/* CG_MODE_WRITE | CG_MODE_READ | CG_MODE_MODIFY */,
    /* output: */int *i_file);
ier = cg_close(int i_file);
/* parallel version (need MPI) */
ier = cgp_open(char *file_name, int mode,
    /* output: */int *i_file);
ier = cgp_close(int i_file);
```

用于新建对象的函数 `*_open()` 或 `*_write()` 总是以 `(int) id` 作为返回值。此 `id` 可以被后续代码用来访问该对象。

## `CGNSBase_t`

```c++
CGNSBase_t
├── Name: 文件名
├── Data: cell_dim  // int
│         phys_dim  // int
└── Zone_t  // 可有多个，各表示一块网格
```

```c
/* write/read CGNSBase_t */
ier = cg_base_write(int i_file, char *base_name, int cell_dim, int phys_dim,
    /* output: */int *i_base);
ier = cg_base_read(int i_file, int i_base,
    /* output: */char *base_name, int *cell_dim, int *phys_dim);
```

## `Zone_t`

```c++
Zone_t
├── ZoneType_t
│   ├── Name: ZoneType
│   └── Data: Structured | Unstructured | UserDefined | Null
├── GridCoordinates_t       // 顶点坐标
├── Elements_t              // 单元顶点列表（非结构网格特有）
├── FlowSolution_t          // 物理量在顶点或单元上的值
├── ZoneBC_t                // 边界条件
└── ZoneGridConnectivity_t  // 多区网格的衔接方式
```

```c
/* write/read Zone_t */
ier = cg_zone_write(int i_file, int i_base, char *zone_name, cgsize_t *zone_size,
    ZoneType_t zone_type/* CGNS_ENUMV(Structured) | CGNS_ENUMV(Unstructured) */,
    /* output: */int *i_zone);
ier = cg_zone_read(int i_file, int i_base, int i_zone,
    /* output: */char *zone_name, cgsize_t *zone_size);
```

其中

- `cell_dim`、`phys_dim` 分别表示*单元（流形）维数*、*物理（空间）维数*。
- `zone_size` 是一个二维数组（的首地址），
  - 其行数为 `3`，各行分别表示*顶点数*、*单元数*、*边界单元数* 。
  - 对于**结构网格 (structured grid)**：
    - *列数*至少等于*空间维数*，每列分别对应一个（逻辑）方向。
    - 各方向的*单元数*总是比*顶点数*少 `1`。
    - *边界单元*没有意义，因此最后一行全部为 `0`。
  - 对于**非结构网格 (unstructured grid)**：
    - *列数*至少为 `1`。
    - 若没有对单元排序（边界单元在前、内部单元在后），则*边界单元数*为 `0`。

## 定位操作

```c
/* Access a node via label-index or name-0 pairs.
 * 0 = CG_OK, 1 = CG_ERROR, 2 = CG_NODE_NOT_FOUND, 3 = CG_INCORRECT_PATH
 */
ier = cg_goto(int i_file, int i_base, ..., "end");
ier = cg_gopath(int i_file, const char *path);
/* Delete a child of current node. */
ier = cg_delete_node(char *node_name);
/* e.g. */
cg_goto(i_file, i_base, "Zone_t", 1, "UserDefinedData_t", 1, NULL);
cg_goto(i_file, i_base "Zone1", 0, "UserDefinedData_t", 1, NULL);
cg_goto(i_file, i_base, "Zone_t", 1, "User", 0, NULL);
cg_goto(i_file, i_base, "Zone1", 0, "User", 0, NULL);
cg_gopath(i_file, "/Base/Zone1/User");
```

# 单区网格

|      |      结构网格      |     非结构网格      |
| :--: | :----------------: | :-----------------: |
| 写出 | `write_grid_str.c` | `write_grid_unst.c` |
| 读入 | `read_grid_str.c`  | `read_grid_unst.c`  |

##  `GridCoordinates_t`

```c++
GridCoordinates_t
├── Name: GridCoordinates
└── DataArray_t  // 个数 == 所属 CGNSBase_t 对象的 phys_dim
    ├── Name: CoordinateX | CoordinateY | CoordinateZ |
    │         CoordinateR | CoordinateTheta | CoordinatePhi
    └── Data: 一维数组，长度 = 顶点数 + 表皮层数  // 沿当前方向
```
```c
/* write/read (GridCoordinates_t) "GridCoordinates" */
ier = cg_grid_write(int i_file, int i_base, int i_zone,
    char *grid_name/* GridCoordinates */,
    /* output: */int *i_grid);
ier = cg_grid_read(int i_file, int i_base, int i_zone, int i_grid,
    /* output: */char *grid_name);
/* write/read (DataArray_t) "Coordinate[XYZ]" */
ier = cg_coord_write(int i_file, int i_base, int i_zone,
    DataType_t data_type/* CGNS_ENUMV(RealDouble) */,
    char *coord_name, void *coord_array,
    /* output: */int *i_coord);
ier = cg_coord_read(int i_file, int i_base, int i_zone,
    char *coord_name, DataType_t data_type,
    cgsize_t *range_min, cgsize_t *range_max,  // 1-based inclusive
    /* output: */void *coord_array);
/* parallel version (need MPI) */
ier = cgp_coord_write(int i_file, int i_base, int i_zone,
    DataType_t data_type/* CGNS_ENUMV(RealDouble) */, char *coord_name,
    /* output: */int *i_coord);
ier = cgp_coord_read();  /* undefined */
ier = cgp_coord_write_data(int i_file, int i_base, int i_zone, int i_coord,
    cgsize_t *range_min, cgsize_t *range_max,  // 1-based inclusive
    void *coord_array);
ier = cgp_coord_read_data(int i_file, int i_base, int i_zone, int i_coord,
    cgsize_t *range_min, cgsize_t *range_max,  // 1-based inclusive
    /* output: */void *coord_array);
```

其中

- 函数 `cg_coord_write()` 写出的是以 `coord_array` 为首地址的前 `N` 个元素：
  -  `N` 可由 `zone_size[0]` 算出。
  - 对于结构网格，`coord_array`  通常声明为多维数组，此时除第一维长度*至少等于*该方向的顶点数外，其余维度的长度*必须等于*相应方向的顶点数。
  - 对于非结构网格，`coord_array`  通常声明为长度不小于 `N` 的一维数组。
- 坐标名 `coord_name` 必须取自《[SIDS-standard names](https://cgns.github.io/CGNS_docs_current/sids/dataname.html)》，即
  - 直角坐标：`CoordinateX | CoordinateY | CoordinateZ`
  - 球坐标：`CoordinateR | CoordinateTheta | CoordinatePhi`
- `data_type` 应当与 `coord_array` 的类型匹配：
  - `CGNS_ENUMV(RealSingle)` 对应 `float`。
  - `CGNS_ENUMV(RealDouble)` 对应 `double`。
- 并行版本的*写*分两步：
  - 先用 `cgp_write_coord()` 创建空 `DataArray_t` 对象；
  - 再用 `cgp_write_coord_data()` 写入当前进程所负责的部分。

## `Elements_t`

结构网格的*顶点信息*已经隐含了*单元信息*，因此不需要显式创建单元。与之相反，非结构网格的单元信息需要显式给出：

```c++
Elements_t
├── IndexRange_t
│   ├── Name: ElementRange
│   └── Data: int[2] = {first, last}  // 单个 Elements_t 内的单元需连续编号
├── ElementType_t
│   └── Data: NODE = 2 | BAR_2 = 3 | TRI_3 = 5 | QUAD_4 = 7 |
│             TETRA_4 = 10 | PYRA_5 = 12 | PENTA_6 = 14 | HEXA_8 = 17 |
│             NGON_n = 22 | NFACE_n = 23 | MIXED = 20
└── DataArray_t
│   ├── Name: ElementConnectivity
│   └── Data: int[ElementDataSize]  // ElementDataSize :=
│       // 同种单元 ElementSize * NPE(ElementType)
│       // 多种单元 Sum(NPE(ElementType[n]) + 1)
│       // NPE := number of nodes for the given ElementType
└── DataArray_t  // 含多种单元时使用
    ├── Name: ElementStartOffset
    └── Data: int[ElementSize + 1] 
```

```c
ier = cg_nsections(int i_file, int i_base, int i_zone,
    /* output: */int *n_sections);
// for fixed-size elements:
ier = cg_section_write(int i_file, int i_base, int i_zone,
    char *section_name, ElementType_t element_type,
    cgsize_t first, cgsize_t last, int n_boundary,
    cgsize_t *connectivity, /* output: */int *i_sect);
ier = cg_section_read(int i_file, int i_base, int i_zone, int i_sect,
    /* output: */char *section_name, ElementType_t *element_type,
    cgsize_t *first, cgsize_t *last, int *n_boundary, int *parent_flag);
ier = cg_elements_write();  /* undefined */
ier = cg_elements_read(int i_file, int i_base, int i_zone, int i_sect,
    /* output: */cgsize_t *connectivity, cgsize_t *parent_data);
ier = cg_elements_partial_write(int i_file, int i_base, int i_zone, int i_sect,
    cgsize_t first, cgsize_t last, cgsize_t *connectivity);
ier = cg_elements_partial_read(int i_file, int i_base, int i_zone, int i_sect,
    cgsize_t first, cgsize_t last,
    /* output: */cgsize_t *connectivity, cgsize_t *parent_data);
// for MIXED | NGON_n | NFACE_n:
ier = cg_poly_section_write(int i_file, int i_base, int i_zone,
    char *section_name, ElementType_t element_type,
    cgsize_t first, cgsize_t last, int n_boundary,
    cgsize_t *connectivity, cgsize_t *offset,
    /* output: */int *i_sect);
ier = cg_poly_section_read();  /* undefined */
ier = cg_poly_elements_write();  /* undefined */
ier = cg_poly_elements_read(int i_file, int i_base, int i_zone, int i_sect,
    /* output: */cgsize_t *connectivity, cgsize_t *offset, cgsize_t *parent_data);
ier = cg_poly_elements_partial_write(int i_file, int i_base, int i_zone, int i_sect,
    cgsize_t first, cgsize_t last, cgsize_t *connectivity, cgsize_t *offset);
ier = cg_poly_elements_partial_read(int i_file, int i_base, int i_zone, int i_sect,
    cgsize_t first, cgsize_t last,
    /* output: */cgsize_t *connectivity, cgsize_t *offset, cgsize_t *parent_data);
/* parallel version (need MPI) */
ier = cgp_section_write(int i_file, int i_base, int i_zone,
    char *section_name, ElementType_t element_type,
    cgsize_t first, cgsize_t last, int n_boundary,
    /* output: */int *i_sect);
ier = cgp_elements_write_data(int i_file, int i_base, int i_zone, int i_sect,
    cgsize_t first, cgsize_t last, cgsize_t *connectivity);
ier = cgp_section_read();  /* undefined */
ier = cgp_elements_read_data(int i_file, int i_base, int i_zone, int i_sect,
    cgsize_t first, cgsize_t last,
    /* output: */cgsize_t *connectivity);
```

其中

- `cg_section_write()` 在给定的 `Zone_t` 对象下新建一个**单元片段 (element section)**，即 `Elements_t` 对象。
- 一个 `Zone_t` 对象下可以有多个 `Elements_t` 对象：
  - 同一个 `Elements_t` 对象下的所有单元必须具有同一种 `element_type`，并且必须是枚举类型 [`ElementType_t`](file:///Users/master/code/mesh/cgns/doc/midlevel/general.html#typedefs) 的有效值之一。
  - 同一个  `Zone_t` 下的所有单元（含所有维数）都必须有*连续*且*互异*的编号。
  - ⚠️ 若要用 [ParaView](../vtk/README.md#ParaView) 打开，则只应保留最高维的单元片段。
- `first`、`last`
  - 在串行版本及 `cgp_section_write()` 中为当前 `Elements_t` 对象内的首、末单元的编号。
  - 在 `cgp_elements_write_data()` 及 `cgp_elements_write_data()` 中为当前进程所读写的首、末单元的编号。
- `n_boundary` 为当前 `Elements_t` 对象的*边界单元数*：若 `n_boundary > 0`，则单元已被排序，且前  `n_boundary` 个单元为边界单元。
- `parent_flag` 用于判断 parent data 是否存在。

## 运行示例

结构网格：

```shell
> ${CGNS_BUILD_DIR}/src/Test_UserGuideCode/C_code/write_grid_str

Program write_grid_str

created simple 3-D grid points
Successfully wrote grid to file grid_c.cgns
> ${CGNS_BUILD_DIR}/src/Test_UserGuideCode/C_code/read_grid_str

Successfully read grid from file grid_c.cgns
  For example, zone 1 x,y,z[8][16][20]= 20.000000, 16.000000, 8.000000

Program successful... ending now
```

非结构网格：

```shell
> ${CGNS_BUILD_DIR}/src/Test_UserGuideCode/C_code/write_grid_unst

Program write_grid_unst

created simple 3-D grid points

Successfully wrote unstructured grid to file grid_c.cgns
> ${CGNS_BUILD_DIR}/src/Test_UserGuideCode/C_code/read_grid_unst

number of sections=4

Reading section data...
   section name=Elem
   section type=HEXA_8
   istart,iend=1, 2560
   reading element data for this element

Reading section data...
   section name=InflowElem
   section type=QUAD_4
   istart,iend=2561, 2688
   not reading element data for this element

Reading section data...
   section name=OutflowElem
   section type=QUAD_4
   istart,iend=2689, 2816
   not reading element data for this element

Reading section data...
   section name=SidewallElem
   section type=QUAD_4
   istart,iend=2817, 3776
   not reading element data for this element

Successfully read unstructured grid from file grid_c.cgns
   for example, element 1 is made up of nodes: 1, 2, 23, 22, 358, 359, 380, 379
   x,y,z of node 357 are: 0.000000, 0.000000, 1.000000
   x,y,z of node 1357 are: 13.000000, 13.000000, 3.000000
```

⚠️ 本节生成的 `grid_c.cgns` 将在后续示例中反复使用，因此必确正确运行 `write_grid_str` 或 `write_grid_unst`，以获得以上输出。

# 流场数据
## `FlowSolution_t`

```c++
FlowSolution_t
├── GridLocation_t
│   └── Data: Vertex | CellCenter | EdgeCenter | FaceCenter
├── IndexRange_t  // 与 IndexList_t 二选一，对 Vertex | CellCenter 非必需
│   ├── Name: PointRange
│   └── Data: int[2] = {first, last}
├── IndexList_t  // 与 IndexRange_t 二选一，对 Vertex | CellCenter 非必需
│   ├── Name: PointList
│   └── Data: int[] = {...}
└── DataArray_t
    ├── Name: Pressure | Density | VelocityX | MomentumX | ...
    └── Data: DataType[DataSize]  /* 编号与相应的 Elements_t 一致
        if (有 PointRange | PointList):
            DataSize = Size(PointRange | PointList)
        else:
            DataSize = VertexSize | CellSize
            if (有 Rind):
                DataSize += RindSize  */
```

## 顶点数据

|      |       结构网格       |      非结构网格       |
| :--: | :------------------: | :-------------------: |
| 写出 | `write_flowvert_str` | `write_flowvert_unst` |
| 读入 | `read_flowvert_str`  | `read_flowvert_unst`  |

新增的 API 如下：

```c
/* write/read FlowSolution_t */
ier = cg_sol_write(int i_file, int i_base, int i_zone, char *sol_name,
    GridLocation_t location/* CGNS_ENUMV(Vertex) */,
    /* output: */int *i_soln);
ier = cg_sol_read();  /* undefined */
ier = cg_sol_info(int i_file, int i_base, int i_zone, int i_soln,
    /* output: */char *sol_name, GridLocation_t *location);
/* Read field info */
ier = cg_nfields(int i_file, int i_base, int i_zone, int i_soln,
    /* output: */int *n_fields);
ier = cg_field_info(int i_file, int i_base, int i_zone, int i_soln, int i_field,
    /* output: */DataType_t *data_type, char *field_name);
/* write DataArray_t */
ier = cg_field_write(int i_file, int i_base, int i_zone, int i_soln,
    DataType_t datatype, char *field_name, void *sol_array,
    /* output: */int *i_field);
ier = cg_field_partial_write(int i_file, int i_base, int i_zone, int i_soln,
    DataType_t datatype, char *field_name,
    cgsize_t *range_min, cgsize_t *range_max,  // 1-based inclusive
    void *sol_array,
    /* output: */int *i_field);
/* read DataArray_t */
ier = cg_field_read(int i_file, int i_base, int i_zone, int i_soln,
    char *field_name, DataType_t data_type,
    cgsize_t *range_min, cgsize_t *range_max,
    /* output: */void *sol_array);
ier = cg_field_partial_read();  /* undefined */
/* parallel version (need MPI) */
ier = cgp_field_write(int i_file, int i_base, int i_zone, int i_soln,
    DataType_t datatype, char *field_name,
    /* output: */int *i_field);
ier = cgp_field_write_data(int i_file, int i_base, int i_zone, int i_soln, int i_field,
    cgsize_t *range_min, cgsize_t *range_max,  // 1-based inclusive
    void *sol_array);
ier = cgp_field_read();  /* undefined, use cg_field_read() */
ier = cgp_field_read_data(int i_file, int i_base, int i_zone, int i_soln, int i_field,
    cgsize_t *range_min, cgsize_t *range_max,  // 1-based inclusive
    /* output: */void *sol_array);
ier = cgp_field_general_read_data(int i_file, int i_base, int i_zone, int i_soln, int i_field,
    cgsize_t *range_min, cgsize_t *range_max,  // range in file
    DataType_t mem_datatype/* data type of the array in memory */,
    int mem_rank/* number of dimensions of array in memory */,
    cgsize_t *mem_dimensions/* dimensions of array in memory */,
    cgsize_t *mem_range_min, cgsize_t *mem_range_max,  // range in memory
    /* output: */void *sol_array);
```

其中

- `cg_sol_write()` 用于在 `Zone_t` 对象下创建一个表示*一组物理量*的 `FlowSolution_t` 对象。
  - 同一个  `Zone_t` 对象下可以有多个 `FlowSolution_t` 对象。
  - 所有 `FlowSolution_t` 对象都平行于 `GridCoordinates_t` 对象。
- `cg_field_write()` 用于在 `FlowSolution_t` 对象下创建一个表示*单个物理量*的对象，例如  `DataArray_t` 对象、`Rind_t` 对象。
  - `sol_array` 尺寸应当与顶点数量匹配：对于结构网格，通常声明为多维数组；对于非结构网格，通常声明为一位数组。
  - `field_name` 应当取自《[SIDS-standard names](https://cgns.github.io/CGNS_docs_current/sids/dataname.html)》，例如 `Density | Pressure`。

## 单元数据

`write_flowcent_str.c` 与 `read_flowcent_str.c` 演示了这种流场表示方法，所用 API 与前一小节几乎完全相同，只需注意：

- 在调用 `cg_sol_write()` 时，将 `location` 的值由 `CGNS_ENUMV(Vertex)` 改为 `CGNS_ENUMV(CellCenter)`。
- 在结构网格的各逻辑方向上，用于存放数据的多维数组的长度必须与单元数量协调。

## 表皮数据

**表皮数据 (rind data)** 是指存储在网格最外侧的一层或多层**影子单元 (ghost cells)** 上的数据 ：

```
┌───╔═══╦═══╦═══╗───┬───┐      ═══ 网格单元
│ o ║ o ║ o ║ o ║ o │ o │
└───╚═══╩═══╩═══╝───┴───┘      ─── 影子单元
```

`write_flowcentrind_str.c` 与 `read_flowcentrind_str.c` 演示了这种表示方法，新增的 API 如下：

```c
/* API in `write_flowcentrind_str.c` and `read_flowcentrind_str.c` */

// Access a node via [label|name]-index pairs:
ier = cg_goto(int i_file, int i_base, ..., "end");
// e.g.
ier = cg_goto(i_file, i_base,
    "Zone_t", i_zone,
    "FlowSolution_t", i_soln,
    "end");

// Number of rind layers for each direction (structured grid):
int rind_data[6] = {
  1/* i_low */, 1/* i_high */,
  1/* j_low */, 1/* j_high */,
  0/* k_low */, 0/* k_high */};
// Write number of rind layers:
ier = cg_rind_write(int *rind_data);
// Read number of rind layers:
ier = cg_rind_read(int *rind_data);
```

其中

- `cg_goto()` 用于定位将要创建 `Rind_t` 对象的那个 `FlowSolution_t` 对象。
- 表皮数据存储在（根据影子单元层数）扩充的流场数组中，因此在结构网格的各逻辑方向上，用于存放数据的多维数组的长度必须与*扩充后的*单元数量协调。

# 量纲信息

## `DataClass_t`

```c
/* Write data class: */
ier = cg_goto(i_file, i_base, "end");
ier = cg_dataclass_write(CGNS_ENUMV(Dimensional));
ier = cg_dataclass_write(CGNS_ENUMV(NondimensionalParameter));
```

## `DimensionalUnits_t`

```c
/* Write first five dimensional units: */
ier = cg_units_write(
    CGNS_ENUMV(Kilogram), CGNS_ENUMV(Meter),
    CGNS_ENUMV(Second), CGNS_ENUMV(Kelvin),
    CGNS_ENUMV(Degree));
```

## `DimensionalExponents_t` 

```c
/* Write first five dimensional exponents of coordinates: */
float dimensional_exponents[5] = {0., 1., 0., 0., 0.};
ier = cg_goto(i_file, i_base, "Zone_t", i_zone, "GridCoordinates_t", 1,
    "DataArray_t", i_coord, "end");
ier = cg_exponents_write(CGNS_ENUMV(RealSingle), dimensional_exponents);
/* Write first five dimensional exponents of pressure: */
dimensional_exponents[0] = +1.0;
dimensional_exponents[1] = -1.0;
dimensional_exponents[2] = -2.0;
ier = cg_goto(i_file, i_base, "Zone_t", i_zone, "FlowSolution_t", i_soln,
    "DataArray_t", i_field, "end");
ier = cg_exponents_write(CGNS_ENUMV(RealSingle), dimensional_exponents);
```

# 边界条件

## 结构网格

两种 BC 表示方法：

- `PointRange` 通过指定*顶点编号范围*来确定边界，因此只适用于*结构网格的长方形边界*。`write_bc_str.c` 与 `read_bc_str.c` 演示了这种方法。
- `PointList` 通过指定*顶点编号列表*来确定边界，因此适用于*所有边界*。`write_bcpnts_str.c` 与 `read_bcpnts_str.c` 演示了这种方法。

尽管本节标题为*结构网格*，但上述方法也可以用于*非结构网格*，只是后者有[更简单的方法](#unstructured-bc)。

```c
/* API in `write_bc_str.c`    and `read_bc_str.c`
      and `write_bcpnt_str.c` and `read_bcpnt_str.c` */

// Write boundary condition type and data:
ier = cg_boco_write(int i_file, int i_base, int i_zone,
    char *boco_name, BCType_t boco_type/* CGNS_ENUMV(BCType_t) */,
    PointSetType_t point_set_type/* CGNS_ENUMV(PointRange) | CGNS_ENUMV(PointList) */,
    cgsize_t n_point, cgsize_t *point_set,
    /* output: */int *i_boco);

// Get number of boundary condition in zone:
ier = cg_nbocos(int i_file, int i_base, int i_zone,
    /* output: */int *n_boco);

// Get boundary condition info:
ier = cg_boco_info(int i_file, int i_base, int i_zone, int i_boco,
    /* output: */char *boco_name, BCType_t *boco_type,
    PointSetType_t *point_set_type, cgsize_t *n_point,
    int *i_normal,
    cgsize_t *normal_list_size,
    DataType_t *normal_data_type,
    int *n_data_set);

// Read boundary condition data and normals:
ier = cg_boco_read(int i_file, int i_base, int i_zone, int i_boco,
    /* output: */cgsize_t *point_set, void *normal_list);
```

其中

- `cg_boco_write()` 用于创建一个表示具体边界条件的 `BC_t` 对象。
  - 每个 `BC_t` 对象都含有一个 `IndexRange_t` 或 `IndexArray_t` 子对象。
  - 所有 `BC_t` 对象都是同一个 `ZoneBC_t` 对象的子对象。
  - 这个唯一的 `ZoneBC_t` 对象是某个 `Zone_t` 对象的子对象，因此是 `GridCoordinates_t` 对象及 `FlowSolution_t` 对象的同辈对象。
- `boco_type` 的取值必须是枚举类型 `BCType_t` 的有效值，例如 `BCWallInviscid | BCInflowSupersonic | BCOutflowSubsonic`，完整列表参见《[Boundary Condition Type Structure Definition](https://cgns.github.io/CGNS_docs_current/sids/bc.html#BCType)》。
- 二维数组 `point_set` 用于指定顶点编号，其行数（至少）为 `n_point`。
  - 对于结构网格，`point_set` 的列数为*空间维数*，而 `n_point`
    - 为 `2`，若 `point_set_type` 为 `CGNS_ENUMV(PointRange)`。此时 `point_set` 的第一、二行分别表示编号的下界、上界。
    - 为 此边界的顶点总数，若 `point_set_type` 为 `CGNS_ENUMV(PointList)`。
  - 对于非结构网格，`point_set` 的列数为 `1`，而 `n_point`
    - 为 此边界的顶点总数，且 `point_set_type` 只能为 `CGNS_ENUMV(PointList)`。

## 非结构网格<a href id="unstructured-bc"></a>

尽管*非结构网格*可以像*结构网格*那样，通过指定边界上的*顶点*来施加边界条件，但利用读写单元时创建的 `Elements_t` 对象来指定边界上的*单元*通常会更加方便。`write_bcpnts_unst.c` 与 `read_bcpnts_unst.c` 演示了这种方法，主要的 API 如下：

```c
/* API in `write_bcpnt_unst.c` and `read_bcpnt_unst.c` */

// Write boundary condition type and data:
ier = cg_boco_write(int i_file, int i_base, int i_zone, char *boco_name,
    BCType_t boco_type/* CGNS_ENUMV(BCType_t) */,
    PointSetType_t point_set_type/* CGNS_ENUMV(PointRange) | CGNS_ENUMV(PointList) */,
    cgsize_t n_cell, cgsize_t *cell_set,
    /* output: */int *i_boco);

// Write grid location:
ier = cg_gridlocation_write(GridLocation_t grid_location/*
    CGNS_ENUMV(CellCenter) | CGNS_ENUMV(FaceCenter) |
    CGNS_ENUMV(EdgeCenter) | CGNS_ENUMV(Vertex) */);
// Read grid location:
ier = cg_gridlocation_read(GridLocation_t *grid_location);
```

其中

- 这里的 `cg_boco_write()` 在形式上与*结构网格*版本一样，只是将 `n_point`、`point_set` 替换成了 `n_cell`、`cell_set`。
- `grid_location == CGNS_ENUMV(FaceCenter)` 表示当前边界条件定义在*面单元*上，即 `cell_set` 是存储*面单元*编号的数组。
- 调用 `cg_gridlocation_write()` 之前必须先用 `cg_goto()` 定位到所需的 `BC_t` 对象。

# 多区网格

# 动态数据

## [迭代数据结构](https://cgns.github.io/CGNS_docs_current/sids/timedep.html#IterativeData)

SIDS 定义了两种迭代数据结构，以管理多个时间（或迭代）步的数据：

`BaseIterativeData_t` 对象位于 `CGNSBase_t` 对象之下，一般用于存储*时间步总数*及*各步的时间值*，有时（如[网格拓扑发生改变](#网格拓扑发生改变)）也用来存储*指向各步的指针*：

```c++
BaseIterativeData_t := {
  int NumberOfSteps                                                  (r)

  DataArray_t<real, 1, NumberOfSteps> TimeValues ;                   (o/r)
  DataArray_t<int,  1, NumberOfSteps> IterationValues ;              (r/o)

  DataArray_t<int,  1, NumberOfSteps> NumberOfZones ;                (o)
  DataArray_t<int,  1, NumberOfSteps> NumberOfFamilies ;             (o)
  DataArray_t<char, 3, [65, MaxNumberOfZones, NumberOfSteps]>
    ZonePointers ;                                                   (o)
  DataArray_t<char, 3, [65, MaxNumberOfFamilies, NumberOfSteps]>
    FamilyPointers ;                                                 (o)

  List( DataArray_t<> DataArray1 ... DataArrayN ) ;                  (o)

  List( Descriptor_t Descriptor1 ... DescriptorN ) ;                 (o)

  DataClass_t DataClass ;                                            (o)

  DimensionalUnits_t DimensionalUnits ;                              (o)

  List( UserDefinedData_t UserDefinedData1 ... UserDefinedDataN ) ;  (o)
}
```

`ZoneIterativeData_t` 对象位于 `Zone_t` 对象之下，一般用于存储*指向各步的指针*：

```c++
ZoneIterativeData_t< int NumberOfSteps > := {
  DataArray_t<char, 2, [32, NumberOfSteps]>
    RigidGridMotionPointers ;                                       (o)
  DataArray_t<char, 2, [32, NumberOfSteps]>
    ArbitraryGridMotionPointers ;                                   (o)
  DataArray_t<char, 2, [32, NumberOfSteps]>
    GridCoordinatesPointers ;                                       (o)
  DataArray_t<char, 2, [32, NumberOfSteps]>
    FlowSolutionPointers ;                                          (o)
  DataArray_t<char, 2, [32, NumberOfSteps]>
    ZoneGridConnectivityPointers ;                                  (o)
  DataArray_t<char, 2, [32, NumberOfSteps]>
    ZoneSubRegionPointers ;                                         (o)

  List( DataArray_t<> DataArray1 ... DataArrayN ) ;                 (o)

  List( Descriptor_t Descriptor1 ... DescriptorN ) ;                (o)

  DataClass_t DataClass ;                                           (o)

  DimensionalUnits_t DimensionalUnits ;                             (o)

  List( UserDefinedData_t UserDefinedData1 ... UserDefinedDataN ) ; (o)
}
```

⚠️ 上述*指针*目前由*字符串*实现。

## 网格固定不变
**网格固定不变**意味着 `GridCoordinates_t` 及 `Elements_t`(s) 可复用，故只需将各时间步上的 `FlowSolution_t`(s) 记录在 `ZoneIterativeData_t` 内的 `(DataArray_t) FlowSolutionPointers` 中。

官方教程中的 `write_timevert_str.c` 与 `read_timevert_str.c` 演示了这种方法。
主要 API 用法如下：

```c
/* API in `write_timevert_str.c` and `read_timevert_str.c` */

// Base-level
ier = cg_simulation_type_write(i_file, i_base, CGNS_ENUMV(TimeAccurate));
ier = cg_biter_write(i_file, i_base, "TimeIterValues", n_steps);
ier = cg_goto(ii_file, i_base, "BaseIterativeData_t", 1, "end");
int n_steps = 3;
double times[3] = {1.0, 2.0, 3.0};
ier = cg_array_write("TimeValues", CGNS_ENUMV(RealDouble), 1, &n_steps, &times);
// Zone-level
ier = cg_ziter_write(i_file, i_base, i_zone, "ZoneIterativeData");
ier = cg_goto(i_file, i_base, "Zone_t", i_zone, "ZoneIterativeData_t", 1, "end");
char names[97];  /* need an extra byte for the terminating '\0' */
strcpy(names   , "FlowSolution1");
strcpy(names+32, "FlowSolution2");
strcpy(names+64, "FlowSolution3");
cgsize_t n_dims[2] = {32, 3};
ier = cg_array_write("FlowSolutionPointers", CGNS_ENUMV(Character),
    2, n_dims, names/* the head of a 2d array, whose type is char[3][32] */);
```

⚠️ 一个 `FlowSolution_t` 有且仅有一个 `GridLocation_t`，这意味着

- 在同一个 `FlowSolution_t` 内，不能既[顶点数据](#顶点数据)、又有[单元数据](#单元数据)。
- 在同一个 `FlowSolutionPointers` 内，所有 `FlowSolution_t`(s) 必须有相同的  `GridLocation_t` 值。
- 若要同时记录[顶点数据](#顶点数据)与[单元数据](#单元数据)，则应[平行于 `FlowSolutionPointers` 再创建一个 `FlowSolutionXXXPointers`](https://gitlab.kitware.com/paraview/paraview/-/issues/19838#note_732151)，这样两类数据都能被 [ParaView](../vtk/README.md#ParaView) 识别。示例文件 [`write_fixed_grid.cpp`](./write_fixed_grid.cpp) 演示了这种方法。

## [网格作刚体运动](https://cgns.github.io/CGNS_docs_current/sids/timedep.html#RigidGridMotion)

**网格作刚体运动**意味着 `Elements_t`(s) 可复用，而顶点坐标可以由*初始位置*（记录在当前 `Zone_t` 下唯一的 `GridCoordinates_t` 中）与*刚体运动信息*（随体坐标系的原点位置、原点速度、坐标架转角、坐标架转速等）快速地算出，后者记录在 `RigidGridMotion_t` 中（一个时间步对应一个这样的  `RigidGridMotion_t`，对应关系由 `ZoneIterativeData_t` 中的 `(DataArray_t) RigidGridMotionPointers` 管理）。

```c++
RigidGridMotion_t := {
  List( Descriptor_t Descriptor1 ... DescriptorN ) ;                 (o)

  RigidGridMotionType_t RigidGridMotionType ;                        (r)

  DataArray_t<real, 2, [PhysicalDimension, 2]> OriginLocation ;      (r)
  DataArray_t<real, 1,  PhysicalDimension>     RigidRotationAngle ;  (o/d)
  DataArray_t<real, 1,  PhysicalDimension>     RigidVelocity ;       (o)
  DataArray_t<real, 1,  PhysicalDimension>     RigidRotationRate ;   (o)

  List( DataArray_t DataArray1 ... DataArrayN ) ;                    (o)

  DataClass_t DataClass ;                                            (o)

  DimensionalUnits_t DimensionalUnits ;                              (o)

  List( UserDefinedData_t UserDefinedData1 ... UserDefinedDataN ) ;  (o)
}
```

## [顶点作任意运动](https://cgns.github.io/CGNS_docs_current/sids/timedep.html#ArbitraryGridMotion)

**顶点作任意运动**意味着 `Elements_t`(s) 仍可复用，但顶点坐标不再能快速算出，故需为每个时间步分别创建

- 一个 `GridCoordinates_t`，用于记录该时间步的*顶点坐标*，并将其*名称*记录在 `ZoneIterativeData_t` 下的 `(DataArray_t) GridCoordinatesPointers` 中。
- 一个 `ArbitraryGridMotion_t`，用于记录其所属 `Zone_t` 的*顶点速度*，并将其*名称*记录在 `ZoneIterativeData_t` 下的 `(DataArray_t) ArbitraryGridMotionPointers` 中。

```c++
ArbitraryGridMotion_t< int IndexDimension, 
                       int VertexSize[IndexDimension], 
                       int CellSize[IndexDimension] > := {
  ArbitraryGridMotionType_t ArbitraryGridMotionType ;                (r)

  List( DataArray_t<real, IndexDimension, DataSize[]>
        GridVelocityX GridVelocityY ... ) ;                          (o)

  List( Descriptor_t Descriptor1 ... DescriptorN ) ;                 (o)

  GridLocation_t GridLocation ;                                      (o/d)

  Rind_t<IndexDimension> Rind ;                                      (o/d)

  DataClass_t DataClass ;                                            (o)

  DimensionalUnits_t DimensionalUnits ;                              (o)

  List( UserDefinedData_t UserDefinedData1 ... UserDefinedDataN ) ;  (o)
}
```

## [网格拓扑发生改变](https://cgns.github.io/CGNS_docs_current/sids/timedep.html#ex:adaptedunstructuredmesh)

**网格拓扑发生改变**意味着 `Elements_t`(s) 不再能复用，故需创建新的 `Zone_t` 以对应*网格拓扑发生变化*的各时间步，对应关系由其所属 `CGNSBase_t` 中的 `(DataArray_t) ZonePointers` 管理。示例文件 [`write_adaptive_grid.cpp`](./write_adaptive_grid.cpp) 演示了这种方法。

```c++
CGNSBase_t "RemeshingCase" {
  BaseIterativeData_t "Steps" {
    int "NumberOfSteps" = 4
    DataArray_t<double> "TimeValues" = {0, 5, 10, 15, 20}
    DataArray_t<int> "NumberOfZones" = {1, 1, 1, 1, 1}
    DataArray_t<char*> "ZonePointers" = {
      "Zone1", "Zone1", "Zone1", "Zone2", "Zone2"
    }
  }
  Zone_t "Zone1" {
    GridCoordinates_t "GridCoordinates"
    Elements_t "Elements"
    ZoneBC_t "ZoneBC"
    FlowSolution_t "Solution0"
    FlowSolution_t "Solution5"
    FlowSolution_t "Solution10"
    ZoneIterativeData_t {
      DataArray_t<char*> "FlowSolutionPointers" = {
        "Solution0", "Solution5", "Solution10", "Null", "Null"
      }
    }
  }
  Zone_t "Zone2" = {
    GridCoordinates_t "GridCoordinates"
    Elements_t "Elements"
    ZoneBC_t "ZoneBC"
    FlowSolution_t "Solution15"
    FlowSolution_t "Solution20"
    ZoneIterativeData_t {
      DataArray_t<char*> "FlowSolutionPointers" = {
        "Null", "Null", "Null", "Solution15", "Solution20"
      }
    }
  }
}
```

另一种更通用的方法是创建  CGNS 文件序列，即各时间步分别对应一个 CGNS 文件。示例文件 [`write_file_series.cpp`](./write_file_series.cpp) 演示了这种方法。在 [ParaView](../vtk/README.md#ParaView) 中加载 CGNS 文件序列时，需勾选 `Ignore FlowSolutionPointers` 及 `Ignore Reader Time`，否则所有时间步会叠在一起显示。

