# CGNS

CGNS 是一种通用（跨平台、易扩展、被广泛采用）的 CFD 文件（数据库）系统。[SIDS](#SIDS) 给出了这种文件格式的抽象定义。Fortran/C 程序库 [MLL](#MLL) 是对 SIDS 的一种具体实现，它提供了读写（以 ADF 或 HDF5 作为底层数据格式的）CGNS 文件的中层 API。

| 术语 |                             含义                             |
| :--: | :----------------------------------------------------------: |
| ADF  | [**A**dvanced **D**ata **F**ormat](https://cgns.github.io/CGNS_docs_current/adf) |
| API  |        **A**pplication **P**rogramming **I**nterface         |
| CFD  |           **C**omputational **F**luid **D**ynamics           |
| CGNS | [**C**FD **G**eneral **N**otation **S**ystem](https://cgns.github.io/) |
| CPEX |           **C**GNS **P**roposal for **EX**tension            |
| FMM  |               **F**ile **M**apping **M**anual                |
| HDF  | [**H**ierarchical **D**ata **F**ormat](https://cgns.github.io/CGNS_docs_current/hdf5) |
| MLL  | [**M**id-**L**evel **L**ibrary](https://cgns.github.io/CGNS_docs_current/midlevel/) |
| SIDS | [**S**tandard **I**nterface **D**ata **S**tructures](https://cgns.github.io/CGNS_docs_current/sids/) |

## SIDS

### 示例文件

- [Example CGNS Files](http://cgns.github.io/CGNSFiles.html)
- [AIAA High Lift Workshop (HiLiftPW)](https://hiliftpw.larc.nasa.gov)
  - [NASA High Lift Common Research Model (HL-CRM) Grids](https://hiliftpw-ftp.larc.nasa.gov/HiLiftPW3/HL-CRM_Grids/)
  - [JAXA Standard Model (JSM) Grids](https://hiliftpw-ftp.larc.nasa.gov/HiLiftPW3/JSM_Grids/)
- [AIAA Drag Prediction Workshop (DPW)](https://aiaa-dpw.larc.nasa.gov) 
  - [Download Grids](https://dpw.larc.nasa.gov/DPW6/)

符合 SIDS 规范的 CGNS 文件（数据库）是按 ADF 或 HDF5 编码的，因此无法用普通的文本编辑器读写，但可以用 [***CGNSview***](https://cgns.github.io/CGNS_docs_current/cgnstools/cgnsview/) 等工具安全地读写，其操作类似于在操作系统中访问 *文件（目录）树*。

### 文件结构

本节内容来自于 SIDS 的入门指南 [***Overview of the SIDS***](https://cgns.github.io/CGNS_docs_current/user/sids.html)。

#### Node

每个 CGNS 文件（数据库）在逻辑上是一棵由若干 ***结点 (node)*** 相互链接而成的 ***树 (tree)***。每个 node 都含有以下数据：

- ***`Label`*** 表示其类型，通常是以 `_t` 为后缀的预定义类型。
- ***`Name`*** 表示其身份，通常是由用户自定义的字符串，但有时需取作预定义的名称。
- ***`Data`*** 是实际数据，可以为空（用 `MT` 表示）。
- 指向其 parent、child(ren) 的链接。

#### Root

最顶层（没有 parent）的那个 node 称为 ***根 (root)***，它以如下 node(s) 作为其 child(ren)：

- 一个 ***`CGNSLibraryVersion_t`*** 结点：
  - `Name` 为 `CGNSLibraryVersion`。
  - `Data` 为 CGNS 的版本号。
- 若干 [***`CGNSBase_t`***](#`CGNSBase_t`)(s)，分别表示一个 ***算例 (case)***：
  - 多数 CGNS 文件只有一个 `CGNSBase_t`。
  - 如果含有多个，则不同 `CGNSBase_t` 之间不共享数据。

#### `CGNSBase_t` 

一个 `CGNSBase_t` 含有以下信息：

- `Name` 可以任取，例如取作文件名。
- `Data` 为两个整数：***单元维数 (cell dim)*** 与 ***物理维数 (physics dim)***，分别表示 *顶级单元* 与 *所处空间* 的维数。
- 若干 [***`Zone_t`***](#`Zone_t`)，分别表示网格中的一块区域。

#### `Zone_t`

一个 `Zone_t` 含有以下信息：

- 一个 ***`ZoneType_t`***，用于表示此区域的（网格）类型：
  - `Name` 为 `ZoneType`。 
  - `Data` 取自 `Structured`、`Unstructured`、`UserDefined`、`Null`。
- 若干 [***`GridCoordinates_t`***](#`GridCoordinates_t`)，用于表示 *结点坐标*。
- 若干 [***`Element_t`***](#`Element_t`)，用于表示 *非结构网格的单元结点列表*。
- 若干 [***`FlowSolution_t`***](#`FlowSolution_t`)，用于表示 *物理量在结点或单元上的值*。
- 若干 [***`ZoneBC_t`***](#`ZoneBC_t`)，用于表示 *边界条件*。
- 若干 [***`ZoneGridConnectivity_t`***](#`ZoneGridConnectivity_t`)，用于表示 *多块网格的连接方式*。

#### `GridCoordinates_t`

每个 `Zone_t` 下可以有多个 `GridCoordinates_t`，用于表示变化的网格。最初的那个通常以 `GridCoordinates` 为其 `Name`。

各 *结点坐标分量* 分别存储为一个 ***`DataArray_t`***：

- `Name` 应取自 `CoordinateX`、`CoordinateY`、`CoordinateZ`、`CoordinateR`、`CoordinateTheta`、`CoordinatePhi`、`CoordinateNormal`。
- 数组 *个数* 必须等于其所属 `CGNSBase_t` 中的 *物理维数*。
- 数组 *长度* 必须等于其所属 `Zone_t` 中（相应方向）的 *结点个数* 与 *外皮层数* 之和。

#### `Element_t`

#### `FlowSolution_t`

- 各物理量分别存储为一个 `DataArray_t`，它们都是 `FlowSolution_t` 的 children。
- 每个 `Zone_t` 下可以有多个 `FlowSolution_t`。

#### `ZoneBC_t`

#### `ZoneGridConnectivity_t`

## MLL

本节主要内容来自于 MLL 的入门指南 [***A User's Guide to CGNS***](https://cgns.github.io/CGNS_docs_current/user/)：

- 原文主要介绍 Fortran-API，这里（平行地）介绍 C-API ，以便 C/C++ 用户参考。完整的 API 参见 [***Mid-Level Library***](https://cgns.github.io/CGNS_docs_current/midlevel/)。⚠️ C 的多维数组 *按行* 存储，Fortran 的多维数组 *按列* 存储，因此 *C 的行* 对应于 *Fortran 的列*。
- 原文采用了 先具体介绍 *结构 (structured) 网格*、再简要介绍 *非结构 (unstructured) 网格* 的展开方式，这里则将二者同步展开，以便读者比较二者的异同。

下载或克隆 [CGNS 代码库](https://github.com/CGNS/CGNS) 后，可在 `${SOURCE_DIR}/src/Test_UserGuideCode/` 中找到所有示例源文件。源文件头部的注释给出了各示例的独立构建方式；若要批量构建所有示例，可以在 CMake 中勾选 `CGNS_BUILD_TESTING`，这样生成的可执行文件位于 `${BUILD_DIR}/src/Test_UserGuideCode/` 中。

### 单区网格

***单区 (single-zone) 网格*** 是最简单的网格，也是任意复杂网格的基本组成单位。

|      |      结构网格      |     非结构网格      |
| :--: | :----------------: | :-----------------: |
| 写出 | `write_grid_str.c` | `write_grid_unst.c` |
| 读入 | `read_grid_str.c`  | `read_grid_unst.c`  |

#### 通用操作

```c
// Open a CGNS file:
ier = cg_open(
    char *file_name,
    int mode/* CG_MODE_WRITE | CG_MODE_READ | CG_MODE_MODIFY */,
    // output:
    int *file_id);
// Close a CGNS file:
ier = cg_close(int file_id);

// Stop the execution of the program:
void cg_error_exit();

// Create and/or write to a CGNS base node:
ier = cg_base_write(
    int file_id, char *base_name, int cell_dim, int phys_dim,
    // output:
    int *base_id);
// Read CGNS base information:
ier = cg_base_read(
    int file_id, int base_id,
    // output:
    char *base_name, int *cell_dim, int *phys_dim);

// Create and/or write to a zone node:
ier = cg_zone_write(
    int file_id, int base_id, char *zone_name, cgsize_t *zone_size,
    ZoneType_t zone_type/* CGNS_ENUMV(Structured) |
                           CGNS_ENUMV(Unstructured) */,
    // output:
    int *zone_id);
// Read zone information:
ier = cg_zone_read(
    int file_id, int base_id, int zone_id,
    // output:
    char *zone_name, cgsize_t *zone_size);
```

其中

- 用于新建对象的函数 `cg_open()` 或 `*_write()` 总是以 `int` 型的 `id` 作为返回值。此 `id` 可以被后续代码用来访问该对象。
- `cell_dim`、`phys_dim` 分别表示 单元（流形）维数、物理（空间）维数。
- `zone_size` 是一个二维数组（的头地址），
  - 其行数为三，各行分别表示 结点、单元、边界点 数量。
  - 对于 结构网格：
    - 列数 至少为 空间维数，各列分别对应于三个（逻辑）方向的数量。
    - 各方向的 单元数 总是比 结点数 少一个。
    - 边界点 没有意义，因此最后一行全部为零。
  - 对于 非结构网格：
    - 列数 至少为 一。
    - 若单元编号没有排序，则边界点数量为零。

#### 读写坐标

```c
// Write grid coordinates:
ier = cg_coord_write(
    int file_id, int base_id, int zone_id,
    DataType_t data_type/* CGNS_ENUMV(RealDouble) */,
    char *coord_name/* "CoordinateX" */,
    void *coord_array,
    // output:
    int *coord_id);
// Read grid coordinates:
ier = cg_coord_read(
    int file_id, int base_id, int zone_id,
    char *coord_name, DataType_t data_type,
    cgsize_t *range_min, cgsize_t *range_max,
    // output:
    void *coord_array);
// Get info for an element section:
```

其中

- 设结点总数为 `N`，则函数 `cg_coord_write()` 写出的是以 `coord_array` 为头地址的前 `N` 个元素。
  - 对于结构网格，`coord_array`  通常声明为多维数组，此时除第一维长度 *至少等于* 该方向的结点数外，其余维度的长度 *必须等于* 相应方向的结点数。
  - 对于非结构网格，`coord_array`  通常声明为长度不小于 `N` 的一维数组。
- 坐标名 `coord_name` 必须取自 [*SIDS-standard names*](https://cgns.github.io/CGNS_docs_current/sids/dataname.html)，即 `CoordinateX`、`CoordinateY`、`CoordinateZ`。
- `data_type` 应当与 `coord_array` 的类型匹配：
  - `CGNS_ENUMV(RealSingle)` 对应 `float`。
  - `CGNS_ENUMV(RealDouble)` 对应 `double`。

#### 读写单元

结构网格的 *结点信息* 已经隐含了 *单元信息*，因此不需要显式创建单元。与之相反，非结构网格的单元信息需要显式给出：

```c
// Write fixed size element data:
ier = cg_section_write(
    int file_id, int base_id, int zone_id,
    char *section_name, ElementType_t element_type,
    cgsize_t first, cgsize_t last, int n_boundary, cgsize_t *elements,
    // output:
    int *section_id);
// Get number of element sections:
ier = cg_nsections(
    int file_id, int base_id, int zone_id,
    // output:
    int *n_section);
// Get info for an element section:
ier = cg_section_read(
    int file_id, int base_id, int zone_id, int section_id,
    // output:
    char *section_name, ElementType_t *element_type,
    cgsize_t *first, cgsize_t *last, int *n_boundary,
    int *parent_flag);
// Read fixed size element data:
ier = cg_elements_read(
    int file_id, int base_id, int zone_id, int section_id,
    // output:
    cgsize_t *elements, cgsize_t *parent_data);
```

其中

- `cg_section_write()` 在给定的 `Zone_t` 对象下新建一个 `Elements_t` 对象，即 *单元片段 (element section)*。
- 一个 `Zone_t` 下可以有多个 `Elements_t`：
  - 同一个  `Zone_t` 下的所有单元（含所有维数）都必须有 *连续* 且 *互异* 的编号。
  - 同一个 `Elements_t` 下的所有单元必须是同一种类型，即 `element_type`，并且必须是枚举类型 [`ElementType_t`](file:///Users/master/code/mesh/cgns/doc/midlevel/general.html#typedefs) 的有效值之一，例如 `NODE`、`BAR_2`、`TRI_3`、`QUAD_4`、`TETRA_4`、`PYRA_5`、`PENTA_6`、`HEXA_8`。
- `first`、`last` 为当前 `Elements_t` 内首、末单元的编号。
- `n_boundary` 为当前 `Elements_t` 的最后一个边界点的编号。如果单元编号没有排序，则设为 `0`。
- `parent_flag` 用于判断 parent data 是否存在。

#### 运行示例

结构网格：

```shell
> ${BUILD_DIR}/src/Test_UserGuideCode/C_code/write_grid_str

Program write_grid_str

created simple 3-D grid points
Successfully wrote grid to file grid_c.cgns
> ${BUILD_DIR}/src/Test_UserGuideCode/C_code/read_grid_str

Successfully read grid from file grid_c.cgns
  For example, zone 1 x,y,z[8][16][20]= 20.000000, 16.000000, 8.000000

Program successful... ending now
```

非结构网格：

```shell
> ${BUILD_DIR}/src/Test_UserGuideCode/C_code/write_grid_unst

Program write_grid_unst

created simple 3-D grid points

Successfully wrote unstructured grid to file grid_c.cgns
> ${BUILD_DIR}/src/Test_UserGuideCode/C_code/read_grid_unst

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

### 流场数据

#### 结点数据

|      |       结构网格       |      非结构网格       |
| :--: | :------------------: | :-------------------: |
| 写出 | `write_flowvert_str` | `write_flowvert_unst` |
| 读入 | `read_flowvert_str`  | `read_flowvert_unst`  |

新增的 API 如下：

```c
// Create and/or write to a `FlowSolution_t` node:
ier = cg_sol_write(
    int file_id, int base_id, int zone_id, char *sol_name,
    GridLocation_t location/* CGNS_ENUMV(Vertex) */,
    // output:
    int *sol_id);
// Get info about a `FlowSolution_t` node:
ier = cg_sol_info(
    int file_id, int base_id, int zone_id, int sol_id,
    // output:
    char *sol_name, GridLocation_t *location);

// Write flow solution:
ier = cg_field_write(
    int file_id, int base_id, int zone_id, int sol_id,
    DataType_t datatype, char *field_name, void *sol_array,
    // output:
    int *field_id);
// Read flow solution:
ier = cg_field_read(
    int file_id, int base_id, int zone_id, int sol_id,
    char *field_name, DataType_t data_type,
    cgsize_t *range_min, cgsize_t *range_max,
    // output:
    void *sol_array);
```

其中

- `cg_sol_write()` 用于在 `Zone_t` 对象下创建一个表示 *一组物理量* 的 `FlowSolution_t` 对象。
  - 同一个  `Zone_t` 下的 `FlowSolution_t` 可以有多个。
  - 所有 `FlowSolution_t` 都平行于 表示网格的 `GridCoordinates_t`。
- `cg_field_write()` 用于在 `FlowSolution_t` 对象下创建一个表示 *单个物理量* 的对象，例如  `DataArray_t`、`Rind_t`。
  - `sol_array` 尺寸应当与结点数量匹配：对于结构网格，通常声明为多维数组；对于非结构网格，通常声明为一位数组。
  - `field_name` 应当取自 [*SIDS-standard names*](https://cgns.github.io/CGNS_docs_current/sids/dataname.html)，例如 `Density`、`Pressure`。

#### 单元数据

`write_flowcent_str.c` 与 `read_flowcent_str.c` 展示了这种流场表示方法，所用 API 与前一小节几乎完全相同，只需注意：

- 在调用 `cg_sol_write()` 时，将 `location` 的值由 `CGNS_ENUMV(Vertex)` 改为 `CGNS_ENUMV(CellCenter)`。
- 在结构网格的各逻辑方向上，用于存放数据的多维数组的长度必须与单元数量协调。

#### 外皮数据

***外皮 (rind) 数据*** 是指存储在网格表面的一层或多层 *影子 (ghost) 单元* 上的数据 ：

```
┌───╔═══╦═══╦═══╗───┬───┐      ═══ 网格单元
│ o ║ o ║ o ║ o ║ o │ o │
└───╚═══╩═══╩═══╝───┴───┘      ─── 影子单元
```

`write_flowcentrind_str.c` 与 `read_flowcentrind_str.c` 展示了这种表示方法，新增的 API 如下：

```c
/* API in `write_flowcentrind_str.c` and `read_flowcentrind_str.c` */

// Access a node via [label|name]-index pairs:
ier = cg_goto(int file_id, int base_id, ..., "end");
// e.g.
ier = cg_goto(
    file_id, base_id,
    "Zone_t", zone_id,
    "FlowSolution_t", sol_id,
    "end");

// Number of rind layers for each direction (structured grid):
int rind_data[6] = {
  1/* i_low */,
  1/* i_high */,
  1/* j_low */,
  1/* j_high */,
  0/* k_low */,
  0/* k_high */
};
// Write number of rind layers:
ier = cg_rind_write(int *rind_data);
// Read number of rind layers:
ier = cg_rind_read(int *rind_data);
```

其中

- `cg_goto()` 用于定位将要创建 `Rind_t` 对象的那个 `FlowSolution_t` 对象。
- 外皮数据存储在（根据影子单元层数）扩充的流场数组中，因此在结构网格的各逻辑方向上，用于存放数据的多维数组的长度必须与 *扩充后的* 单元数量协调。

### 边界条件

#### 结构网格

两种 BC 表示方法：

- `PointRange` 通过 *指定结点编号范围* 来确定边界，因此只适用于 *结构网格的长方形* 边界。`write_bc_str.c` 与 `read_bc_str.c` 展示了这种方法。
- `PointList` 通过 *指定结点编号列表* 来确定边界，因此适用于 *所有* 边界。`write_bcpnts_str.c` 与 `read_bcpnts_str.c` 展示了这种方法。

尽管本节标题为  *结构网格*，但上述方法也可以用于 *非结构网格*，只是后者有更简单的方法（见下一节）。

```c
/* API in `write_bc_str.c`    and `read_bc_str.c`
      and `write_bcpnt_str.c` and `read_bcpnt_str.c` */

// Write boundary condition type and data:
ier = cg_boco_write(
    int file_id, int base_id, int zone_id, char *boco_name,
    BCType_t boco_type/* CGNS_ENUMV(BCType_t) */,
    PointSetType_t point_set_type/* CGNS_ENUMV(PointRange) |
                                    CGNS_ENUMV(PointList) */,
    cgsize_t n_point, cgsize_t *point_set,
    // output:
    int *boco_id);

// Get number of boundary condition in zone:
ier = cg_nbocos(
    int file_id, int base_id, int zone_id,
    // output:
    int *n_boco);

// Get boundary condition info:
ier = cg_boco_info(
    int file_id, int base_id, int zone_id, int boco_id,
    // output:
    char *boco_name, BCType_t *boco_type,
    PointSetType_t *point_set_type, cgsize_t *n_point,
    int *normal_id,
    cgsize_t *normal_list_size,
    DataType_t *normal_data_type,
    int *n_data_set);

// Read boundary condition data and normals:
ier = cg_boco_read(
    int file_id, int base_id, int zone_id, int boco_id,
    // output:
    cgsize_t *point_set, void *normal_list);
```

其中

- `cg_boco_write()` 用于创建一个表示具体边界条件的 `BC_t` 对象。
  - 每个 `BC_t` 都是某个 `IndexRange_t` 或 `IndexArray_t`  的 parent。
  - 所有 `BC_t` 都是同一个 `ZoneBC_t` 的 child(ren)。
  - 这个唯一的 `ZoneBC_t`是 `Zone_t` 的 child，因此是 `GridCoordinates_t` 及 `FlowSolution_t` 的 sibling。
- `boco_type` 的取值必须是枚举类型 `BCType_t` 的有效值，例如 `BCWallInviscid`、`BCInflowSupersonic`、`BCOutflowSubsonic`，完整列表参见 [***Boundary Condition Type Structure Definition***](https://cgns.github.io/CGNS_docs_current/sids/bc.html#BCType)。
- 二维数组 `point_set` 用于指定结点编号，其行数（至少）为 `n_point`。
  - 对于结构网格，`point_set` 的列数 为 空间维数，而 `n_point`
    - 为 `2`，若 `point_set_type` 为 `CGNS_ENUMV(PointRange)`。此时 `point_set` 的第一、二行分别表示编号的下界、上界。
    - 为 此边界的结点总数，若 `point_set_type` 为 `CGNS_ENUMV(PointList)`。
  - 对于非结构网格，`point_set` 的列数为 `1`，而 `n_point`
    - 为 此边界的结点总数，且 `point_set_type` 只能为 `CGNS_ENUMV(PointList)`。

#### 非结构网格

尽管 *非结构网格* 可以像 *结构网格* 那样，通过指定边界上的 *结点* 来施加边界条件，但利用 [*读写单元*](#读写单元) 时创建的 `Element_t` 对象来指定边界上的 *单元* 通常会更加方便。`write_bcpnts_unst.c` 与 `read_bcpnts_unst.c` 展示了这种方法，主要的 API 如下：

```c
/* API `write_bcpnt_unst.c` and `read_bcpnt_unst.c` */

// Write boundary condition type and data:
ier = cg_boco_write(
    int file_id, int base_id, int zone_id, char *boco_name,
    BCType_t boco_type/* CGNS_ENUMV(BCType_t) */,
    PointSetType_t point_set_type/* CGNS_ENUMV(PointRange) |
                                    CGNS_ENUMV(PointList) */,
    cgsize_t n_cell, cgsize_t *cell_set,
    // output:
    int *boco_id);

// Write grid location:
ier = cg_gridlocation_write(GridLocation_t grid_location/*
    CGNS_ENUMV(CellCenter) | CGNS_ENUMV(FaceCenter) |
    CGNS_ENUMV(EdgeCenter) | CGNS_ENUMV(Vertex) */);
// Read grid location:
ier = cg_gridlocation_read(GridLocation_t *grid_location);
```

其中

- 这里的 `cg_boco_write()` 在形式上与 *结构网格* 版本一样，只是将 `n_point`、`point_set` 替换成了 `n_cell`、`cell_set`。
- `grid_location == CGNS_ENUMV(FaceCenter)` 表当前 BC 作用在 *面单元* 上，即 `cell_set` 是存储面单元编号的数组。
- 调用 `cg_gridlocation_write()` 之前必须先用 `cg_goto()` 定位到所需的 `BC_t` 对象。

### 多区网格

