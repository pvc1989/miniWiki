# CGNS

## 术语表

|      术语      |                             含义                             |
| :------------: | :----------------------------------------------------------: |
|      CGNS      |    [CFD General Notation System](http://cgns.github.io/)     |
|      SIDS      | [Standard Interface Data Structures](http://cgns.github.io/CGNS_docs_current/sids/) |
| SIDS-compliant |             符合 SIDS 定义的 CGNS 文件（数据库）             |
|      MLL       | [Mid-Level Library](http://cgns.github.io/CGNS_docs_current/midlevel/) |
|      API       |              Application Programming Interface               |


## 引言

### 数据结构

CGNS 文件（数据库）
- 在物理上是按二进制存储的，因此无法用普通的文本编辑器（安全地）读写。
- 在逻辑上是一棵由若干 `Node`(s) 相互链接而组成的 `Tree`，每个 `Node` 都含有 `Name`、`Label`、`Data` 三个基本部分，可以用 [`cgnsview`](http://cgns.github.io/CGNS_docs_current/cgnstools/cgnsview/index.html) 安全地读写。



## MLL 入门

本节主要内容来自于 CGNS/MLL 的入门指南 [*A User's Guide to CGNS*](http://cgns.github.io/CGNS_docs_current/user/index.html)：
- 原文主要介绍 Fortran-API，这里（平行地）介绍 C-API ，以便 C/C++ 用户参考。⚠️ C 的多维数组 *按行* 存储，Fortran 的多维数组 *按列* 存储，因此 *C 的行* 对应于 *Fortran 的列*。
- 原文采用了 先具体介绍 *结构 (structured) 网格*、再简要介绍 *非结构 (unstructured) 网格* 的展开方式，这里则将二者同步展开，以便读者比较二者的异同。

下载或克隆 [CGNS 代码库](https://github.com/CGNS/CGNS) 后，可在 `${SOURCE_DIR}/src/Test_UserGuideCode/` 中找到所有示例源文件。源文件头部的注释给出了各示例的独立构建方式；若要批量构建所有示例，可以在 CMake 中勾选 `CGNS_BUILD_TESTING`，这样生成的可执行文件位于 `${BUILD_DIR}/src/Test_UserGuideCode/` 中。

### 单区网格

*单区 (single-zone) 网格* 是最简单的网格，也是任意复杂网格的基本组成单位。

|        | 结构网格 | 非结构网格 |
| :----: | :--------------------: | :------------------------: |
|  写出  |   `write_grid_str.c`   |    `write_grid_unst.c`     |
| 读入 |   `read_grid_str.c`    |     `read_grid_unst.c`     |

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
- 坐标名 `coord_name` 必须取自 [*SIDS-standard names*](http://cgns.github.io/CGNS_docs_current/sids/dataname.html)，即 `CoordinateX`、`CoordinateY`、`CoordinateZ`。
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
  - `field_name` 应当取自 [*SIDS-standard names*](http://cgns.github.io/CGNS_docs_current/sids/dataname.html)，例如 `Density`、`Pressure`。

#### 单元数据

`write_flowcent_str.c` 与 `read_flowcent_str.c` 展示了这种流场表示方法，所用 API 与前一小节几乎完全相同，只需注意：

- 在调用 `cg_sol_write()` 时，将 `location` 的值由 `CGNS_ENUMV(Vertex)` 改为 `CGNS_ENUMV(CellCenter)`。
- 在结构网格的各逻辑方向上，用于存放数据的多维数组的长度必须与单元数量协调。

#### 外层数据

*外层 (rind) 数据* 是指存储在网格表面的一层或多层 *影子 (ghost) 单元* 上的数据 ：

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
- 外层数据存储在（根据影子单元层数）扩充的流场数组中，因此在结构网格的各逻辑方向上，用于存放数据的多维数组的长度必须与 *扩充后的* 单元数量协调。

### 边界条件

`BCType_t` 是一个枚举对象，所有具体的 BC 都是它的成员，完整列表参见 [*Boundary Condition Type Structure Definition*](http://cgns.github.io/CGNS_docs_current/sids/bc.html#BCType)。

两种 BC 表示方法：
- `PointRange` 通过 *指定结点编号范围* 来确定边界，因此只适用于 *结构网格的长方形* 边界。`write_bc_str.c` 与 `read_bc_str.c` 展示了这种方法。
- `PointList` 通过 *指定结点编号列表* 来确定边界，因此适用于 *所有* 边界。`write_bcpnts_str.c` 与 `read_bcpnts_str.c` 展示了这种方法。

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
- 二维数组 `point_set` 用于指定结点编号，其行数（至少）为 `n_point`。
  - 对于结构网格，`point_set` 的列数 为 空间维数，而 `n_point`
    - 为 `2`，若 `point_set_type` 为 `CGNS_ENUMV(PointRange)`。此时 `point_set` 的第一、二行分别表示编号的下界、上界。
    - 为 此边界的结点总数，若 `point_set_type` 为 `CGNS_ENUMV(PointList)`。
  - 对于非结构网格，`point_set` 的列数为 `1`，而 `n_point`
    - 为 此边界的结点总数，且 `point_set_type` 只能为 `CGNS_ENUMV(PointList)`。

### 多区网格

### 

