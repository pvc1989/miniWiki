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

[*A User's Guide to CGNS*](http://cgns.github.io/CGNS_docs_current/user/index.html) 是一份 CGNS/MLL 的入门指南。下载或克隆 [CGNS 代码库](https://github.com/CGNS/CGNS) 后，可在 `${SOURCE_DIR}/src/Test_UserGuideCode/` 中找到所有示例源文件。源文件头部的注释给出了各示例的独立构建方式；若要批量构建所有示例，可以在 CMake 中勾选 `CGNS_BUILD_TESTING`，这样生成的可执行文件位于 `${BUILD_DIR}/src/Test_UserGuideCode/` 中。

原始文档主要介绍 Fortran-API，这里（平行地）介绍 C-API ，以便 C/C++ 用户参考。⚠️ C 的多维数组 *按行* 存储，Fortran 的多维数组 *按列* 存储，因此 C 的行 对应于 Fortran 的列。

### 单区 结构网格

`write_grid_str.c` 与 `read_grid_str.c` 展示了利用 MLL 创建、读取 *单区 (single zone) 结构网格* 的方法，用到的 API 如下：

```c
/* API in `write_grid_str.c` and `read_grid_str.c` */

// Open a CGNS file:
ier = cg_open(
    char *file_name,
    int mode/* CG_MODE_WRITE | CG_MODE_READ |
               CG_MODE_MODIFY */,
    int *file_id);
// Close a CGNS file:
ier = cg_close(int file_id);

// Stop the execution of the program:
void cg_error_exit();

// Create and/or write to a CGNS base node:
ier = cg_base_write(
    int file_id, char *base_name,
    int cell_dim, int phys_dim,
    int *base_id);
// Read CGNS base information:
ier = cg_base_read(
    int file_id, int base_id, char *basename,
    int *cell_dim, int *phys_dim);

// Create and/or write to a zone node:
ier = cg_zone_write(
    int file_id, int base_id,
    char *zone_name, cgsize_t *zone_size,
    ZoneType_t zone_type/* CGNS_ENUMV(Structured) */,
    int *zone_id);
// Read zone information:
ier = cg_zone_read(
    int file_id, int base_id, int zone_id,
    char *zone_name, cgsize_t *zone_size);

// Write grid coordinates:
ier = cg_coord_write(
    int file_id, int base_id, int zone_id,
    DataType_t data_type/* CGNS_ENUMV(RealDouble) */,
    char *coord_name/* "CoordinateX" */,
    void *coord_array,
    int *coord_id);
// Read grid coordinates:
ier = cg_coord_read(
    int file_id, int base_id, int zone_id,
    char *coord_name, DataType_t datatype,
    cgsize_t *range_min, cgsize_t *range_max,
    void *coord_array);
```
其中
- 用于新建对象的函数 `cg_open()` 或 `*_write()` 总是以 `int` 型的 `id` 作为返回值。此 `id` 将被后续代码用来访问该对象。
- 对于含有 `X*Y*Z` 个结点的三维结构网格，函数 `cg_coord_write()` 写出的是以 `coord_array` 为头地址的前 `X*Y*Z` 个元素。
- `cell_dim` 表示 单元（流形）维数，`phys_dim` 表示 物理（空间）维数。
- `zone_size` 是一个二维数组的头地址，各行分别表示 结点、单元、边界点的数量。对于三维结构网格：
  - 每行至少需要三列，各列分别对应于三个（逻辑）方向的数量。
  - 各方向的 单元数 总是比 结点数 少一个。
  - 边界点 没有意义，因此最后一行全部为零。
- 坐标名 `coord_name` 必须取自 [*SIDS-standard names*](http://cgns.github.io/CGNS_docs_current/sids/dataname.html)，即 `CoordinateX`、`CoordinateY`、`CoordinateZ`。
- `data_type` 应当与 `coord_array` 的类型匹配：
  - `CGNS_ENUMV(RealSingle)` 对应 `float`。
  - `CGNS_ENUMV(RealDouble)` 对应 `double`。

运行结果：
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

### 单区 结构网格 + 流场

⚠️ 本节示例依赖于前一节输出的 `grid_c.cgns`。

#### Flow Solution at Vertices

`write_flowvert_str.c` 与 `read_flowvert_str.c` 展示了这种流场表示方法，新增的 API 如下：

```c
/* API in `write_flowvert_str.c` and `read_flowvert_str.c` */

// Create and/or write to a `FlowSolution_t` node:
ier = cg_sol_write(
    int file_id, int base_id, int zone_id,
    char *sol_name,
    GridLocation_t location/* CGNS_ENUMV(Vertex) */,
    int *sol_id);
// Get info about a `FlowSolution_t` node:
ier = cg_sol_info(
    int file_id, int base_id, int zone_id,
    int sol_id, char *sol_name,
    GridLocation_t *location);

// Write flow solution:
ier = cg_field_write(
    int file_id, int base_id, int zone_id, int sol_id,
    DataType_t datatype, char *field_name, void *sol_array,
    int *field_id);
// Read flow solution:
ier = cg_field_read(
    int file_id, int base_id, int zone_id, int sol_id,
    char *field_name, DataType_t data_type,
    cgsize_t *range_min, cgsize_t *range_max,
    void *sol_array);
```

其中
- `cg_sol_write()` 用于在 `Zone_t` 对象下创建一个 表示流场的 `FlowSolution_t` 对象。
  - 同一个  `Zone_t` 对象下的 `FlowSolution_t` 对象可以有多个。
  - 所有 `FlowSolution_t` 对象都平行于 表示网格的 `GridCoordinates_t` 对象。
- `cg_field_write()` 用于在 `FlowSolution_t` 对象下创建一个 表示单个物理量的 `DataArray_t` 对象。
- 物理量名称 `field_name` 必须取自 [*SIDS-standard names*](http://cgns.github.io/CGNS_docs_current/sids/dataname.html)，例如 `Density`、`Pressure`。

运行结果：
```shell
> ${BUILD_DIR}/src/Test_UserGuideCode/C_code/write_flowvert_str 

Program write_flowvert_str

created simple 3-D rho and p flow solution

Successfully added Vertex flow solution data to file grid_c.cgns

Note:  if the original CGNS file already had a FlowSolution_t node,
          it has been overwritten
> ${BUILD_DIR}/src/Test_UserGuideCode/C_code/read_flowvert_str 

Successfully read flow solution from file grid_c.cgns
  For example, r,p[8][16][20]= 20.000000, 16.000000

Program successful... ending now
```

#### Flow Solution at Cell Centers

`write_flowcent_str.c` 与 `read_flowcent_str.c` 展示了这种流场表示方法，所用 API 与前一小节几乎完全相同，只需注意：

- 在调用 `cg_sol_write()` 时，将 `location` 的值由 `CGNS_ENUMV(Vertex)` 改为 `CGNS_ENUMV(CellCenter)`。
- 在结构网格的各逻辑方向上，用于存放数据的多维数组的长度必须与单元数量协调。

运行结果：
```shell
> ${BUILD_DIR}/src/Test_UserGuideCode/C_code/write_flowcent_str 

Program write_flowcent_str

created simple 3-D rho and p flow solution

Successfully added CellCenter flow solution data to file grid_c.cgns

Note:  if the original CGNS file already had a FlowSolution_t node,
          it has been overwritten
> ${BUILD_DIR}/src/Test_UserGuideCode/C_code/read_flowcent_str 

Successfully read flow solution from file grid_c.cgns
  For example, r,p[7][15][19]= 19.000000, 15.000000

Program successful... ending now
```

#### Flow Solution at Cell Centers With Additional Rind Data

*外皮 (rind) 数据* 是指存储在网格表面的一层或多层 *影子 (ghost) 单元* 上的数据 ：

```
┌---╔═══╦═══╦═══╗---┬---┐      ═══ 网格单元
╎ o ║ o ║ o ║ o ║ o ╎ o ╎
└---╚═══╩═══╩═══╝---┴---┘      --- 影子单元
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

运行结果：
```shell
> ${BUILD_DIR}/src/Test_UserGuideCode/C_code/write_flowcentrind_str 

Program write_flowcentrind_str

created simple 3-D rho and p flow solution with rind data

Successfully added flow solution data to file grid_c.cgns

Note:  if the original CGNS file already had a FlowSolution_t node,
          it has been overwritten
> ${BUILD_DIR}/src/Test_UserGuideCode/C_code/read_flowcentrind_str 

Successfully read flow solution from file grid_c.cgns
  For example, r,p[7][16][20]= 19.000000, 15.000000
         rind: r,p[7][16][0]= 1055.000000, 1056.000000
         rind: r,p[7][16][21]= -1055.000000, -1056.000000

Program successful... ending now
```

### 单区 结构网格 + 边界条件

### 多区 结构网格

### 单区 非结构网格

### 单区 非结构网格 + 流场

### 单区 非结构网格 + 边界条件

