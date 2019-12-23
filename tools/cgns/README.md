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

#### Vertex-Based Flow Solution
`write_flowvert_str.c` 与 `read_flowvert_str.c` 展示了 *基于结点的 (vertex-based)* 流场表示方法，新增的 API 如下：

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

### 单区 结构网格 + 边界条件

### 多区 结构网格

### 单区 非结构网格

### 单区 非结构网格 + 流场

### 单区 非结构网格 + 边界条件

