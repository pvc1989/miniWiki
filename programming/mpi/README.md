---
title: Message Passing Interface (MPI)
---

# 文件读写

## [Hadoop](https://hadoop.apache.org/)

# 负载平衡

## 静态分配

适用场景：事先知道任务总量及计算资源分布。

### Kernighan--Lin

先将 $V$ 粗略分为 $V_1\cup V_2$，再从二者中寻找子集 $E_1\subset V_1$ 与 $E_2\subset V_2$，使得交换后的总收益最优。

### Graph Coarsening

- 在初始 `Graph` 上为每个 `Vertex` 和 `Edge` 赋权。
- Divide：将一组 `Vertex`s 为一个 `SuperVertex`，相应的 `Edge`s 合并为一个 `SuperEdge`，合并时叠加权重。
- Conquer：先在 `CoarseGraph` 上分割，再到上一级 `FineGraph` 上微调。

### Spectral Partitioning

- Graph Laplacian：$ \Mat{C}^{\mathsf{T}}\cdot\Mat{C}=\Mat{D}-\Mat{W} $
- 质量弹簧系统：$ \Mat{M}\cdot\Mat{\ddot{U}}+\Mat{K}\cdot\Mat{U}=\Mat{0} $

### METIS

[Karypis/METIS](https://github.com/KarypisLab/METIS) 是由 [George Karypis](http://glaros.dtc.umn.edu) 开发的网格分区工具包。
为便于导入 CMake 项目，[pvcStillinGradSchool/METIS](https://github.com/pvcStillinGradSchool/METIS) 在其基础上对 `CMakeLists.txt` 做了部分修改。

某些编译器会将『指针与数据格式不一致』视为错误，此时可在 `CMakeCache.txt` 中开启如下编译选项：

```cmake
//Flags used by the C compiler during all build types.
CMAKE_C_FLAGS:STRING=-Wno-error=format
```

本节代码中的
- `idx_t` 是一种长度由预定义宏 `IDXTYPEWIDTH` 确定的整数类型。
- `real_t` 是一种长度由预定义宏 `REALTYPEWIDTH` 确定的浮点类型。

⚠️ 使用时，应确保上述类型与业务代码中的相应类型保持一致。

公共接口：

```c
int METIS_PartMeshDual(
    idx_t *n_elems,
    idx_t *n_nodes,
    idx_t *range_of_each_elem,
    idx_t *nodes_in_each_elem,
    idx_t *cost_of_each_elem = NULL,  /* computational cost */
    idx_t *size_of_each_elem = NULL,  /* communication size */
    idx_t *n_common_nodes,
    idx_t *n_parts,
    real_t *weight_of_each_part = NULL,  /* sum must be 1.0 */
    idx_t *options = NULL,
    idx_t *objective_value,  /* edge cut or communication volume */
    idx_t *elem_parts,
    idx_t *node_parts
);
int METIS_MeshToDual(
    idx_t *n_elems,
    idx_t *n_nodes,
    idx_t *range_of_each_elem,
    idx_t *nodes_in_each_elem,
    idx_t *n_common_nodes,
    idx_t *index_base,  /* 0 or 1 */
    idx_t **range_of_each_dual_vertex,
    idx_t **neighbors_of_each_dual_vertex
);
int METIS_PartGraphKway(  // or METIS_PartGraphRecursive
    idx_t *n_nodes,
    idx_t *n_constraints,  /* number of balancing constraints, >= 1 */
    idx_t *range_of_each_node,
    idx_t *neighbors_of_each_node,
    idx_t *cost_of_each_node = NULL,  /* computational cost */
    idx_t *size_of_each_node = NULL,  /* communication size */
    idx_t *cost_of_each_edge = NULL,  /* weight of each edge */
    idx_t *n_parts,
    real_t *weight_of_each_part = NULL,  /* sum must be 1.0 for each constraint */
    real_t *unbalances = NULL,  /* unbalance tolerance, 1.001 for NULL */
    idx_t *options = NULL,
    idx_t *objective_value,  /* edge cut or communication volume */
    idx_t *parts
);
```

实现细节：

```c++
void CreateGraphDual(
    idx_t n_elems,
    idx_t n_nodes,
    idx_t *range_of_each_elem,
    idx_t *nodes_in_each_elem,
    idx_t n_common_nodes,
    idx_t **range_of_each_dual_vertex,
    idx_t **neighbors_of_each_dual_vertex
);
idx_t FindCommonElements(
    idx_t i_curr_elem,
    idx_t n_nodes_in_curr_elem,
    idx_t *nodes_in_curr_elem,
    idx_t *range_of_each_node,
    idx_t *elems_in_each_node,
    idx_t *range_of_each_elem,
    idx_t n_common_nodes,
    idx_t *n_visits_of_each_elem,
    idx_t *neighbors_of_curr_elem) {
  idx_t n_neighbors = 0;
  /* find all elements that share at least one node with i_curr_elem */
  for (idx_t i_node_local = 0;
      i_node_local < n_nodes_in_curr_elem; i_node_local++) {
    // for each nodes in curr elem
    idx_t i_node_global = nodes_in_curr_elem[i_node_local];
    idx_t i_elem_begin = range_of_each_node[i_node_global];
    idx_t i_elem_end = range_of_each_node[i_node_global+1];
    for (idx_t i_elem_curr = i_elem_begin;
        i_elem_curr < i_elem_end; i_elem_curr++) {
      // for each elems in curr node
      i_elem_global = elems_in_each_node[i_elem_curr];
      if (n_visits_of_each_elem[i_elem_global] == 0) {
        // find a new neighbor (elem) of curr elem
        neighbors_of_curr_elem[n_neighbors++] = i_elem_global;
      }
      n_visits_of_each_elem[i_elem_global]++;
    }
  }
  /* put i_curr_elem into the neighbor list (in case it is not there) so that it
     will be removed in the next step */
  if (n_visits_of_each_elem[i_curr_elem] == 0)
    neighbors_of_curr_elem[n_neighbors++] = i_curr_elem;
  n_visits_of_each_elem[i_curr_elem] = 0;
  /* compact the list to contain only those with at least n_common_nodes nodes */
  idx_t n_real_neighbors = 0;
  for (idx_t i_neighbor_local = 0;
      i_neighbor_local < n_neighbors;
      i_neighbor_local++) {  // for each (possibly trivial) neighbor (elem)
    idx_t i_neighbor_global = neighbors_of_curr_elem[i_neighbor_local];
    idx_t n_visits_of_curr_elem = n_visits_of_each_elem[i_neighbor_global];
    if (/* trivial case */n_visits_of_curr_elem >= n_common_nodes ||
        /* In case when (n_common_nodes >= n_nodes_in_curr_elem). */
        n_visits_of_curr_elem >= (n_nodes_in_curr_elem - 1) ||
        /* In case when (n_common_nodes >= n_nodes_in_curr_neighbor). */
        n_visits_of_curr_elem >= range_of_each_elem[i_neighbor_global+1]
                               - range_of_each_elem[i_neighbor_global] - 1)
      neighbors_of_curr_elem[n_real_neighbors++] = i_neighbor_global;
    n_visits_of_each_elem[i_neighbor_global] = 0;
  }
  return n_real_neighbors;
}
```

## 动态分配

所有闲置进程也排成一个 Queue：当一个进程完成现有的工作，从工作状态中转入闲置状态时，就将该进程 从工作进程的队列中删除，放到闲置进程的队列中，并将它请求任务的申请排上日程。这样，所有进程在工作和空闲状态之间转换，轮流获取任务。

### [Slurm](https://slurm.schedmd.com/)

# 稀疏矩阵

## 存储格式

### CSR: Compressed Sparse Row

### CSC: Compressed Sparse Column

### DOK: Dictionary Of Keys

## [`scipy.sparse`](https://docs.scipy.org/doc/scipy/reference/sparse.html)

## PETSc

### 功能模块

![](https://docs.petsc.org/en/latest/_images/library_structure.svg)

### `IS`: Index Sets
### `Vec`: Vectors
### `Mat`: Matrices

#### Partitioning

```c
MatCreateMPIAdj(/*
  Creates a sparse matrix representing an adjacency list. */
    MPI_Comm comm,
    int n_rows_local,
    PetscInt n_cols_global,
    const PetscInt ia[]/* row pointers in CSR format */,
    const PetscInt ja[]/* col pointers in CSR format */,
    PetscInt *weights/* [optional] edge weights */,
  /* output: */
    Mat *adj/*  */
);
MatPartitioningCreate(/*
  Creates a partitioning context. */
    MPI_Comm comm,
  /* output: */
    MatPartitioning *part
);
MatPartitioningSetAdjacency(/*
  Sets the adjacency graph (matrix) of the thing to be partitioned. */
    MatPartitioning part,
    Mat Adj
);
MatPartitioningSetFromOptions(
    MatPartitioning part
);
MatPartitioningApply(/*
  Gets a partitioning for a graph (matrix). */
    MatPartitioning part,
  /* output: */
    IS *is/* rank for each local vertex */
);
MatPartitioningDestroy(
    MatPartitioning *part
);
MatDestroy(
    Mat *Adj
);
ISPartitioningToNumbering(/*
  Gets an IS that contains a new global number for each local vertex. */
    IS is/* rank for each local vertex */,
  /* output: */
    IS *isg/* new global id for each local vertex */
);
AOCreateBasicIS(
    isg,
    NULL,
    &ao
);
AOPetscToApplication(
);
AOApplicationToPetsc(
);
```



### `KSP`: Linear System Solvers
### `SNES`: Nonlinear Solvers

### `TS`: Time Steppers

### `DM`: Domain Management

# 参考资料

- 实现
  - [MPICH](http://www.mpich.org/)
  - [Open MPI](https://www.open-mpi.org/)
- 教程
  - 《[Message Passing Interface (MPI)](https://computing.llnl.gov/tutorials/mpi/)》 by *Blaise Barney, Lawrence Livermore National Laboratory*
  - 《[MPI 并行编程讲稿](ftp://ftp.cc.ac.cn/pub/home/zlb/bxjs/bxjs.pdf)》张林波 2012

