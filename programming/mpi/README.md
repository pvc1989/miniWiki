---
title: Message Passing Interface (MPI)
---

# 文件读写

## [Hadoop](https://hadoop.apache.org/)

# 负载平衡

## 静态分配

前提：事先知道 任务总量、计算资源分布情况。

### Kernighan--Lin

先将 $V$ 粗略分为 $V_1\cup V_2$，再从二者中寻找子集 $E_1\subset V_1$ 与 $E_2\subset V_2$，使得交换后的总收益最优。

### Graph Coarsening

- 在初始 `Graph` 上为每个 `Vertex` 和 `Edge` 赋权。
- Divide：将一组 `Vertex`s 为一个 `SuperVertex`，相应的 `Edge`s 合并为一个 `SuperEdge`，合并时叠加权重。
- Conquer：先在 `CoarseGraph` 上分割，再到上一级 `FineGraph` 上微调。

### Spectral Partitioning

- Graph Laplacian：$ \Mat{C}^{\mathsf{T}}\Mat{C}=\Mat{D}-\Mat{W} $
- 质量弹簧系统：$ \Mat{M}\ket{\ddot{U}}+\Mat{K}\ket{U}=\ket{0} $

### [METIS](http://glaros.dtc.umn.edu/gkhome/views/metis)

```c
index_t FindCommonElements(
    index_t i_curr_elem,
    index_t n_nodes_in_curr_elem,
    index_t *nodes_in_curr_elem,
    index_t *range_of_each_node,
    index_t *elems_in_each_node,
    index_t *range_of_each_elem,
    index_t n_common_nodes,
    index_t *n_visits_of_each_elem,
    index_t *neighbors_of_curr_elem) {
  index_t n_neighbors = 0;
  /* find all elements that share at least one node with i_curr_elem */
  for (index_t i_node_local = 0;
      i_node_local < n_nodes_in_curr_elem; i_node_local++) {
    // for each nodes in curr elem
    index_t i_node_global = nodes_in_curr_elem[i_node_local];
    index_t i_elem_begin = range_of_each_node[i_node_global];
    index_t i_elem_end = range_of_each_node[i_node_global+1];
    for (index_t i_elem_curr = i_elem_begin;
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
  index_t n_real_neighbors = 0;
  for (index_t i_neighbor_local = 0;
      i_neighbor_local < n_neighbors;
      i_neighbor_local++) {  // for each (possibly trivial) neighbor (elem)
    index_t i_neighbor_global = neighbors_of_curr_elem[i_neighbor_local];
    index_t n_visits_of_curr_elem = n_visits_of_each_elem[i_neighbor_global];
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

所有闲置进程也排成一个 Queue：当一个 进程完成现有的工作，从工作状态中转入闲置状态时，就将该进程 从工作进程的队列中删除，放到闲置进程的队列中，并将它请求任 务的申请排上日程。这样，所有进程在工作和空闲状态之间转换，轮 流获取任务。

### [Slurm](https://slurm.schedmd.com/)

# 参考资料

- 实现
  - [MPICH](http://www.mpich.org/)
  - [Open MPI](https://www.open-mpi.org/)
- 教程
  - 《[Message Passing Interface (MPI)](https://computing.llnl.gov/tutorials/mpi/)》 by *Blaise Barney, Lawrence Livermore National Laboratory*
  - 《[MPI 并行编程讲稿](ftp://ftp.cc.ac.cn/pub/home/zlb/bxjs/bxjs.pdf)》张林波 2012

