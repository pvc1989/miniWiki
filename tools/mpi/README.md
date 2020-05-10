# MPI

## 文件读写

[Hadoop](https://hadoop.apache.org/)

## 负载平衡

### 静态分配

前提：事先知道 任务总量、计算资源分布情况。

#### Kernighan--Lin

先将 $V$ 粗略分为 $V_1\cup V_2$，再从二者中寻找子集 $E_1\subset V_1$ 与 $E_2\subset V_2$，使得交换后的总收益最优。

#### Graph Coarsening

- 在初始 `Graph` 上为每个 `Vertex` 和 `Edge` 赋权。
- Divide：将一组 `Vertex`s 为一个 `SuperVertex`，相应的 `Edge`s 合并为一个 `SuperEdge`，合并时叠加权重。
- Conquer：先在 `CoarseGraph` 上分割，再到上一级 `FineGraph` 上微调。

[METIS](http://glaros.dtc.umn.edu/gkhome/views/metis)

#### Spectral Partitioning

- Graph Laplacian：$ \underline{C}^{\mathsf{T}}\underline{C}=\underline{D}-\underline{W} $
- 质量弹簧系统：$\underline{M}\ddot{\left|U\right\rangle}+\underline{K}\left|U\right\rangle=\left|0\right\rangle$


### 动态分配

所有闲置进程也排成一个 Queue：当一个 进程完成现有的工作，从工作状态中转入闲置状态时，就将该进程 从工作进程的队列中删除，放到闲置进程的队列中，并将它请求任 务的申请排上日程。这样，所有进程在工作和空闲状态之间转换，轮 流获取任务。

[Slurm](https://slurm.schedmd.com/)

## 参考资料

- 实现
  - [MPICH](http://www.mpich.org/)
  - [Open MPI](https://www.open-mpi.org/)
- 教程
  - [Message Passing Interface (MPI)](https://computing.llnl.gov/tutorials/mpi/) by *Blaise Barney, Lawrence Livermore National Laboratory*
  - [《MPI 并行编程讲稿》张林波 2012](ftp://ftp.cc.ac.cn/pub/home/zlb/bxjs/bxjs.pdf)

