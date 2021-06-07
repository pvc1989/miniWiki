---
title: Message Passing Interface (MPI)
---

# 平台搭建

## 操作系统

假设有 `3` 台计算机，分别在其上[安装 Linux 发行版](../linux/install/README.md)，例如 [Ubuntu 20.04+](https://ubuntu.com/download/desktop)。

### 局域网

搭建如下局域网：

| 编号 | 主机名  | 管理员  | 静态 IP 地址  |
| :--: | :-----: | :-----: | :-----------: |
| `1`  | `host1` | `admin` | `192.168.5.1` |
| `2`  | `host2` | `admin` | `192.168.5.2` |
| `3`  | `host3` | `admin` | `192.168.5.3` |

为方便后续操作，在各台主机的 `/etc/hosts` 文件中（用 `sudo` 权限）加入以下三行：

```
192.168.5.1    host1
192.168.5.2    host2
192.168.5.3    host3
```

完成后，可在各台主机上运行以下三行，以验证两两连通：

```shell
ping -c 4 host1
ping -c 4 host2
ping -c 4 host3
```

### 同名账户

在各台主机上分别创建名为 `common` 的用户，并赋予其 `root` 权限（加入 `sudo` 用户组）：

```shell
sudo adduser common  # 根据提示，先输入 admin 的密码，
                     # 再创建 common 的密码，并输入个人信息（可选）
sudo usermod -aG sudo common
```

完成后，切换到 `common` 用户，并验证权限设置正确：

```shell
su - common  # 需输入 common 的密码
sudo whoami  # 需输入 common 的密码，应返回 root
```

## 免密互访

### 服务端

在各台主机上分别安装并开启 [SSH](../linux/ssh.md) 服务：

```shell
sudo apt install openssh-server
# 开启 SSH 服务：
sudo systemctl start  ssh
sudo systemctl enable ssh
# 检查 SSH 服务的状态：
systemctl status ssh
```

### 客户端

在各台主机上分别生成一对密钥（以 `common@host1` 为例）：

```shell
cd ~
ssh-keygen  # 根据提示，输入密钥的文件名 key1
```

在各台主机上分别用 [`ssh-copy-id`](https://www.ssh.com/academy/ssh/copy-id) 将其公钥写到 `~/.ssh/authorized_keys` 中：

```shell
ssh-copy-id -i ~/.ssh/key1.pub host1  # 根据提示，输入 common@host1 的密码
ssh-copy-id -i ~/.ssh/key1.pub host2  # 根据提示，输入 common@host2 的密码
ssh-copy-id -i ~/.ssh/key1.pub host3  # 根据提示，输入 common@host3 的密码
```

在各台主机上分别开启 SSH 认证代理（以 `common@host1` 为例，每次开启 shell 时均需重做）：

```shell
eval `ssh-agent`
ssh-add ~/.ssh/key1  # 加入私钥（不带 .pub）
```

在各台主机上分别验证免密互访：

```shell
ssh host1 hostname
ssh host2 hostname
ssh host3 hostname
```

若设置成功，则以上三行应分别返回：

```
host1
host2
host3
```

## 公共目录

### 服务端

在 `host1` 上安装并开启 NFS 服务：

```shell
sudo apt install nfs-kernel-server
```

在 `common@host1:~` 中创建并共享 `shared` 目录：

```shell
cd ~
mkdir shared
sudo vim /etc/exports
```

在 `/etc/exports` 文件中（用 `sudo` 权限）加入如下一行：

```
/home/common/shared *(rw,sync,no_root_squash,no_subtree_check)
```

保存后“输出文件系统 (**export** **f**ile **s**ystem)”，并重启 NFS 服务：

```shell
sudo exportfs -a
sudo service nfs-kernel-server restart
```

### 客户端

在 `host2` 上安装 NFS 客户端：

```shell
sudo apt install nfs-common
```

在 `common@host2:~` 中创建同名目录并进行“挂载 (mount)”：

```shell
cd 
sudo mount -t nfs host1:/home/common/shared ~/shared
df -h
```

此后，在 `common@host2:~/shared` 中读写，相当于在 `common@host1:~/shared` 中读写。

为避免重启后手动挂载，可在 `/etc/fstab` 文件中（用 `sudo` 权限）加入如下一行：

```
host1:/home/common/shared /home/common/shared nfs
```

在 `host3` 中重复以上步骤。至此，三台主机共享了 `common@host1:~/shared` 这个目录。

## 编译运行

```shell

```

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

