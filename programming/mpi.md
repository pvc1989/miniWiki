---
title: MPI (Message Passing Interface)
---

# 平台搭建

## 操作系统

假设有 `3` 台可以组网的计算机，分别在其上[安装 Linux 发行版](../linux/install.md)（例如 [Ubuntu](https://ubuntu.com/download/desktop)）。

- 为避免版本不一致带来的麻烦，建议所有主机安装同一版本。
- 由于绝大多数操作可在命令行中完成，建议首次安装后执行以下命令：
  ```shell
  sudo systemctl set-default multi-user.target  # 重启后，以命令行界面启动
  ```

### 局域网

搭建如下局域网：

| 编号 | 主机名  | 管理员  | 静态 IP 地址  |
| :--: | :-----: | :-----: | :-----------: |
| `1`  | `host1` | `admin` | `192.168.5.1` |
| `2`  | `host2` | `admin` | `192.168.5.2` |
| `3`  | `host3` | `admin` | `192.168.5.3` |

为方便后续操作，在各台主机的 `/etc/hosts` 文件中（用 `sudo` 权限）加入以下三行：

```
# 若 host1 等主机名已被占用，则应改用其他主机名
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

在各台主机上分别创建名同名账户（为确保读写权限一致，建议 UID 也相同），并赋予其 `root` 权限（加入 `sudo` 用户组）：

```shell
sudo adduser --uid 2000 mpiuser  # 根据提示，先输入 admin 的密码，
                                 # 再创建 mpiuser 的密码，并输入个人信息（可选）
sudo usermod -aG sudo mpiuser    # 加入 sudo 用户组
```

完成后，切换到 `mpiuser` 用户，并验证权限设置正确：

```shell
su - mpiuser  # 需输入 mpiuser 的密码
sudo whoami   # 需输入 mpiuser 的密码，应返回 root
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

在各台主机的同名账户（以 `mpiuser@host1` 为例）内分别生成一对密钥：

```shell
cd ~/.ssh   # 若不存在，则以 mkdir ~/.ssh 创建之
ssh-keygen  # 根据提示，输入密钥的文件名 key1
```

在各台主机上分别用 [`ssh-copy-id`](https://www.ssh.com/academy/ssh/copy-id) 将其公钥写到 `~/.ssh/authorized_keys` 中：

```shell
ssh-copy-id -i ~/.ssh/key1.pub host1  # 根据提示，输入 mpiuser@host1 的密码
ssh-copy-id -i ~/.ssh/key1.pub host2  # 根据提示，输入 mpiuser@host2 的密码
ssh-copy-id -i ~/.ssh/key1.pub host3  # 根据提示，输入 mpiuser@host3 的密码
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

⚠️ 若不成功，则需开启**认证代理 (authentication agent)**：

```shell
eval `ssh-agent`     # 启动认证代理
ssh-add ~/.ssh/key1  # 加入私钥
```

上述命令可置于 `~/.bashrc` 或 `~/.zshrc` 中，以便每次启动 shell 时自动加载。

## 公共目录

### 服务端

在 `host1` 上安装并开启 NFS 服务：

```shell
sudo apt install nfs-kernel-server
```

在 `mpiuser@host1:~` 中创建并共享 `shared` 目录：

```shell
mkdir ~/shared
sudo vim /etc/exports
```

在 `/etc/exports` 文件中（用 `sudo` 权限）加入如下一行：

```
/home/mpiuser/shared *(rw,sync,no_root_squash,no_subtree_check)
```

保存后**输出文件系统 (*export* *f*ile *s*ystem)**，并重启 NFS 服务：

```shell
sudo exportfs -a
sudo service nfs-kernel-server restart
```

### 客户端

在 `host2` 上安装 NFS 客户端：

```shell
sudo apt install nfs-common
```

在 `mpiuser@host2:~` 中创建同名目录并进行**挂载 (mount)**：

```shell
mkdir ~/shared
sudo mount -t nfs host1:/home/mpiuser/shared ~/shared
df -h
```

此后，在 `mpiuser@host2:~/shared` 中读写，相当于在 `mpiuser@host1:~/shared` 中读写。

为避免重启后手动挂载，可在 `/etc/fstab` 文件中（用 `sudo` 权限）加入如下一行：

```
host1:/home/mpiuser/shared /home/mpiuser/shared nfs
```

在 `host3` 中重复以上步骤。至此，三台主机共享了 `mpiuser@host1:~/shared` 这个目录。

## 编译运行

本节所有操作均在 `mpiuser@host1` 上完成。

### 下载、配置

在 `mpiuser@host1:~/shared` 下创建如下目录树： 

```shell
mkdir ~/shared/mpich
cd ~/shared/mpich
# 代码目录：
wget https://www.mpich.org/static/downloads/3.4.2/mpich-3.4.2.tar.gz
tar xfz mpich-3.4.2.tar.gz
mv mpich-3.4.2 source
# 安装目录：
mkdir install
# 构建目录：
mkdir build
```

进入 `build` 中完成配置：

```shell
# 安装编译器：
sudo apt install build-essential
# 配置构建选项：
cd ~/shared/mpich/build
../source/configure --prefix=/home/mpiuser/shared/mpich/install --with-device=ch3 --disable-fortran 2>&1 | tee c.txt
```

若配置成功，则应当显示以下信息：

```
...
config.status: executing libtool commands
***
*** device configuration: ch3:nemesis
*** nemesis networks: tcp
***
Configuration completed.
```

### 编译、安装

```
make 2>&1 | tee m.txt
make install 2>&1 | tee mi.txt
```

修改环境变量（每次启动 shell 时要重做）：

```shell
PATH=/home/mpiuser/mpich/install/bin:$PATH
export PATH
which mpiexec
```

若设置成功，则应当返回以下路径：

```
/home/mpiuser/mpich/install/bin/mpiexec
```

### 运行、测试

```shell
# host1、host2、host3 分别可以承担 10、20、30 个进程：
echo "host1:10" >> hostlist
echo "host2:20" >> hostlist
echo "host3:30" >> hostlist
# 启动一个需要 60 个进程的任务：
mpiexec -n 60 -f hostlist ./examples/cpi
# 运行测试：
make testing
```

# 负载平衡

## 静态分配（区域分解）<a href id="decomposition"></a>

【区域分解 (domain decomposition)】适用场景：事先知道任务总量及计算资源分布。

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

## 动态分配（任务调度）<a href id="scheduling"></a>

所有闲置进程也排成一个 Queue：当一个进程完成现有的工作，从工作状态中转入闲置状态时，就将该进程从工作进程的队列中删除，放到闲置进程的队列中，并将它请求任务的申请排上日程。这样，所有进程在工作和空闲状态之间转换，轮流获取任务。

### [Slurm](https://slurm.schedmd.com/)

# 通信接口

## 点对点通信

### 非阻塞通信

```c
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
    int target, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Irecv(      void *buf, int count, MPI_Datatype datatype,
    int source, int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Wait(MPI_Request *request, MPI_Status *status);  /* 等到 request 完成时返回 */
int MPI_Waitall(int count, MPI_Request array_of_requests[],
    MPI_Status *array_of_statuses);  /* 等到 array_of_requests 中的所有收发全部完成时返回 */
int MPI_Waitany(int count, MPI_Request array_of_requests[],
    int *index, MPI_Status *status);  /* 等到 array_of_requests 中的任一收发完成时返回 */
int MPI_Test(MPI_Request *request, int *flag/* 若 request 已完成，则设为 true */, MPI_Status *status);
```

## 聚合通信

### Barrier

```c
int MPI_Barrier(MPI_Comm comm);  /* 等到 comm 中的所有进程都运行到此 */
int MPI_Ibarrier(MPI_Comm comm, MPI_Request *request);
```

### Broadcast

```c
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
    int root, MPI_Comm comm);
int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype,
    int root, MPI_Comm comm, MPI_Request *request);
```

### Gather

```c
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
    MPI_Comm comm);
int MPI_Igather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
    MPI_Comm comm, MPI_Request *request);
```

### Scatter

```c
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
    MPI_Comm comm)
int MPI_Iscatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
    MPI_Comm comm, MPI_Request *request)
```

### Reduce

```c
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root,
               MPI_Comm comm)
int MPI_Ireduce(const void *sendbuf, void *recvbuf, int count,
                MPI_Datatype datatype, MPI_Op op, int root,
                MPI_Comm comm, MPI_Request *request)
```

其中 `op` 可以是预定义或自定义的运算：

```c
/* 预定义的运算 */
MPI_MAX
MPI_MIN
MPI_SUM
MPI_PROD
MPI_LAND
MPI_LOR
MPI_LXOR
MPI_BAND
MPI_BOR
MPI_BXOR
MPI_MAXLOC
MPI_MINLOC
/* 自定义的运算 */
typedef void MPI_User_function(
    void *invec, void *inoutvec, int *len, MPI_Datatype *datatype);
int MPI_Op_create(MPI_User_function *function, int commute/* 是否可交换 */,
    MPI_Op *op)
int MPI_Op_free(MPI_Op *op);
/* e.g. */
void user_fn(invec, inoutvec, len, datatype) {
  for (int i = 0; i < len; ++i)
    inoutvec[i] = invec[i] op inoutvec[i];
}
MPI_Op op;
MPI_Op_create(user_fn, commutes, &op);
/* use op */
MPI_Op_free(&op);
```

# 稀疏矩阵

## 存储格式

### CSR: Compressed Sparse Row

### CSC: Compressed Sparse Column

### DOK: Dictionary Of Keys

## [`scipy.sparse`](https://docs.scipy.org/doc/scipy/reference/sparse.html)

## PETSc

### 功能模块

![](https://petsc.org/release/_images/library_structure.svg)

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

# 开发工具

## 性能分析

- [使用 MPI 应用程序运行 perf](http://cn.voidcc.com/question/p-nexvhitv-be.html)

## 单元测试

- [Unit testing with MPI, googletest, and cmake](https://bbanerjee.github.io/ParSim/mpi/c++/mpi-unit-testing-googletests-cmake)
- [An extension to Googletest with MPI](https://github.com/AdhocMan/gtest_mpi)

# 参考资料

- 实现
  - [MPICH](http://www.mpich.org/)
  - [Open MPI](https://www.open-mpi.org/)
- 教程
  - 《[Message Passing Interface (MPI)](https://computing.llnl.gov/tutorials/mpi/)》 by *Blaise Barney, Lawrence Livermore National Laboratory*
  - 《[MPI 并行编程讲稿](ftp://ftp.cc.ac.cn/pub/home/zlb/bxjs/bxjs.pdf)》张林波 2012

